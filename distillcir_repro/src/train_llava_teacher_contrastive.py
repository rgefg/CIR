import argparse
import functools
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import webdataset as wds
import webdataset.handlers as wds_handlers
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from distillcir_data import _attach_meta, _select_sample, expand_shards, load_distillcir_jsonl


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return (not is_dist()) or dist.get_rank() == 0


def init_dist(args):
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if os.environ.get("CUDA_VISIBLE_DEVICES") and not args.allow_cuda_visible_devices:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is set; unset it to avoid physical GPU remapping.")
    if torch.cuda.is_available():
        physical_gpus = []
        if args.physical_gpus:
            physical_gpus = [int(item) for item in str(args.physical_gpus).split(",") if item.strip()]
            if len(physical_gpus) < args.world_size:
                raise ValueError(
                    f"--physical-gpus needs at least WORLD_SIZE={args.world_size} ids, got {physical_gpus}"
                )
        device_index = physical_gpus[args.local_rank] if physical_gpus else args.local_rank
        torch.cuda.set_device(device_index)
        args.device = torch.device("cuda", device_index)
        args.physical_gpu = device_index
    else:
        args.device = torch.device("cpu")
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")


def setup_logging(args):
    log_path = Path(args.output_dir) / "teacher_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if args.rank == 0:
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if args.rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_all(seed, rank):
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def jsonable_args(args):
    clean = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = list(value)
        elif isinstance(value, dict):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def collate(samples):
    return {
        "id": [item["id"] for item in samples],
        "images": [item["ref_img"].convert("RGB") for item in samples],
        "instruction": [item["instruction"] for item in samples],
        "modified_caption": [item["modified_caption"] for item in samples],
    }


def build_wds(args):
    id_to_meta = load_distillcir_jsonl(args.cc3m_cir_jsonl, reasoning_jsonl_path=args.reasoning_jsonl)
    shards = expand_shards(args.wds_shards)
    pipeline = [wds.ResampledShards(shards)]
    if args.distributed:
        pipeline.append(wds.split_by_node)
    pipeline.extend(
        [
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds_handlers.warn_and_continue),
            wds.shuffle(args.wds_shuffle),
            wds.decode("pil"),
            wds.rename(image=args.wds_image_key, src_caption=args.wds_text_key, __key__="__key__"),
            wds.select(functools.partial(_select_sample, id_to_meta=id_to_meta)),
            wds.map(functools.partial(_attach_meta, id_to_meta=id_to_meta)),
        ]
    )
    return DataLoader(
        wds.DataPipeline(*pipeline),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )


def load_model_and_processor(args):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32
    processor = AutoProcessor.from_pretrained(args.teacher_model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlavaForConditionalGeneration.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    vision_config = getattr(model.config, "vision_config", None)
    if getattr(processor, "patch_size", None) is None and vision_config is not None:
        processor.patch_size = getattr(vision_config, "patch_size", None)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = getattr(
            model.config, "vision_feature_select_strategy", "default"
        )
    if not getattr(processor, "num_additional_image_tokens", None):
        processor.num_additional_image_tokens = 1

    for param in model.parameters():
        param.requires_grad = False

    if args.use_lora:
        # This environment has bitsandbytes installed without the matching
        # triton.ops package. We train ordinary fp16 LoRA, so keep PEFT from
        # probing the broken bnb backend during adapter injection.
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
        from peft import LoraConfig, get_peft_model

        peft_import_utils.is_bnb_available.cache_clear()
        peft_import_utils.is_bnb_4bit_available.cache_clear()
        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
            modules_to_save=["multi_modal_projector"] if args.train_projector else None,
        )
        model = get_peft_model(model, config)

    if args.train_projector and hasattr(model, "multi_modal_projector"):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    return processor, model


def sync_grads(model):
    if not is_dist():
        return
    world = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world)


def broadcast_model(model):
    if not is_dist():
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)


def gather_targets(features):
    if not is_dist():
        return features
    world = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(features) for _ in range(world)]
    dist.all_gather(gathered, features.contiguous())
    return torch.cat([features] + gathered[:rank] + gathered[rank + 1 :], dim=0)


def contrastive(query, target, temperature):
    all_target = gather_targets(target)
    logits = (query @ all_target.t()) / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels), logits, labels


def last_token_embedding_from_outputs(outputs, attention_mask):
    hidden = outputs.hidden_states[-1]
    last_index = attention_mask.sum(dim=1).sub(1).clamp_min(0)
    emb = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
    return F.normalize(emb.float(), dim=-1)


def encode_caption(model, tokenizer, captions, device, max_length):
    batch = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    language_model = getattr(model, "language_model", model)
    outputs = language_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return last_token_embedding_from_outputs(outputs, batch["attention_mask"])


def encode_image_instruction(model, processor, images, instructions, device, max_length):
    prompts = [f"<image>\n{instruction}" for instruction in instructions]
    batch = processor(
        text=prompts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    outputs = model(
        **batch,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return last_token_embedding_from_outputs(outputs, batch["attention_mask"])


def train(args):
    processor, model = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    model.to(args.device)
    broadcast_model(model)
    model.train()

    trainable = [param for param in model.parameters() if param.requires_grad]
    if is_master():
        logging.info("Teacher trainable params: %.2fM", sum(p.numel() for p in trainable) / 1e6)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=args.dtype == "fp16", init_scale=args.amp_init_scale)
    dataloader = build_wds(args)
    iterator = iter(dataloader)

    optimizer.zero_grad(set_to_none=True)
    end = time.time()
    for step in range(args.max_steps):
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        for _ in range(args.accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            with autocast(enabled=args.dtype == "fp16"):
                query = encode_image_instruction(
                    model,
                    processor,
                    batch["images"],
                    [args.query_prompt_template.format(instruction=instruction) for instruction in batch["instruction"]],
                    args.device,
                    args.max_length,
                )
                target = encode_caption(
                    model,
                    tokenizer,
                    [
                        args.caption_prompt_template.format(caption=caption)
                        for caption in batch["modified_caption"]
                    ],
                    args.device,
                    args.max_length,
                )
                loss, logits, labels = contrastive(query, target, args.temperature)
                loss = loss / float(args.accum_steps)
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        sync_grads(model)
        if args.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
        else:
            grad_norm = torch.zeros((), device=args.device)
            for param in trainable:
                if param.grad is not None:
                    grad_norm = grad_norm + param.grad.detach().float().pow(2).sum()
            grad_norm = grad_norm.sqrt()
        finite_grad = torch.isfinite(grad_norm).to(args.device)
        if is_dist():
            dist.all_reduce(finite_grad, op=dist.ReduceOp.MIN)
        if bool(finite_grad.item()):
            scaler.step(optimizer)
        elif is_master():
            logging.warning("teacher step=%d skipped non-finite grad_norm=%s", step, grad_norm.item())
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if args.device.type == "cuda":
            peak_mem_mb = torch.tensor(
                torch.cuda.max_memory_allocated(args.device) / (1024**2),
                device=args.device,
            )
            if is_dist():
                dist.all_reduce(peak_mem_mb, op=dist.ReduceOp.MAX)
        else:
            peak_mem_mb = torch.tensor(0.0)

        if is_master() and step % args.log_interval == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            logging.info(
                "teacher step=%d/%d loss=%.4f acc=%.3f grad_norm=%.3f peak_mem=%.0fMiB time=%.2fs",
                step,
                args.max_steps,
                loss.detach().item() * args.accum_steps,
                acc,
                grad_norm.detach().float().item(),
                peak_mem_mb.detach().float().item(),
                time.time() - end,
            )
            end = time.time()

        if is_master() and args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_teacher(model, processor, args, f"step_{step + 1}")

    if is_master():
        save_teacher(model, processor, args, "final")


def save_teacher(model, processor, args, name):
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    logging.info("Saved teacher adapter/model to %s", out)


def parse_args():
    parser = argparse.ArgumentParser("Train LLaVA-Phi3 DistillCIR teacher with contrastive Lcom")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--cc3m-cir-jsonl", required=True)
    parser.add_argument("--reasoning-jsonl", default=None)
    parser.add_argument("--wds-shards", required=True)
    parser.add_argument("--wds-image-key", default="jpg;png;jpeg;webp")
    parser.add_argument("--wds-text-key", default="txt;text;caption")
    parser.add_argument("--wds-shuffle", type=int, default=10000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2807)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--query-prompt-template",
        default="Apply the prompt: {instruction} to the image. Provide one word for the conditioned image:",
    )
    parser.add_argument(
        "--caption-prompt-template",
        default="Summarize the caption for retrieval: {caption}",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--train-projector", action="store_true", default=True)
    parser.add_argument("--no-train-projector", dest="train_projector", action="store_false")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--allow-cuda-visible-devices", action="store_true", default=False)
    parser.add_argument(
        "--physical-gpus",
        type=str,
        default="",
        help="Comma-separated physical GPU ids for torchrun local ranks, e.g. 0,1,2,7.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_dist(args)
    setup_logging(args)
    seed_all(args.seed, args.rank)
    if is_master():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "teacher_args.json").write_text(
            json.dumps(jsonable_args(args), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    train(args)
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
