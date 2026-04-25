import argparse
import json
import logging
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from distillcir_data import build_distillcir_wds
from model.clip import _transform, load
from model.model import CLIP, IM2TEXT, Phi, LoRALinear, LoRAMultiheadAttention
from params import get_default_params
from third_party.open_clip.clip import tokenize
from third_party.open_clip.scheduler import cosine_lr
from utils import convert_models_to_fp32


def safe_torch_load(path, map_location=None, weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=weights_only)


def is_master() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def broadcast_modules(modules):
    if not (dist.is_available() and dist.is_initialized()):
        return
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=0)


def sync_gradients(modules):
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = float(dist.get_world_size())
    for module in modules:
        for param in module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def setup_logging(log_path: Path, rank: int):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


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


def seed_everything(seed: int, rank: int):
    final_seed = int(seed) + int(rank)
    random.seed(final_seed)
    np.random.seed(final_seed % (2**32))
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = world_size > 1
    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank

    if os.environ.get("CUDA_VISIBLE_DEVICES") and not args.allow_cuda_visible_devices:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is set. This launcher intentionally refuses GPU remapping; "
            "run on physical GPUs directly or pass --allow-cuda-visible-devices only if you have verified the mapping."
        )

    if not torch.cuda.is_available():
        args.device = torch.device("cpu")
        return

    torch.cuda.set_device(local_rank)
    args.device = torch.device("cuda", local_rank)
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://")


def log_physical_gpu_state():
    if not is_master():
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            logging.info("Physical GPU state before training:\n%s", result.stdout.strip())
        elif result.stderr.strip():
            logging.warning("nvidia-smi did not return GPU state: %s", result.stderr.strip())
    except FileNotFoundError:
        logging.warning("nvidia-smi not found; cannot verify physical GPU occupancy.")


def apply_lora_to_linear_layers(module: nn.Module, r: int, alpha: int, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRAMultiheadAttention, LoRALinear)):
            continue
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear_layers(child, r=r, alpha=alpha, dropout=dropout)


def freeze_clip_except_lora_and_logit_scale(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if name == "logit_scale" or name.endswith(".A") or name.endswith(".B"):
            param.requires_grad = True


def build_clip_and_projection(args):
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(args.model, jit=False)
    elif args.pic2word_pretrained:
        checkpoint = safe_torch_load(args.pic2word_pretrained, map_location="cpu", weights_only=False)
        if "model_config" in checkpoint:
            model = CLIP(**checkpoint["model_config"])
        else:
            logging.warning("model_config missing in Pic2Word checkpoint; loading %s architecture.", args.model)
            model, _, _ = load(args.model, jit=False)
        state_dict = checkpoint["state_dict"]
        if next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info("Loaded Pic2Word CLIP state: missing=%d unexpected=%d", len(msg.missing_keys), len(msg.unexpected_keys))
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)
    else:
        raise ValueError("Use --openai-pretrained or --pic2word-pretrained.")

    if args.img2text_arch == "phi":
        img2text = Phi(
            input_dim=model.embed_dim,
            hidden_dim=int(args.middle_dim),
            output_dim=model.token_embedding.weight.shape[1],
            dropout=float(args.droprate),
        )
    else:
        img2text = IM2TEXT(
            embed_dim=model.embed_dim,
            middle_dim=int(args.middle_dim),
            output_dim=model.token_embedding.weight.shape[1],
            n_layer=int(args.n_layer),
            dropout=float(args.droprate),
        )

    if args.img2text_pretrained:
        ckpt = safe_torch_load(args.img2text_pretrained, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "Phi" in ckpt:
            state_dict = ckpt["Phi"]
        elif isinstance(ckpt, dict) and "state_dict_img2text" in ckpt:
            state_dict = ckpt["state_dict_img2text"]
        else:
            state_dict = ckpt
        if next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        msg = img2text.load_state_dict(state_dict, strict=False)
        logging.info("Loaded img2text checkpoint: missing=%s unexpected=%s", msg.missing_keys, msg.unexpected_keys)
    elif args.pic2word_pretrained:
        checkpoint = safe_torch_load(args.pic2word_pretrained, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict_img2text")
        if state_dict is None:
            raise RuntimeError("--pic2word-pretrained does not contain state_dict_img2text")
        if next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        msg = img2text.load_state_dict(state_dict, strict=False)
        logging.info("Loaded Pic2Word img2text state: missing=%s unexpected=%s", msg.missing_keys, msg.unexpected_keys)

    if args.precision in {"amp", "fp32"}:
        convert_models_to_fp32(model)

    if not args.no_lora:
        apply_lora_to_linear_layers(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    freeze_clip_except_lora_and_logit_scale(model)
    for param in img2text.parameters():
        param.requires_grad = True

    return model, img2text, preprocess_train, preprocess_val


class DistillCIRHeads(nn.Module):
    def __init__(self, text_width: int, teacher_dim: int, reason_prompt_tokens: int):
        super().__init__()
        self.reason_prompt_tokens = int(reason_prompt_tokens)
        if self.reason_prompt_tokens > 0:
            self.reason_prompt = nn.Parameter(torch.empty(self.reason_prompt_tokens, text_width))
            nn.init.normal_(self.reason_prompt, std=0.02)
        else:
            self.register_parameter("reason_prompt", None)

        self.feature_projector = None
        if int(teacher_dim) > 0:
            self.feature_projector = nn.Linear(text_width if text_width == teacher_dim else teacher_dim, teacher_dim)

    def reset_feature_projector(self, student_dim: int, teacher_dim: int):
        if teacher_dim <= 0:
            self.feature_projector = None
        else:
            self.feature_projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, query_features):
        projected = self.feature_projector(query_features) if self.feature_projector is not None else None
        return self.reason_prompt, projected


class TeacherEmbeddingStore:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        ids_path = self.cache_dir / "ids.txt"
        emb_path = self.cache_dir / "embeddings.npy"
        if not meta_path.exists() or not ids_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"Teacher cache must contain meta.json, ids.txt, embeddings.npy under {self.cache_dir}"
            )
        with meta_path.open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.embeddings = np.load(emb_path, mmap_mode="r")
        with ids_path.open("r", encoding="utf-8") as handle:
            ids = [line.strip() for line in handle if line.strip()]
        self.id_to_index = {sample_id: idx for idx, sample_id in enumerate(ids)}
        if len(ids) != self.embeddings.shape[0]:
            raise ValueError("Teacher cache id count does not match embedding rows.")
        self.dim = int(self.embeddings.shape[1])
        logging.info("Loaded teacher cache: rows=%d dim=%d dir=%s", len(ids), self.dim, self.cache_dir)

    def get(self, sample_ids, device):
        vectors = []
        valid = []
        for sample_id in sample_ids:
            idx = self.id_to_index.get(str(sample_id))
            if idx is None:
                valid.append(False)
                vectors.append(np.zeros((self.dim,), dtype=np.float16))
            else:
                valid.append(True)
                vectors.append(self.embeddings[idx])
        array = np.asarray(vectors)
        tensor = torch.as_tensor(array, device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid, device=device, dtype=torch.bool)
        return tensor, mask


def text_to_str_list(values):
    return [str(v) if v is not None else "" for v in values]


def build_retrieval_prompts(instructions, placeholder: str, connector: str):
    prompts = []
    for instruction in text_to_str_list(instructions):
        instruction = instruction.strip()
        if instruction:
            prompts.append(f"a photo of {placeholder} {connector} {instruction}")
        else:
            prompts.append(f"a photo of {placeholder}")
    return prompts


def build_reason_prompts(modified_captions, placeholder: str, connector: str):
    prompts = []
    for caption in text_to_str_list(modified_captions):
        caption = caption.strip()
        if caption:
            prompts.append(f"a photo of {placeholder} {connector} {caption}")
        else:
            prompts.append(f"a photo of {placeholder}")
    return prompts


def encode_text_img_prompt(model, text, img_tokens, placeholder_token_id: int, prompt_tokens=None):
    base = unwrap(model)
    x = base.token_embedding(text).type(base.dtype)
    collect_ind = (text == base.end_id).nonzero()[:, 1]

    new_x = []
    for idx, sample in enumerate(x):
        insert_positions = (text[idx] == placeholder_token_id).nonzero()
        if len(insert_positions) == 0:
            raise ValueError(f"Placeholder token id {placeholder_token_id} not found in prompt index {idx}")
        insert_pos = insert_positions[0]
        image_token = img_tokens[idx].view(1, 1, -1).to(dtype=base.dtype)
        sample = sample.view(1, x.size(1), -1)
        sample = torch.cat([sample[:, :insert_pos], image_token, sample[:, insert_pos + 1 :]], dim=1)
        new_x.append(sample)
    x = torch.cat(new_x, dim=0)

    if prompt_tokens is not None and prompt_tokens.numel() > 0:
        k = prompt_tokens.shape[0]
        prompt = prompt_tokens.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        body_len = x.size(1) - 1 - k
        if body_len <= 0:
            raise ValueError("Too many reasoning prompt tokens for CLIP context length.")
        x = torch.cat([x[:, :1], prompt, x[:, 1 : 1 + body_len]], dim=1)
        collect_ind = torch.clamp(collect_ind + k, max=x.size(1) - 1)

    x = x + base.positional_embedding.type(base.dtype)
    x = x.permute(1, 0, 2)
    x = base.transformer(x)
    x = x.permute(1, 0, 2)
    x = base.ln_final(x).type(base.dtype)
    return x[torch.arange(x.size(0), device=x.device), collect_ind] @ base.text_projection


def normalize(features):
    return F.normalize(features, dim=-1, eps=1e-6)


def gather_targets(features):
    if not (dist.is_available() and dist.is_initialized()):
        return features
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features.contiguous())
    return torch.cat([features] + gathered[:rank] + gathered[rank + 1 :], dim=0)


def contrastive_loss(query, target, logit_scale, bidirectional=False):
    all_target = gather_targets(target)
    logits = logit_scale * (query @ all_target.t())
    labels = torch.arange(query.size(0), device=query.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    stats = {"acc": float((logits.argmax(dim=-1) == labels).float().mean().detach().item())}
    if bidirectional:
        all_query = gather_targets(query)
        logits_t = logit_scale * (target @ all_query.t())
        loss = 0.5 * (loss + F.cross_entropy(logits_t, labels))
    return loss, stats


def compute_losses(model, img2text, heads, batch, teacher_store, args):
    base = unwrap(model)
    device = args.device
    images = batch["ref_img"].to(device, non_blocking=True)
    sample_ids = list(batch["id"])
    instructions = batch["instruction"]
    modified_captions = batch["modified_caption"]
    reasoning = batch["reasoning"]

    image_features = base.encode_image(images)
    token_features = img2text(image_features)

    placeholder_token_id = int(tokenize([args.prompt_placeholder])[0][1].item())
    query_tokens = tokenize(
        build_retrieval_prompts(instructions, args.prompt_placeholder, args.retrieval_prompt_connector),
        truncate=True,
    ).to(device, non_blocking=True)
    query_features = normalize(
        encode_text_img_prompt(base, query_tokens, token_features, placeholder_token_id)
    )

    reason_prompt, projected_query = heads(query_features)
    caption_tokens = tokenize(text_to_str_list(modified_captions), truncate=True).to(device, non_blocking=True)
    caption_features = normalize(base.encode_text(caption_tokens))

    logit_scale = torch.clamp(base.logit_scale.exp(), max=float(args.max_logit_scale)).mean()
    lcom, lcom_stats = contrastive_loss(
        query_features,
        caption_features,
        logit_scale,
        bidirectional=args.bidirectional_contrastive,
    )

    reason_tokens = tokenize(
        build_reason_prompts(modified_captions, args.prompt_placeholder, args.reason_prompt_connector),
        truncate=True,
    ).to(device, non_blocking=True)
    reason_query = normalize(
        encode_text_img_prompt(base, reason_tokens, token_features, placeholder_token_id, prompt_tokens=reason_prompt)
    )
    reasoning_tokens = tokenize(text_to_str_list(reasoning), truncate=True).to(device, non_blocking=True)
    reasoning_features = normalize(base.encode_text(reasoning_tokens))
    lrea, lrea_stats = contrastive_loss(
        reason_query,
        reasoning_features,
        logit_scale,
        bidirectional=args.bidirectional_contrastive,
    )

    lfea = query_features.sum() * 0.0
    lfea_stats = {"acc": 0.0, "valid": 0.0}
    if teacher_store is not None and projected_query is not None:
        teacher_features, valid_mask = teacher_store.get(sample_ids, device)
        valid_count = int(valid_mask.sum().item())
        lfea_stats["valid"] = float(valid_count) / float(max(1, len(sample_ids)))
        if valid_count > 0:
            projected_query = normalize(projected_query[valid_mask])
            teacher_features = normalize(teacher_features[valid_mask])
            lfea, lfea_stats_inner = contrastive_loss(
                projected_query,
                teacher_features,
                logit_scale,
                bidirectional=args.bidirectional_contrastive,
            )
            lfea_stats.update(lfea_stats_inner)

    total = lcom + (float(args.alpha_reason) * lrea) + (float(args.beta_feature) * lfea)
    stats = {
        "loss": float(total.detach().item()),
        "lcom": float(lcom.detach().item()),
        "lrea": float(lrea.detach().item()),
        "lfea": float(lfea.detach().item()),
        "lcom_acc": lcom_stats["acc"],
        "lrea_acc": lrea_stats["acc"],
        "lfea_acc": lfea_stats["acc"],
        "lfea_valid": lfea_stats["valid"],
        "logit_scale": float(logit_scale.detach().item()),
    }
    return total, stats


def train_one_epoch(model, img2text, heads, data, teacher_store, optimizer, scaler, scheduler, epoch, args):
    model.train()
    img2text.train()
    heads.train()
    dataloader = data.dataloader
    num_updates = int(args.wds_epoch_steps)
    accum_steps = int(args.accum_steps)
    iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    end = time.time()

    for update_idx in range(num_updates):
        last_stats = None
        for _ in range(accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            with autocast(enabled=args.precision == "amp"):
                loss, stats = compute_losses(model, img2text, heads, batch, teacher_store, args)
                loss = loss / float(accum_steps)
            if args.precision == "amp":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            last_stats = stats

        global_step = epoch * num_updates + update_idx
        if args.precision == "amp":
            scaler.unscale_(optimizer)
        sync_gradients([model, img2text, heads])
        scheduler(global_step)
        if args.precision == "amp":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_master() and (update_idx % int(args.log_interval) == 0):
            elapsed = time.time() - end
            end = time.time()
            logging.info(
                "epoch=%d step=%d/%d loss=%.4f lcom=%.4f lrea=%.4f lfea=%.4f "
                "acc=(%.3f %.3f %.3f) teacher_valid=%.2f lr=%.6g logit_scale=%.2f time=%.2fs",
                epoch,
                update_idx,
                num_updates,
                last_stats["loss"],
                last_stats["lcom"],
                last_stats["lrea"],
                last_stats["lfea"],
                last_stats["lcom_acc"],
                last_stats["lrea_acc"],
                last_stats["lfea_acc"],
                last_stats["lfea_valid"],
                optimizer.param_groups[0]["lr"],
                last_stats["logit_scale"],
                elapsed,
            )

        if args.dry_run_steps > 0 and (update_idx + 1) >= args.dry_run_steps:
            break


def parameter_groups(modules, weight_decay: float):
    decay = []
    no_decay = []
    seen = set()
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if name.endswith("bias") or "ln" in name.lower() or "norm" in name.lower() or name == "logit_scale":
                no_decay.append(param)
            else:
                decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def save_checkpoint(model, img2text, heads, optimizer, epoch, args, filename):
    if not is_master():
        return
    payload = {
        "epoch": epoch,
        "name": args.name,
        "state_dict": unwrap(model).state_dict(),
        "state_dict_img2text": unwrap(img2text).state_dict(),
        "state_dict_distillcir_heads": unwrap(heads).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    path = Path(args.checkpoint_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logging.info("Saved checkpoint: %s", path)


def parse_args():
    parser = argparse.ArgumentParser("DistillCIR student training")
    parser.add_argument("--model", default="ViT-L/14")
    parser.add_argument("--openai-pretrained", action="store_true", default=False)
    parser.add_argument("--pic2word-pretrained", type=str, default=None)
    parser.add_argument("--img2text-arch", choices=["im2text", "phi"], default="im2text")
    parser.add_argument("--img2text-pretrained", type=str, default=None)
    parser.add_argument("--middle-dim", "--middle_dim", dest="middle_dim", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--droprate", type=float, default=0.0)
    parser.add_argument("--cc3m-cir-jsonl", type=str, required=True)
    parser.add_argument("--reasoning-jsonl", type=str, default=None)
    parser.add_argument("--wds-shards", type=str, required=True)
    parser.add_argument("--wds-image-key", type=str, default="jpg;png;jpeg;webp")
    parser.add_argument("--wds-text-key", type=str, default="txt;text;caption")
    parser.add_argument("--wds-shuffle", type=int, default=20000)
    parser.add_argument("--wds-shardshuffle", type=int, default=1000)
    parser.add_argument("--wds-resampled", action="store_true", default=True)
    parser.add_argument("--no-wds-resampled", dest="wds_resampled", action="store_false")
    parser.add_argument("--wds-deterministic", action="store_true", default=False)
    parser.add_argument("--wds-epoch-steps", type=int, default=2807)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-lora", action="store_true", default=False)
    parser.add_argument("--prompt-placeholder", type=str, default="*")
    parser.add_argument("--retrieval-prompt-connector", choices=["and", "that"], default="and")
    parser.add_argument("--reason-prompt-connector", choices=["and", "that"], default="that")
    parser.add_argument("--reason-prompt-tokens", type=int, default=8)
    parser.add_argument("--alpha-reason", type=float, default=1.0)
    parser.add_argument("--beta-feature", type=float, default=1.0)
    parser.add_argument("--teacher-cache", type=str, default=None)
    parser.add_argument("--teacher-dim", type=int, default=0)
    parser.add_argument("--bidirectional-contrastive", action="store_true", default=False)
    parser.add_argument("--max-logit-scale", type=float, default=100.0)
    parser.add_argument("--logs", type=str, default="./logs")
    parser.add_argument("--name", type=str, default="DistillCIR_ViTL14_8x3090")
    parser.add_argument("--save-frequency", type=int, default=1)
    parser.add_argument("--save-most-recent", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--allow-cuda-visible-devices", action="store_true", default=False)
    parser.add_argument("--dry-run-steps", type=int, default=0)
    args = parser.parse_args()
    defaults = get_default_params(args.model)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def main():
    args = parse_args()
    init_distributed(args)
    args.log_path = str(Path(args.logs) / args.name / "out.log")
    args.checkpoint_path = str(Path(args.logs) / args.name / "checkpoints")
    setup_logging(Path(args.log_path), args.rank)
    seed_everything(args.seed, args.rank)
    log_physical_gpu_state()

    if is_master():
        logging.info("Args:\n%s", json.dumps(jsonable_args(args), indent=2, sort_keys=True))
        Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
        with (Path(args.logs) / args.name / "params.json").open("w", encoding="utf-8") as handle:
            json.dump(jsonable_args(args), handle, indent=2, sort_keys=True)

    model, img2text, preprocess_train, _ = build_clip_and_projection(args)
    teacher_store = None
    teacher_dim = int(args.teacher_dim)
    if float(args.beta_feature) > 0.0:
        if not args.teacher_cache:
            raise ValueError("--teacher-cache is required when --beta-feature > 0")
        teacher_store = TeacherEmbeddingStore(args.teacher_cache)
        teacher_dim = teacher_store.dim

    heads = DistillCIRHeads(
        text_width=model.embed_dim,
        teacher_dim=0,
        reason_prompt_tokens=args.reason_prompt_tokens,
    )
    heads.reset_feature_projector(model.embed_dim, teacher_dim if float(args.beta_feature) > 0.0 else 0)

    model.to(args.device)
    img2text.to(args.device)
    heads.to(args.device)
    broadcast_modules([model, img2text, heads])

    trainable = sum(p.numel() for m in [model, img2text, heads] for p in m.parameters() if p.requires_grad)
    if is_master():
        logging.info("Trainable parameters: %.2fM", trainable / 1e6)

    data = build_distillcir_wds(args, preprocess_train, is_train=True)
    modules_for_optim = [unwrap(model), unwrap(img2text), unwrap(heads)]
    optimizer = torch.optim.AdamW(
        parameter_groups(modules_for_optim, args.wd),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    total_steps = int(args.wds_epoch_steps) * int(args.epochs)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    scaler = GradScaler(enabled=args.precision == "amp")

    for epoch in range(args.epochs):
        train_one_epoch(model, img2text, heads, data, teacher_store, optimizer, scaler, scheduler, epoch, args)
        if is_master() and ((epoch + 1) % int(args.save_frequency) == 0 or (epoch + 1) == args.epochs):
            save_checkpoint(model, img2text, heads, optimizer, epoch + 1, args, f"epoch_{epoch + 1}.pt")
        if is_master() and args.save_most_recent:
            save_checkpoint(model, img2text, heads, optimizer, epoch + 1, args, "epoch_latest.pt")
        if args.dry_run_steps > 0:
            break

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
