import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def init_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def read_jsonl_for_rank(path: Path, rank: int, world_size: int):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % world_size != rank or not line.strip():
                continue
            obj = json.loads(line)
            sample_id = str(obj.get("id", "")).strip()
            caption = obj.get("modified_caption") or obj.get("caption") or obj.get("text") or ""
            if isinstance(caption, dict):
                caption = caption.get("modified_caption") or caption.get("caption") or caption.get("text") or ""
            if sample_id and str(caption).strip():
                records.append((sample_id, str(caption).strip()))
    return records


def load_tokenizer_and_model(model_path: str, adapter_path: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16 if dtype == "bf16" else torch.float32
    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        processor = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model


@torch.no_grad()
def encode_text_batch(tokenizer, model, texts, device, max_length: int):
    batch = tokenizer(
        texts,
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
    hidden = outputs.hidden_states[-1]
    last_index = batch["attention_mask"].sum(dim=1).sub(1).clamp_min(0)
    emb = hidden[torch.arange(hidden.size(0), device=device), last_index]
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser("Cache trained LLaVA teacher caption embeddings")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()

    rank, world_size, local_rank = init_dist()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl_for_rank(Path(args.jsonl), rank, world_size)
    tokenizer, model = load_tokenizer_and_model(args.model_path, args.adapter_path, args.dtype)
    model.to(device)
    model.eval()

    ids_path = out_dir / f"ids.rank{rank}.txt"
    emb_path = out_dir / f"embeddings.rank{rank}.npy"
    meta_path = out_dir / f"meta.rank{rank}.json"

    embeddings = []
    ids = []
    iterator = range(0, len(records), args.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc="teacher-cache")
    for start in iterator:
        chunk = records[start : start + args.batch_size]
        ids.extend([item[0] for item in chunk])
        texts = [item[1] for item in chunk]
        embeddings.append(encode_text_batch(tokenizer, model, texts, device, args.max_length))

    if embeddings:
        array = np.concatenate(embeddings, axis=0)
    else:
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", 0)
        array = np.zeros((0, hidden_size), dtype=np.float16)
    np.save(emb_path, array)
    ids_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "dtype": args.dtype,
                "rank": rank,
                "world_size": world_size,
                "rows": int(array.shape[0]),
                "dim": int(array.shape[1]) if array.ndim == 2 else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

