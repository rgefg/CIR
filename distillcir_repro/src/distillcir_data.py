import functools
import glob
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import braceexpand
import numpy as np
import PIL
import torch
import webdataset as wds
import webdataset.handlers as wds_handlers
import webdataset.utils as wds_utils
from PIL import Image
from torch.utils.data import DataLoader


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: Optional[object] = None


def seed_dataloader_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _first_text(value, keys):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, str):
                return item
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item, keys)
            if text:
                return text
        return ""
    return str(value)


def _reasoning_text(obj: dict) -> str:
    text = _first_text(
        obj,
        [
            "reasoning",
            "rationale",
            "brainstorming",
            "analysis",
            "cot",
            "chain_of_thought",
            "response",
        ],
    )
    if text:
        return text

    instruction = _first_text(obj.get("instruction"), ["instruction", "text", "value"])
    modified = _first_text(
        obj.get("modified_caption"),
        ["modified_caption", "caption", "text", "value"],
    )
    if instruction or modified:
        return f"Instruction: {instruction}. Modified caption: {modified}."
    return ""


def load_distillcir_jsonl(jsonl_path: str, reasoning_jsonl_path: Optional[str] = None) -> Dict[str, dict]:
    """Load CC3M synthetic triplets with DistillCIR reasoning.

    Expected fields are id, instruction, modified_caption, and one of
    reasoning/rationale/brainstorming. Existing LLM data in this repo uses
    brainstorming, which is used as the reasoning target by default.
    """
    meta: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            sample_id = str(obj.get("id", "")).strip()
            if not sample_id:
                continue
            instruction = _first_text(obj.get("instruction"), ["instruction", "text", "value", "en"])
            modified = _first_text(
                obj.get("modified_caption"),
                ["modified_caption", "caption", "text", "value", "en", "description"],
            )
            reasoning = _reasoning_text(obj)
            if not instruction or not modified:
                continue
            meta[sample_id] = {
                "instruction": instruction.strip(),
                "modified_caption": modified.strip(),
                "reasoning": reasoning.strip(),
            }

    if reasoning_jsonl_path and os.path.exists(reasoning_jsonl_path):
        updated = 0
        with open(reasoning_jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sample_id = str(obj.get("id", "")).strip()
                if sample_id not in meta:
                    continue
                reasoning = _reasoning_text(obj).strip()
                if reasoning:
                    meta[sample_id]["reasoning"] = reasoning
                    updated += 1
        logging.info("Loaded reasoning sidecar: %s updated=%d", reasoning_jsonl_path, updated)

    logging.info("Loaded DistillCIR metadata: %s samples from %s", len(meta), jsonl_path)
    return meta


def _select_sample(sample: dict, id_to_meta: Dict[str, dict]) -> bool:
    key = sample.get("__key__")
    return key is not None and key in id_to_meta


def _attach_meta(sample: dict, id_to_meta: Dict[str, dict]) -> dict:
    key = sample["__key__"]
    meta = id_to_meta[key]
    src_caption = sample.get("src_caption", "")
    if isinstance(src_caption, bytes):
        src_caption = src_caption.decode("utf-8", errors="ignore")
    elif src_caption is None:
        src_caption = ""
    else:
        src_caption = str(src_caption)
    return {
        "id": key,
        "ref_img": sample["image"],
        "src_caption": src_caption,
        "instruction": meta["instruction"],
        "modified_caption": meta["modified_caption"],
        "reasoning": meta.get("reasoning", ""),
    }


def _to_tensor(sample: dict, preprocess_fn) -> dict:
    image = sample["ref_img"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    sample["ref_img"] = preprocess_fn(image)
    return sample


def expand_shards(shards_pattern: str) -> Iterable[str]:
    if shards_pattern.count("{") != shards_pattern.count("}"):
        raise ValueError(f"Malformed shard pattern with unmatched braces: {shards_pattern!r}")

    expanded = list(braceexpand.braceexpand(shards_pattern))
    shard_list = []
    for pattern in expanded:
        if "*" in pattern or "?" in pattern:
            shard_list.extend(glob.glob(pattern))
        else:
            shard_list.append(pattern)
    shard_list = [
        item
        for item in shard_list
        if str(item).startswith(("http://", "https://", "pipe:")) or Path(item).exists()
    ]
    if not shard_list:
        raise FileNotFoundError(f"No shards matched pattern: {shards_pattern!r}")
    return sorted(shard_list)


def build_distillcir_wds(args, preprocess_fn, is_train: bool = True) -> DataInfo:
    if not args.cc3m_cir_jsonl:
        raise ValueError("--cc3m-cir-jsonl is required")
    if not args.wds_shards:
        raise ValueError("--wds-shards is required")

    id_to_meta = load_distillcir_jsonl(
        args.cc3m_cir_jsonl,
        reasoning_jsonl_path=getattr(args, "reasoning_jsonl", None),
    )
    shards = expand_shards(args.wds_shards)
    logging.info("Expanded WebDataset shards: %d", len(shards))

    deterministic = bool(is_train and getattr(args, "wds_deterministic", False))
    base_seed = int(getattr(args, "seed", 0))
    image_keys = getattr(args, "wds_image_key", "jpg;png;jpeg;webp")
    text_keys = getattr(args, "wds_text_key", "txt;text;caption")

    pipeline = []
    if is_train and getattr(args, "wds_resampled", True):
        if deterministic:
            pipeline.append(
                wds.ResampledShards(
                    shards,
                    seed=base_seed,
                    worker_seed=wds_utils.pytorch_worker_seed,
                    deterministic=True,
                )
            )
        else:
            pipeline.append(wds.ResampledShards(shards))
    else:
        pipeline.append(wds.SimpleShardList(shards))

    if is_train and getattr(args, "distributed", False):
        pipeline.append(wds.split_by_node)
    pipeline.append(wds.split_by_worker)

    if is_train and not getattr(args, "wds_resampled", True):
        pipeline.append(
            wds.shuffle(
                int(getattr(args, "wds_shardshuffle", 1000)),
                seed=(base_seed + 17) if deterministic else None,
            )
        )

    pipeline.append(wds.tarfile_to_samples(handler=wds_handlers.warn_and_continue))
    if is_train:
        pipeline.append(
            wds.shuffle(
                int(getattr(args, "wds_shuffle", 20000)),
                seed=(base_seed + 23) if deterministic else None,
            )
        )
    pipeline.append(wds.decode("pil"))
    pipeline.append(wds.rename(image=image_keys, src_caption=text_keys, __key__="__key__"))
    pipeline.append(wds.select(functools.partial(_select_sample, id_to_meta=id_to_meta)))
    pipeline.append(wds.map(functools.partial(_attach_meta, id_to_meta=id_to_meta)))
    pipeline.append(wds.map(functools.partial(_to_tensor, preprocess_fn=preprocess_fn)))

    dataset = wds.DataPipeline(*pipeline)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
        pin_memory=True,
        drop_last=is_train,
        worker_init_fn=seed_dataloader_worker if is_train else None,
    )
    dataloader.num_samples = None
    dataloader.num_batches = None
    return DataInfo(dataloader=dataloader, sampler=None)
