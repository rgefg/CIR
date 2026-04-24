#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import CIRR, CIRCODataset  # noqa: E402
from eval_retrieval import load_model  # noqa: E402
from third_party.open_clip.clip import tokenize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure composed-query direction alignment on CIRR/CIRCO validation tuples."
    )
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", type=str, default="ViT-L/14")
    parser.add_argument("--img2text-arch", type=str, default="phi")
    parser.add_argument("--middle-dim", type=int, default=3072)
    parser.add_argument("--img2text-pretrained", type=str, default=None)
    parser.add_argument("--retrieval-prompt-connector", choices=["and", "that"], default="and")
    parser.add_argument("--datasets", type=str, default="cirr,circo")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--name", type=str, default="query_direction_eval")
    parser.add_argument("--logs", type=str, default=str(REPO_ROOT / "logs"))
    return parser.parse_args()


def build_eval_args(args):
    return SimpleNamespace(
        model=args.model,
        middle_dim=args.middle_dim,
        n_layer=2,
        img2text_arch=args.img2text_arch,
        img2text_pretrained=args.img2text_pretrained,
        precision="amp",
        gpu=args.gpu,
        distributed=False,
        use_bn_sync=False,
        dp=False,
        multigpu=None,
        resume=args.resume,
        checkpoint_path="",
        logs=args.logs,
        name=args.name,
        no_lora=False,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.0,
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def normalize(x, eps):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, values):
        values = values.detach().float().cpu()
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.sum += float(values.sum().item())
        self.sum_sq += float((values * values).sum().item())

    def to_dict(self):
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None}
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return {"count": self.count, "mean": mean, "std": var ** 0.5}


class DirectionStats:
    def __init__(self):
        self.query_target = RunningMoments()
        self.query_source = RunningMoments()
        self.target_source = RunningMoments()
        self.query_dir = RunningMoments()
        self.delta_target_norm = RunningMoments()
        self.delta_query_norm = RunningMoments()
        self.skipped = 0

    def update(self, src, tgt, query, eps):
        src = normalize(src, eps)
        tgt = normalize(tgt, eps)
        query = normalize(query, eps)
        target_delta = tgt - src
        query_delta = query - src
        target_norm = target_delta.norm(dim=-1)
        query_norm = query_delta.norm(dim=-1)
        valid = (target_norm > eps) & (query_norm > eps)
        self.skipped += int((~valid).sum().item())
        if not valid.any():
            return
        src = src[valid]
        tgt = tgt[valid]
        query = query[valid]
        target_delta = normalize(target_delta[valid], eps)
        query_delta = normalize(query_delta[valid], eps)
        self.query_target.update((query * tgt).sum(dim=-1))
        self.query_source.update((query * src).sum(dim=-1))
        self.target_source.update((tgt * src).sum(dim=-1))
        self.query_dir.update((query_delta * target_delta).sum(dim=-1))
        self.delta_target_norm.update(target_norm[valid])
        self.delta_query_norm.update(query_norm[valid])

    def to_dict(self):
        return {
            "query_target": self.query_target.to_dict(),
            "query_source": self.query_source.to_dict(),
            "target_source": self.target_source.to_dict(),
            "query_dir": self.query_dir.to_dict(),
            "delta_target_norm": self.delta_target_norm.to_dict(),
            "delta_query_norm": self.delta_query_norm.to_dict(),
            "skipped": self.skipped,
        }


def make_prompts(captions, connector):
    if connector == "and":
        return [f"a photo of * and {cap}" for cap in captions]
    return [f"a photo of * that {cap}" for cap in captions]


def encode_query(model, img2text, ref_images, prompt_tokens, placeholder_token_id, eps):
    src_features = model.encode_image(ref_images)
    soft_tokens = img2text(src_features)
    query = model.encode_text_img_retrieval(
        prompt_tokens,
        soft_tokens,
        split_ind=placeholder_token_id,
        repeat=False,
    )
    return normalize(src_features, eps), normalize(query, eps)


def load_cirr_target_images(answer_paths, preprocess):
    root_img = REPO_ROOT / "data" / "CIRR" / "dev"
    images = []
    for answer in answer_paths:
        path = Path(str(answer))
        if not path.is_absolute():
            path = root_img / path.name
        images.append(preprocess(Image.open(path).convert("RGB")))
    return torch.stack(images, dim=0)


def eval_cirr(model, img2text, preprocess, gpu, batch_size, workers, connector, max_samples, eps):
    dataset = CIRR(transforms=preprocess, root=str(REPO_ROOT / "data"), mode="caps")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    stats = DirectionStats()
    placeholder_token_id = tokenize(["*"])[0][1].item()
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="CIRR-QueryDir", leave=False):
            ref_images, _text_tokens, _caption_only, _ref_paths, answer_paths, raw_captions, _group_members = batch
            if max_samples and seen >= max_samples:
                break
            if max_samples:
                keep = min(ref_images.shape[0], max_samples - seen)
                ref_images = ref_images[:keep]
                answer_paths = list(answer_paths)[:keep]
                raw_captions = list(raw_captions)[:keep]
            prompts = make_prompts(list(raw_captions), connector)
            prompt_tokens = tokenize(prompts).cuda(gpu, non_blocking=True)
            ref_images = ref_images.cuda(gpu, non_blocking=True)
            target_images = load_cirr_target_images(answer_paths, preprocess).cuda(gpu, non_blocking=True)
            src_features, query = encode_query(model, img2text, ref_images, prompt_tokens, placeholder_token_id, eps)
            target_features = normalize(model.encode_image(target_images), eps)
            stats.update(src_features, target_features, query, eps)
            seen += int(ref_images.shape[0])
    return stats.to_dict()


def eval_circo(model, img2text, preprocess, gpu, batch_size, workers, connector, max_samples, eps):
    dataset = CIRCODataset(
        data_path=str(REPO_ROOT / "data" / "CIRCO"),
        split="val",
        mode="relative",
        transforms=preprocess,
        preprocess=preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    stats = DirectionStats()
    placeholder_token_id = tokenize(["*"])[0][1].item()
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="CIRCO-QueryDir", leave=False):
            ref_images = batch["reference_img"]
            target_images = batch["target_img"]
            captions = list(batch["relative_caption"])
            if max_samples and seen >= max_samples:
                break
            if max_samples:
                keep = min(ref_images.shape[0], max_samples - seen)
                ref_images = ref_images[:keep]
                target_images = target_images[:keep]
                captions = captions[:keep]
            prompts = make_prompts(captions, connector)
            prompt_tokens = tokenize(prompts).cuda(gpu, non_blocking=True)
            ref_images = ref_images.cuda(gpu, non_blocking=True)
            target_images = target_images.cuda(gpu, non_blocking=True)
            src_features, query = encode_query(model, img2text, ref_images, prompt_tokens, placeholder_token_id, eps)
            target_features = normalize(model.encode_image(target_images), eps)
            stats.update(src_features, target_features, query, eps)
            seen += int(ref_images.shape[0])
    return stats.to_dict()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    model, img2text, preprocess = load_model(build_eval_args(args))
    model.eval()
    img2text.eval()
    m = unwrap_model(model)
    selected = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    result = {
        "resume": args.resume,
        "model": args.model,
        "img2text_arch": args.img2text_arch,
        "retrieval_prompt_connector": args.retrieval_prompt_connector,
        "max_samples": args.max_samples,
    }
    if "cirr" in selected:
        result["cirr"] = eval_cirr(
            m,
            img2text,
            preprocess,
            args.gpu,
            args.batch_size,
            args.workers,
            args.retrieval_prompt_connector,
            args.max_samples,
            args.eps,
        )
    if "circo" in selected:
        result["circo"] = eval_circo(
            m,
            img2text,
            preprocess,
            args.gpu,
            args.batch_size,
            args.workers,
            args.retrieval_prompt_connector,
            args.max_samples,
            args.eps,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
