import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from real_text_geom_stats import (
    build_retrieval_model,
    collect_real_samples,
    encode_image_anchor_query,
    encode_texts,
)
from model.clip import load as load_clip
from third_party.open_clip.clip import tokenize


def parse_args():
    parser = argparse.ArgumentParser("Text-only oracle transfer probe")
    parser.add_argument("--mode", choices=["cc_exact", "fashioniq_proxy"], required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--oracle-model", type=str, default="ViT-L/14")

    parser.add_argument("--pic-model", type=str, default="ViT-L/14")
    parser.add_argument("--pic-ckpt", type=str, default="checkpoint/pic2word_model.pt")
    parser.add_argument("--pic-img2text-arch", type=str, default="im2text")
    parser.add_argument("--pic-middle-dim", type=int, default=512)
    parser.add_argument("--pic-connector", type=str, default="that")

    parser.add_argument("--ours-model", type=str, default="ViT-L/14")
    parser.add_argument("--ours-retrieval-ckpt", type=str, required=True)
    parser.add_argument("--ours-merged-ckpt", type=str, required=True)
    parser.add_argument("--ours-img2text-arch", type=str, default="phi")
    parser.add_argument("--ours-middle-dim", type=int, default=3072)
    parser.add_argument("--ours-connector", type=str, default="that")

    parser.add_argument("--wds-shards", type=str, default="")
    parser.add_argument("--cc3m-cir-jsonl", type=str, default="")
    parser.add_argument("--wds-image-key", type=str, default="jpg;png;jpeg;webp")
    parser.add_argument("--wds-text-key", type=str, default="txt;text;caption")
    parser.add_argument("--wds-shuffle", type=int, default=10000)
    parser.add_argument("--wds-shardshuffle", type=int, default=1000)
    parser.add_argument("--wds-resampled", action="store_true", default=True)
    parser.add_argument("--wds-deterministic", action="store_true", default=True)

    parser.add_argument("--fashioniq-root", type=str, default=str(ROOT / "data"))
    parser.add_argument("--fashioniq-image-root", type=str, default="")
    parser.add_argument("--fashioniq-cloths", type=str, default="dress,shirt,toptee")
    parser.add_argument("--fashioniq-exclude-same", action="store_true", default=False)
    parser.add_argument("--fashioniq-min-content-words", type=int, default=0)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def encode_image_features(model, images_cpu: torch.Tensor, gpu: int, chunk_size: int = 128) -> torch.Tensor:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    outs = []
    with torch.no_grad():
        for start in range(0, images_cpu.shape[0], chunk_size):
            batch = images_cpu[start:start + chunk_size].to(device=device, dtype=torch.float32, non_blocking=True)
            feats = model.encode_image(batch)
            outs.append(l2_normalize(feats).float().cpu())
    return torch.cat(outs, dim=0)


def build_model_args(args, which: str) -> SimpleNamespace:
    if which == "pic":
        model = args.pic_model
        arch = args.pic_img2text_arch
        middle = args.pic_middle_dim
    else:
        model = args.ours_model
        arch = args.ours_img2text_arch
        middle = args.ours_middle_dim
    return SimpleNamespace(
        model=model,
        img2text_arch=arch,
        middle_dim=middle,
        n_layer=2,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.0,
        geo_lora_r=64,
        geo_lora_alpha=16,
        geo_lora_dropout=0.0,
        gpu=args.gpu,
        retrieval_prompt_connector=args.pic_connector if which == "pic" else args.ours_connector,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        wds_shards=args.wds_shards,
        cc3m_cir_jsonl=args.cc3m_cir_jsonl,
        wds_image_key=args.wds_image_key,
        wds_text_key=args.wds_text_key,
        wds_shuffle=args.wds_shuffle,
        wds_shardshuffle=args.wds_shardshuffle,
        wds_resampled=args.wds_resampled,
        wds_deterministic=args.wds_deterministic,
        use_image_anchor=True,
    )


def build_oracle_model(model_name: str, gpu: int):
    model, _, preprocess = load_clip(model_name, jit=False)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device).float().eval()
    return model, preprocess


def choose_hard_negative_by_instruction(instr_feats: torch.Tensor, src_feats: torch.Tensor) -> list[int]:
    sim_instr = instr_feats @ instr_feats.t()
    sim_src = src_feats @ src_feats.t()
    n = sim_instr.shape[0]
    out = []
    for i in range(n):
        order = torch.argsort(sim_instr[i], descending=True)
        chosen = None
        for j in order.tolist():
            if j == i:
                continue
            if sim_src[i, j].item() < 0.75:
                chosen = j
                break
        if chosen is None:
            chosen = int(order[1].item())
        out.append(chosen)
    return out


def summarize_candidate_scores(scores: np.ndarray, negative_names: list[str] | None = None) -> dict:
    pos = scores[:, 0]
    neg = scores[:, 1:]
    hard_neg = neg.max(axis=1)
    mean_neg = neg.mean(axis=1)
    top1 = (scores.argmax(axis=1) == 0).mean()
    out = {
        "mean_positive": float(pos.mean()),
        "mean_mean_negative": float(mean_neg.mean()),
        "mean_hard_negative": float(hard_neg.mean()),
        "mean_margin_to_mean_negative": float((pos - mean_neg).mean()),
        "mean_margin_to_hard_negative": float((pos - hard_neg).mean()),
        "top1_acc": float(top1),
    }
    if negative_names:
        for idx, name in enumerate(negative_names, start=1):
            out[f"pairwise_vs_{name}"] = float((scores[:, 0] > scores[:, idx]).mean())
    return out


def score_text_candidates(
    query_feats: torch.Tensor,
    candidate_texts: list[list[str]],
    model,
    gpu: int,
    negative_names: list[str] | None = None,
) -> dict:
    unique_texts = []
    for row in candidate_texts:
        unique_texts.extend(row)
    unique_texts = list(dict.fromkeys(unique_texts))
    feat_map = {}
    text_feats = encode_texts(model, unique_texts, gpu)
    for text, feat in zip(unique_texts, text_feats):
        feat_map[text] = feat
    cand = torch.stack([torch.stack([feat_map[t] for t in row], dim=0) for row in candidate_texts], dim=0)
    scores = torch.einsum("bd,bkd->bk", query_feats, cand).cpu().numpy()
    return summarize_candidate_scores(scores, negative_names=negative_names)


def score_image_candidates(
    query_feats: torch.Tensor,
    candidate_paths: list[list[str]],
    model,
    preprocess,
    gpu: int,
    negative_names: list[str] | None = None,
) -> dict:
    unique_paths = []
    for row in candidate_paths:
        unique_paths.extend(row)
    unique_paths = list(dict.fromkeys(unique_paths))
    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in unique_paths], dim=0)
    img_feats = encode_image_features(model, images, gpu)
    feat_map = {path: feat for path, feat in zip(unique_paths, img_feats)}
    cand = torch.stack([torch.stack([feat_map[p] for p in row], dim=0) for row in candidate_paths], dim=0)
    scores = torch.einsum("bd,bkd->bk", query_feats, cand).cpu().numpy()
    return summarize_candidate_scores(scores, negative_names=negative_names)


def run_cc_exact(args):
    oracle_model, _ = build_oracle_model(args.oracle_model, args.gpu)
    pic_args = build_model_args(args, "pic")
    ours_args = build_model_args(args, "ours")

    sample_pack = collect_real_samples(ours_args)
    valid_n = sample_pack["valid_num_samples"]
    n = min(valid_n, args.num_samples)
    sample_pack = {k: (v[:n] if isinstance(v, list) else v[:n]) for k, v in sample_pack.items() if k in {"src", "tgt", "fwd", "rev", "images"}}

    src_texts = sample_pack["src"]
    tgt_texts = sample_pack["tgt"]
    fwd_texts = sample_pack["fwd"]
    images = sample_pack["images"]

    instr_feats = encode_texts(oracle_model, fwd_texts, args.gpu)
    src_feats = encode_texts(oracle_model, src_texts, args.gpu)
    hard_idx = choose_hard_negative_by_instruction(instr_feats, src_feats)
    rng = random.Random(args.seed)
    rand_idx = []
    for i in range(n):
        choices = [j for j in range(n) if j not in {i, hard_idx[i]}]
        rand_idx.append(rng.choice(choices))

    candidate_texts = [
        [tgt_texts[i], src_texts[i], tgt_texts[hard_idx[i]], tgt_texts[rand_idx[i]]]
        for i in range(n)
    ]

    oracle_prompts = [f"a photo of {src}, but {ins}" for src, ins in zip(src_texts, fwd_texts)]
    oracle_q = encode_texts(oracle_model, oracle_prompts, args.gpu)

    pic_model, pic_img2text = build_retrieval_model(pic_args, ckpt_path=args.pic_ckpt)
    pic_prompts = [f"a photo of * {args.pic_connector} {ins}" for ins in fwd_texts]
    pic_q = encode_image_anchor_query(pic_model, pic_img2text, images, pic_prompts, args.gpu)

    ours_model, ours_img2text = build_retrieval_model(ours_args, ckpt_path=args.ours_retrieval_ckpt)
    ours_prompts = [f"a photo of * {args.ours_connector} {ins}" for ins in fwd_texts]
    ours_q = encode_image_anchor_query(ours_model, ours_img2text, images, ours_prompts, args.gpu)

    merged_model, merged_img2text = build_retrieval_model(ours_args, ckpt_path=args.ours_merged_ckpt)
    merged_q = encode_image_anchor_query(merged_model, merged_img2text, images, ours_prompts, args.gpu)

    results = {
        "mode": "cc_exact",
        "num_samples": n,
        "candidate_definition": {
            "positive": "true target caption",
            "source_negative": "source caption",
            "edit_wrong_source": "target caption from instruction-similar but source-different sample",
            "random_negative": "random target caption",
        },
        "oracle_clip_text": score_text_candidates(oracle_q, candidate_texts, oracle_model, args.gpu),
        "pic2word": score_text_candidates(pic_q, candidate_texts, pic_model, args.gpu),
        "ours_retrieval": score_text_candidates(ours_q, candidate_texts, ours_model, args.gpu),
        "ours_merged": score_text_candidates(merged_q, candidate_texts, merged_model, args.gpu),
        "examples": [
            {
                "source": src_texts[i],
                "instruction": fwd_texts[i],
                "positive": tgt_texts[i],
                "source_negative": src_texts[i],
                "edit_wrong_source": tgt_texts[hard_idx[i]],
                "random_negative": tgt_texts[rand_idx[i]],
            }
            for i in range(min(8, n))
        ],
    }
    return results


def _content_word_count(text: str) -> int:
    words = [w for w in text.lower().replace(".", " ").replace(",", " ").split() if w not in {"a", "an", "the", "is", "are", "and", "with", "to", "of", "be", "more", "less"}]
    return len(words)


def load_fashioniq_samples(root: str, image_root: str, cloths: list[str], num_samples: int, seed: int, exclude_same: bool, min_content_words: int):
    root_path = Path(root) / "fashion-iq"
    img_root = Path(image_root) if image_root else root_path / "images"
    items = []
    for cloth in cloths:
        data = json.load(open(root_path / "json" / f"cap.{cloth}.val.json", "r", encoding="utf-8"))
        for d in data:
            joined = " ".join(d["captions"]).lower()
            if exclude_same and "same" in joined:
                continue
            if min_content_words > 0 and _content_word_count(joined) < min_content_words:
                continue
            items.append({
                "cloth": cloth,
                "ref_path": str(img_root / f"{d['candidate']}.png"),
                "tgt_path": str(img_root / f"{d['target']}.png"),
                "captions": tuple(d["captions"]),
                "prompt": f"a photo of fashion item that {d['captions'][0]} and {d['captions'][1]}",
            })
    rng = random.Random(seed)
    rng.shuffle(items)
    items = [x for x in items if Path(x["ref_path"]).exists() and Path(x["tgt_path"]).exists()]
    return items[:num_samples]


def run_fashioniq_proxy(args):
    cloths = [x.strip() for x in args.fashioniq_cloths.split(",") if x.strip()]
    samples = load_fashioniq_samples(
        args.fashioniq_root,
        args.fashioniq_image_root,
        cloths,
        args.num_samples,
        args.seed,
        args.fashioniq_exclude_same,
        args.fashioniq_min_content_words,
    )
    n = len(samples)

    oracle_model, oracle_preprocess = build_oracle_model(args.oracle_model, args.gpu)
    prompt_texts = [s["prompt"] for s in samples]
    prompt_feats = encode_texts(oracle_model, prompt_texts, args.gpu)

    hard_idx = []
    rng = random.Random(args.seed)
    sim = prompt_feats @ prompt_feats.t()
    for i in range(n):
        order = torch.argsort(sim[i], descending=True).tolist()
        chosen = None
        for j in order:
            if j == i:
                continue
            if samples[j]["ref_path"] != samples[i]["ref_path"] and samples[j]["tgt_path"] != samples[i]["tgt_path"]:
                chosen = j
                break
        if chosen is None:
            chosen = order[1]
        hard_idx.append(chosen)
    rand_idx = []
    for i in range(n):
        choices = [j for j in range(n) if j not in {i, hard_idx[i]}]
        rand_idx.append(rng.choice(choices))

    candidate_paths = [
        [samples[i]["tgt_path"], samples[i]["ref_path"], samples[hard_idx[i]]["tgt_path"], samples[rand_idx[i]]["tgt_path"]]
        for i in range(n)
    ]
    candidate_sets = {
        "full4": {
            "paths": candidate_paths,
            "negative_names": ["source_negative", "edit_wrong_source", "random_negative"],
        },
        "source2": {
            "paths": [[row[0], row[1]] for row in candidate_paths],
            "negative_names": ["source_negative"],
        },
        "edit2": {
            "paths": [[row[0], row[2]] for row in candidate_paths],
            "negative_names": ["edit_wrong_source"],
        },
    }

    oracle_q = encode_texts(oracle_model, prompt_texts, args.gpu)

    pic_args = build_model_args(args, "pic")
    pic_model, pic_img2text = build_retrieval_model(pic_args, ckpt_path=args.pic_ckpt)
    pic_images = torch.stack([oracle_preprocess(Image.open(s["ref_path"]).convert("RGB")) for s in samples], dim=0)
    pic_prompts = [f"a photo of * {args.pic_connector} {s['captions'][0]} and {s['captions'][1]}" for s in samples]
    pic_q = encode_image_anchor_query(pic_model, pic_img2text, pic_images, pic_prompts, args.gpu)

    ours_args = build_model_args(args, "ours")
    ours_model, ours_img2text = build_retrieval_model(ours_args, ckpt_path=args.ours_retrieval_ckpt)
    ours_images = torch.stack([oracle_preprocess(Image.open(s["ref_path"]).convert("RGB")) for s in samples], dim=0)
    ours_prompts = [f"a photo of * {args.ours_connector} {s['captions'][0]} and {s['captions'][1]}" for s in samples]
    ours_q = encode_image_anchor_query(ours_model, ours_img2text, ours_images, ours_prompts, args.gpu)

    merged_model, merged_img2text = build_retrieval_model(ours_args, ckpt_path=args.ours_merged_ckpt)
    merged_q = encode_image_anchor_query(merged_model, merged_img2text, ours_images, ours_prompts, args.gpu)

    results = {
        "mode": "fashioniq_proxy",
        "num_samples": n,
        "candidate_definition": {
            "positive": "true target image",
            "source_negative": "reference image",
            "edit_wrong_source": "target image from caption-similar but source-different sample",
            "random_negative": "random target image",
        },
        "examples": [
            {
                "cloth": samples[i]["cloth"],
                "prompt": samples[i]["prompt"],
                "positive": samples[i]["tgt_path"],
                "source_negative": samples[i]["ref_path"],
                "edit_wrong_source": samples[hard_idx[i]]["tgt_path"],
                "random_negative": samples[rand_idx[i]]["tgt_path"],
            }
            for i in range(min(8, n))
        ],
    }
    for name, cfg in candidate_sets.items():
        results[name] = {
            "oracle_clip_text": score_image_candidates(
                oracle_q, cfg["paths"], oracle_model, oracle_preprocess, args.gpu, cfg["negative_names"]
            ),
            "pic2word": score_image_candidates(
                pic_q, cfg["paths"], pic_model, oracle_preprocess, args.gpu, cfg["negative_names"]
            ),
            "ours_retrieval": score_image_candidates(
                ours_q, cfg["paths"], ours_model, oracle_preprocess, args.gpu, cfg["negative_names"]
            ),
            "ours_merged": score_image_candidates(
                merged_q, cfg["paths"], merged_model, oracle_preprocess, args.gpu, cfg["negative_names"]
            ),
        }
    return results


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "cc_exact":
        if not args.wds_shards or not args.cc3m_cir_jsonl:
            raise ValueError("cc_exact requires --wds-shards and --cc3m-cir-jsonl")
        results = run_cc_exact(args)
    else:
        results = run_fashioniq_proxy(args)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved oracle transfer probe to %s", out)


if __name__ == "__main__":
    main()
