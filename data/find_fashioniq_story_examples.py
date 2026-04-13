#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from data import FashionIQ  # noqa: E402
from eval_utils import build_bidirectional_fashion_prompts  # noqa: E402
from real_text_geom_stats import build_retrieval_model  # noqa: E402
from third_party.open_clip.clip import _transform, tokenize  # noqa: E402
from model.clip import load as load_clip  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Find FashionIQ story examples where merged fixes retrieval failures.")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--fashioniq-root", type=str, default=str(ROOT / "data"))
    parser.add_argument("--cloths", type=str, default="dress,shirt,toptee")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--merged-ckpt", type=str, default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT / "logs" / "fashioniq_story_examples_step1000.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "docs" / "paper_examples" / "fashioniq_story_examples"),
    )
    return parser.parse_args()


def safe_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def strip_module_prefix(name: str) -> str:
    return name[len("module."):] if name.startswith("module.") else name


def is_text_lora_key(name: str) -> bool:
    core = strip_module_prefix(name)
    return (core.endswith(".A") or core.endswith(".B")) and not core.startswith("visual.")


def build_geo_proxy_checkpoint(full_ckpt_path: Path, geo_lora_path: Path) -> Path:
    ckpt = safe_load(full_ckpt_path)
    geo_lora = safe_load(geo_lora_path)
    state_dict = ckpt["state_dict"]
    norm_to_full = {strip_module_prefix(k): k for k in state_dict.keys() if is_text_lora_key(k)}
    replaced = 0
    for key, value in geo_lora.items():
        if not is_text_lora_key(key):
            continue
        actual_key = norm_to_full.get(strip_module_prefix(key))
        if actual_key is None:
            continue
        state_dict[actual_key] = value
        replaced += 1
    if replaced == 0:
        raise RuntimeError(f"No geo text LoRA keys matched for {geo_lora_path}")
    ckpt["state_dict"] = state_dict
    with tempfile.NamedTemporaryFile(prefix="fashioniq_geo_proxy_", suffix=".pt", dir="/tmp", delete=False) as tmp:
        out = Path(tmp.name)
    torch.save(ckpt, out)
    return out


def build_merged_checkpoint(raw_ckpt: Path, geo_ckpt: Path) -> Path:
    with tempfile.NamedTemporaryFile(prefix="fashioniq_merged_", suffix=".pt", dir="/tmp", delete=False) as tmp:
        out = Path(tmp.name)
    cmd = [
        sys.executable,
        str(ROOT / "data" / "merge_lora_ties.py"),
        "--ckpt-a", str(raw_ckpt),
        "--ckpt-b", str(geo_ckpt),
        "--output", str(out),
        "--weights", "0.5", "0.5",
        "--density", "0.9",
        "--text-only",
        "--base", "a",
        "--include-b-only",
        "--alpha-a", "16",
        "--rank-a", "64",
        "--alpha-b", "16",
        "--rank-b", "64",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not out.exists():
        raise RuntimeError(f"merge_lora_ties failed:\n{completed.stdout}")
    return out


def build_oracle_model(model_name: str, gpu: int):
    model, _, _ = load_clip(model_name, jit=False)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return model.to(device).float().eval()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


@torch.no_grad()
def encode_gallery(model, loader, gpu: int):
    all_paths = []
    all_feats = []
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    for images, paths in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        feats = l2_normalize(model.encode_image(images)).float().cpu()
        all_feats.append(feats)
        all_paths.extend(list(paths))
    feats = torch.cat(all_feats, dim=0)
    return feats, all_paths


@torch.no_grad()
def encode_oracle_queries(model, captions, gpu: int):
    prompts_fwd, prompts_rev = build_bidirectional_fashion_prompts(captions)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    toks_fwd = tokenize(prompts_fwd, truncate=True).to(device)
    toks_rev = tokenize(prompts_rev, truncate=True).to(device)
    feats_fwd = l2_normalize(model.encode_text(toks_fwd))
    feats_rev = l2_normalize(model.encode_text(toks_rev))
    return l2_normalize((feats_fwd + feats_rev) / 2).float().cpu()


@torch.no_grad()
def encode_image_grounded_queries(model, img2text, ref_images, captions, gpu: int):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    ref_images = ref_images.to(device=device, dtype=torch.float32, non_blocking=True)
    ref_feats = model.encode_image(ref_images)
    query_tokens = img2text(ref_feats)
    prompts_fwd, prompts_rev = build_bidirectional_fashion_prompts(captions)
    toks_fwd = tokenize(prompts_fwd, truncate=True).to(device)
    toks_rev = tokenize(prompts_rev, truncate=True).to(device)
    split_ind = int(tokenize(["*"])[0][1].item())
    feats_fwd = l2_normalize(model.encode_text_img_retrieval(toks_fwd, query_tokens, split_ind=split_ind, repeat=False))
    feats_rev = l2_normalize(model.encode_text_img_retrieval(toks_rev, query_tokens, split_ind=split_ind, repeat=False))
    return l2_normalize((feats_fwd + feats_rev) / 2).float().cpu()


def rank_targets(query_feats: torch.Tensor, gallery_feats: torch.Tensor, gallery_paths: list[str], answer_paths: list[str], topk: int):
    sims = query_feats @ gallery_feats.t()
    sorted_indices = torch.argsort(sims, dim=-1, descending=True)
    topk_indices = sorted_indices[:, :topk]
    topk_paths = [[gallery_paths[j] for j in row.tolist()] for row in topk_indices]
    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    answer_to_rank = []
    for i, answer in enumerate(answer_paths):
        target_idx = path_to_idx.get(answer)
        if target_idx is None:
            answer_to_rank.append(None)
            continue
        match = torch.nonzero(sorted_indices[i] == target_idx, as_tuple=False)
        rank = int(match[0].item()) + 1 if match.numel() > 0 else None
        answer_to_rank.append(rank)
    return {
        "sims": sims,
        "sorted_indices": sorted_indices,
        "topk_paths": topk_paths,
        "ranks": answer_to_rank,
    }


def choose_hard_negative_indices(prompt_feats: torch.Tensor, samples: list[dict]):
    sim = prompt_feats @ prompt_feats.t()
    hard_idx = []
    rand_idx = []
    g = torch.Generator().manual_seed(3407)
    n = sim.shape[0]
    for i in range(n):
        order = torch.argsort(sim[i], descending=True).tolist()
        chosen = None
        for j in order:
            if j == i:
                continue
            if samples[j]["ref_path"] != samples[i]["ref_path"] and samples[j]["target_path"] != samples[i]["target_path"]:
                chosen = j
                break
        if chosen is None:
            chosen = order[1]
        hard_idx.append(chosen)
        candidates = [j for j in range(n) if j not in {i, chosen}]
        rand = int(candidates[torch.randint(len(candidates), (1,), generator=g).item()])
        rand_idx.append(rand)
    return hard_idx, rand_idx


def score_candidate_set(query_feat: torch.Tensor, path_list: list[str], feat_map: dict[str, torch.Tensor]):
    cand_feats = torch.stack([feat_map[p] for p in path_list], dim=0)
    scores = (query_feat.unsqueeze(0) @ cand_feats.t()).squeeze(0)
    return scores.tolist(), int(scores.argmax().item())


def ensure_feature_map(paths: list[str], model, preprocess, gpu: int, feat_map: dict[str, torch.Tensor]):
    missing = [p for p in paths if p not in feat_map]
    if not missing:
        return
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in missing], dim=0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        feats = l2_normalize(model.encode_image(batch)).float().cpu()
    for p, feat in zip(missing, feats):
        feat_map[p] = feat


def copy_case_images(case: dict, output_dir: Path):
    case_dir = output_dir / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "reference.png": case["reference_image"],
        "target_gt.png": case["target_image"],
        "retrieval_top1.png": case["retrieval_top1"],
        "geo_top1.png": case["geo_top1"],
        "pic2word_top1.png": case["pic2word_top1"],
        "merged_top1.png": case["merged_top1"],
        "oracle_top1.png": case["oracle_top1"],
    }
    for name, src in files.items():
        if src and Path(src).exists():
            shutil.copy2(src, case_dir / name)


def main():
    args = parse_args()
    gpu = args.gpu
    output_json = Path(args.output_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_root = ROOT / "logs" / "DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_NoDrop_FashionIQ"
    ckpt_dir = run_root / "checkpoints"
    retrieval_ckpt = ckpt_dir / f"epoch_0_step_{args.step}.pt"
    geo_ckpt = ckpt_dir / f"epoch_0_step_{args.step}_geo_lora_ema.pt"
    if not retrieval_ckpt.exists() or not geo_ckpt.exists():
        raise FileNotFoundError(f"Missing FashionIQ step checkpoint(s): {retrieval_ckpt} / {geo_ckpt}")

    temp_geo_proxy = build_geo_proxy_checkpoint(retrieval_ckpt, geo_ckpt)
    merged_ckpt_override = Path(args.merged_ckpt) if args.merged_ckpt else None
    temp_merged = merged_ckpt_override if merged_ckpt_override and merged_ckpt_override.exists() else build_merged_checkpoint(retrieval_ckpt, geo_ckpt)

    try:
        print(f"[info] retrieval_ckpt={retrieval_ckpt}", flush=True)
        print(f"[info] geo_ckpt={geo_ckpt}", flush=True)
        print(f"[info] merged_ckpt={temp_merged}", flush=True)
        # All image-grounded models in this search use ViT-L/14 + no-pad FashionIQ protocol.
        ours_args = argparse.Namespace(
            model="ViT-L/14",
            img2text_arch="phi",
            middle_dim=3072,
            n_layer=2,
            lora_r=64,
            lora_alpha=16,
            lora_dropout=0.0,
            geo_lora_r=64,
            geo_lora_alpha=16,
            geo_lora_dropout=0.0,
            gpu=gpu,
            retrieval_prompt_connector="that",
            num_samples=0,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=3407,
            wds_shards="",
            cc3m_cir_jsonl="",
            wds_image_key="jpg;png;jpeg;webp",
            wds_text_key="txt;text;caption",
            wds_shuffle=10000,
            wds_shardshuffle=1000,
            wds_resampled=True,
            wds_deterministic=True,
            use_image_anchor=True,
        )
        pic_args = argparse.Namespace(
            model="ViT-L/14",
            img2text_arch="im2text",
            middle_dim=512,
            n_layer=2,
            lora_r=64,
            lora_alpha=16,
            lora_dropout=0.0,
            geo_lora_r=64,
            geo_lora_alpha=16,
            geo_lora_dropout=0.0,
            gpu=gpu,
            retrieval_prompt_connector="that",
            num_samples=0,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=3407,
            wds_shards="",
            cc3m_cir_jsonl="",
            wds_image_key="jpg;png;jpeg;webp",
            wds_text_key="txt;text;caption",
            wds_shuffle=10000,
            wds_shardshuffle=1000,
            wds_resampled=True,
            wds_deterministic=True,
            use_image_anchor=True,
        )

        oracle_model = build_oracle_model("ViT-L/14", gpu)
        ours_retrieval_model, ours_retrieval_img2text = build_retrieval_model(ours_args, ckpt_path=str(retrieval_ckpt))
        ours_geo_model, ours_geo_img2text = build_retrieval_model(ours_args, ckpt_path=str(temp_geo_proxy))
        ours_merged_model, ours_merged_img2text = build_retrieval_model(ours_args, ckpt_path=str(temp_merged))
        pic_model, pic_img2text = build_retrieval_model(pic_args, ckpt_path=str(ROOT / "checkpoint" / "pic2word_model.pt"))

        preprocess = _transform(224)
        cloths = [x.strip() for x in args.cloths.split(",") if x.strip()]
        all_cases = []
        summary = defaultdict(int)

        for cloth in cloths:
            print(f"[info] processing cloth={cloth}", flush=True)
            source_dataset = FashionIQ(
                cloth=cloth,
                transforms=preprocess,
                root=args.fashioniq_root,
                is_return_target_path=True,
            )
            target_dataset = FashionIQ(
                cloth=cloth,
                transforms=preprocess,
                root=args.fashioniq_root,
                mode="imgs",
            )
            source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
            target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

            ours_gallery_feats, gallery_paths = encode_gallery(ours_retrieval_model, target_loader, gpu)
            oracle_gallery_feats, oracle_gallery_paths = encode_gallery(oracle_model, target_loader, gpu)
            pic_gallery_feats, pic_gallery_paths = encode_gallery(pic_model, target_loader, gpu)
            if gallery_paths != oracle_gallery_paths or gallery_paths != pic_gallery_paths:
                raise RuntimeError(f"Gallery path mismatch for cloth={cloth}")

            ours_feat_map = {p: f for p, f in zip(gallery_paths, ours_gallery_feats)}
            oracle_feat_map = {p: f for p, f in zip(gallery_paths, oracle_gallery_feats)}
            pic_feat_map = {p: f for p, f in zip(gallery_paths, pic_gallery_feats)}

            samples = []
            query_feats = {
                "oracle": [],
                "pic2word": [],
                "retrieval": [],
                "geo": [],
                "merged": [],
            }

            for batch in source_loader:
                ref_images, _target_images, _target_caption, _caption_only, answer_paths, ref_names, captions = batch
                oracle_q = encode_oracle_queries(oracle_model, captions, gpu)
                pic_q = encode_image_grounded_queries(pic_model, pic_img2text, ref_images, captions, gpu)
                retrieval_q = encode_image_grounded_queries(ours_retrieval_model, ours_retrieval_img2text, ref_images, captions, gpu)
                geo_q = encode_image_grounded_queries(ours_geo_model, ours_geo_img2text, ref_images, captions, gpu)
                merged_q = encode_image_grounded_queries(ours_merged_model, ours_merged_img2text, ref_images, captions, gpu)

                query_feats["oracle"].append(oracle_q)
                query_feats["pic2word"].append(pic_q)
                query_feats["retrieval"].append(retrieval_q)
                query_feats["geo"].append(geo_q)
                query_feats["merged"].append(merged_q)

                caps_list = list(zip(*captions)) if isinstance(captions, (list, tuple)) and len(captions) == 2 else list(captions)
                for ref_path, target_path, pair in zip(ref_names, answer_paths, caps_list):
                    samples.append({
                        "cloth": cloth,
                        "ref_path": str(ref_path),
                        "target_path": str(target_path),
                        "captions": [str(pair[0]), str(pair[1])],
                        "prompt": f"a photo of fashion item that {pair[0]} and {pair[1]}",
                    })

            query_feats = {k: torch.cat(v, dim=0) for k, v in query_feats.items()}
            answer_paths = [s["target_path"] for s in samples]

            oracle_rank = rank_targets(query_feats["oracle"], oracle_gallery_feats, gallery_paths, answer_paths, args.topk)
            pic_rank = rank_targets(query_feats["pic2word"], pic_gallery_feats, gallery_paths, answer_paths, args.topk)
            retrieval_rank = rank_targets(query_feats["retrieval"], ours_gallery_feats, gallery_paths, answer_paths, args.topk)
            geo_rank = rank_targets(query_feats["geo"], ours_gallery_feats, gallery_paths, answer_paths, args.topk)
            merged_rank = rank_targets(query_feats["merged"], ours_gallery_feats, gallery_paths, answer_paths, args.topk)

            prompt_feats = encode_oracle_queries(oracle_model, [tuple(s["captions"]) for s in samples], gpu)
            hard_idx, rand_idx = choose_hard_negative_indices(prompt_feats, samples)

            ensure_feature_map([s["ref_path"] for s in samples], ours_retrieval_model, preprocess, gpu, ours_feat_map)
            ensure_feature_map([s["ref_path"] for s in samples], oracle_model, preprocess, gpu, oracle_feat_map)
            ensure_feature_map([s["ref_path"] for s in samples], pic_model, preprocess, gpu, pic_feat_map)

            for i, sample in enumerate(samples):
                cand_paths = [
                    sample["target_path"],
                    sample["ref_path"],
                    samples[hard_idx[i]]["target_path"],
                    samples[rand_idx[i]]["target_path"],
                ]
                cand_labels = ["positive", "source_negative", "edit_wrong_source", "random_negative"]
                ensure_feature_map(cand_paths, ours_retrieval_model, preprocess, gpu, ours_feat_map)
                ensure_feature_map(cand_paths, oracle_model, preprocess, gpu, oracle_feat_map)
                ensure_feature_map(cand_paths, pic_model, preprocess, gpu, pic_feat_map)

                oracle_scores, oracle_top = score_candidate_set(query_feats["oracle"][i], cand_paths, oracle_feat_map)
                pic_scores, pic_top = score_candidate_set(query_feats["pic2word"][i], cand_paths, pic_feat_map)
                retrieval_scores, retrieval_top = score_candidate_set(query_feats["retrieval"][i], cand_paths, ours_feat_map)
                geo_scores, geo_top = score_candidate_set(query_feats["geo"][i], cand_paths, ours_feat_map)
                merged_scores, merged_top = score_candidate_set(query_feats["merged"][i], cand_paths, ours_feat_map)

                record = {
                    "cloth": cloth,
                    "query_index": i,
                    "step": args.step,
                    "reference_image": sample["ref_path"],
                    "target_image": sample["target_path"],
                    "captions": sample["captions"],
                    "prompt": sample["prompt"],
                    "full_gallery": {
                        "oracle_rank": oracle_rank["ranks"][i],
                        "pic2word_rank": pic_rank["ranks"][i],
                        "retrieval_rank": retrieval_rank["ranks"][i],
                        "geo_rank": geo_rank["ranks"][i],
                        "merged_rank": merged_rank["ranks"][i],
                        "oracle_topk": oracle_rank["topk_paths"][i],
                        "pic2word_topk": pic_rank["topk_paths"][i],
                        "retrieval_topk": retrieval_rank["topk_paths"][i],
                        "geo_topk": geo_rank["topk_paths"][i],
                        "merged_topk": merged_rank["topk_paths"][i],
                    },
                    "candidate_set": {
                        "positive": cand_paths[0],
                        "source_negative": cand_paths[1],
                        "edit_wrong_source": cand_paths[2],
                        "random_negative": cand_paths[3],
                        "oracle_top1_label": cand_labels[oracle_top],
                        "pic2word_top1_label": cand_labels[pic_top],
                        "retrieval_top1_label": cand_labels[retrieval_top],
                        "geo_top1_label": cand_labels[geo_top],
                        "merged_top1_label": cand_labels[merged_top],
                        "oracle_scores": dict(zip(cand_labels, oracle_scores)),
                        "pic2word_scores": dict(zip(cand_labels, pic_scores)),
                        "retrieval_scores": dict(zip(cand_labels, retrieval_scores)),
                        "geo_scores": dict(zip(cand_labels, geo_scores)),
                        "merged_scores": dict(zip(cand_labels, merged_scores)),
                    },
                    "oracle_top1": oracle_rank["topk_paths"][i][0],
                    "pic2word_top1": pic_rank["topk_paths"][i][0],
                    "retrieval_top1": retrieval_rank["topk_paths"][i][0],
                    "geo_top1": geo_rank["topk_paths"][i][0],
                    "merged_top1": merged_rank["topk_paths"][i][0],
                }

                merged_good = record["full_gallery"]["merged_rank"] == 1
                others_bad = (
                    record["full_gallery"]["pic2word_rank"] != 1
                    and record["full_gallery"]["retrieval_rank"] != 1
                    and record["full_gallery"]["geo_rank"] != 1
                )
                oracle_good = record["candidate_set"]["oracle_top1_label"] == "positive"
                strict_story = (
                    merged_good
                    and others_bad
                    and oracle_good
                    and record["candidate_set"]["retrieval_top1_label"] == "source_negative"
                    and record["candidate_set"]["geo_top1_label"] == "edit_wrong_source"
                    and record["candidate_set"]["merged_top1_label"] == "positive"
                    and record["candidate_set"]["pic2word_top1_label"] != "positive"
                )
                relaxed_story = (
                    merged_good
                    and others_bad
                    and oracle_good
                    and record["candidate_set"]["merged_top1_label"] == "positive"
                )
                record["strict_story"] = strict_story
                record["relaxed_story"] = relaxed_story
                if strict_story:
                    summary["strict_story_count"] += 1
                if relaxed_story:
                    summary["relaxed_story_count"] += 1
                all_cases.append(record)

        strict_cases = [x for x in all_cases if x["strict_story"]]
        relaxed_cases = [x for x in all_cases if x["relaxed_story"]]
        strict_cases.sort(key=lambda x: (x["full_gallery"]["pic2word_rank"] + x["full_gallery"]["retrieval_rank"] + x["full_gallery"]["geo_rank"]), reverse=True)
        relaxed_cases.sort(key=lambda x: (x["full_gallery"]["pic2word_rank"] + x["full_gallery"]["retrieval_rank"] + x["full_gallery"]["geo_rank"]), reverse=True)

        selected = strict_cases[: args.max_cases]
        if len(selected) < args.max_cases:
            extra = [x for x in relaxed_cases if x not in selected]
            selected.extend(extra[: args.max_cases - len(selected)])

        for idx, case in enumerate(selected, start=1):
            case["case_id"] = f"case_{idx:02d}_{case['cloth']}_step{case['step']}"
            copy_case_images(case, output_dir)

        result = {
            "fashioniq_run": str(run_root),
            "step": args.step,
            "num_total_queries": len(all_cases),
            "num_strict_story": len(strict_cases),
            "num_relaxed_story": len(relaxed_cases),
            "selection_policy": {
                "strict_story": "merged full-gallery top1 correct; pic2word/retrieval/geo all wrong; oracle candidate top1 positive; retrieval candidate top1 source_negative; geo candidate top1 edit_wrong_source; merged candidate top1 positive",
                "relaxed_story": "merged full-gallery top1 correct; pic2word/retrieval/geo all wrong; oracle candidate top1 positive; merged candidate top1 positive",
            },
            "selected_cases": selected,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "num_total_queries": result["num_total_queries"],
            "num_strict_story": result["num_strict_story"],
            "num_relaxed_story": result["num_relaxed_story"],
            "selected_case_ids": [x["case_id"] for x in selected],
        }, indent=2, ensure_ascii=False))
    finally:
        Path(temp_geo_proxy).unlink(missing_ok=True)
        if not merged_ckpt_override:
            Path(temp_merged).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
