#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import CIRR, CIRCODataset, CustomFolder, GeneCISDataset  # noqa: E402
from eval_retrieval import load_model  # noqa: E402
from third_party.open_clip.clip import tokenize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Find hard validation examples where merged DeCIR wins.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--genecis-batch-size", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="cirr,circo,genecis")
    parser.add_argument("--model", type=str, default="ViT-L/14")
    parser.add_argument("--img2text-arch", type=str, default="phi")
    parser.add_argument("--middle-dim", type=int, default=3072)
    parser.add_argument("--img2text-pretrained", type=str, required=True)
    parser.add_argument("--cirr-retrieval", type=Path, required=True)
    parser.add_argument("--cirr-joint", type=Path, required=True)
    parser.add_argument("--cirr-merged", type=Path, required=True)
    parser.add_argument("--multi-retrieval", type=Path, required=True)
    parser.add_argument("--multi-joint", type=Path, required=True)
    parser.add_argument("--multi-merged", type=Path, required=True)
    return parser.parse_args()


def build_eval_args(args, resume):
    return SimpleNamespace(
        model=args.model,
        middle_dim=args.middle_dim,
        n_layer=2,
        img2text_arch=args.img2text_arch,
        img2text_pretrained=str(args.img2text_pretrained),
        precision="amp",
        gpu=args.gpu,
        distributed=False,
        use_bn_sync=False,
        dp=False,
        multigpu=None,
        resume=str(resume),
        checkpoint_path="",
        logs=str(args.output_dir),
        name="hard_case_probe",
        no_lora=False,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.0,
    )


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_eval_model(args, resume):
    eval_args = build_eval_args(args, resume)
    model, img2text, preprocess = load_model(eval_args)
    model.eval()
    img2text.eval()
    return unwrap(model), img2text, preprocess


def unload(model, img2text):
    del model
    del img2text
    gc.collect()
    torch.cuda.empty_cache()


def rank_from_scores(scores, target_index):
    order = torch.argsort(scores, descending=True)
    rank = int((order == int(target_index)).nonzero(as_tuple=False)[0].item()) + 1
    top1 = int(order[0].item())
    return rank, top1, order[:10].detach().cpu().tolist()


def markdown_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def eval_cirr_model(args, resume):
    model, img2text, preprocess = load_eval_model(args, resume)
    root_project = REPO_ROOT / "data"
    source_dataset = CIRR(transforms=preprocess, root=str(root_project))
    target_dataset = CIRR(transforms=preprocess, root=str(root_project), mode="imgs")
    gallery_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    query_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    placeholder = tokenize(["*"])[0][1].item()

    gallery_features = []
    gallery_names = []
    with torch.no_grad():
        for images, paths in tqdm(gallery_loader, desc=f"CIRR gallery {Path(resume).name}", leave=False):
            images = images.cuda(args.gpu, non_blocking=True)
            gallery_features.append(normalize(model.encode_image(images)).cpu())
            gallery_names.extend([Path(str(p)).name for p in paths])
    gallery_features = torch.cat(gallery_features, dim=0).cuda(args.gpu, non_blocking=True)
    name_to_index = {name: i for i, name in enumerate(gallery_names)}

    results = {}
    offset = 0
    with torch.no_grad():
        for batch in tqdm(query_loader, desc=f"CIRR query {Path(resume).name}", leave=False):
            ref_images, ref_text_tokens, _, ref_names, target_names, captions, group_members = batch
            ref_images = ref_images.cuda(args.gpu, non_blocking=True)
            ref_text_tokens = ref_text_tokens.cuda(args.gpu, non_blocking=True)
            ref_feats = model.encode_image(ref_images)
            soft_tokens = img2text(ref_feats)
            query_feats = model.encode_text_img_retrieval(
                ref_text_tokens,
                soft_tokens,
                split_ind=placeholder,
                repeat=False,
            )
            query_feats = normalize(query_feats)
            sims = query_feats @ gallery_features.t()
            for i in range(sims.shape[0]):
                idx = offset + i
                target = str(target_names[i])
                target_index = name_to_index[target]
                rank, top1_index, top10_indices = rank_from_scores(sims[i], target_index)
                members = source_dataset.group_members[idx]
                member_indices = [name_to_index[m] for m in members if m in name_to_index]
                member_scores = sims[i, member_indices]
                member_order = torch.argsort(member_scores, descending=True)
                member_sorted_indices = [member_indices[int(j)] for j in member_order.detach().cpu().tolist()]
                subset_rank = member_sorted_indices.index(target_index) + 1 if target_index in member_sorted_indices else None
                results[idx] = {
                    "rank": rank,
                    "subset_rank": subset_rank,
                    "top1": gallery_names[top1_index],
                    "top10": [gallery_names[j] for j in top10_indices],
                    "subset_top1": gallery_names[member_sorted_indices[0]] if member_sorted_indices else None,
                }
            offset += sims.shape[0]
    meta = []
    for idx in range(len(source_dataset)):
        meta.append(
            {
                "dataset": "cirr",
                "query_index": idx,
                "caption": source_dataset.target_caps[idx],
                "reference": source_dataset.ref_imgs[idx],
                "target": source_dataset.target_imgs[idx],
                "group_members": source_dataset.group_members[idx],
                "reference_path": str(Path(source_dataset.root_img) / source_dataset.ref_imgs[idx]),
                "target_path": str(Path(source_dataset.root_img) / source_dataset.target_imgs[idx]),
            }
        )
    unload(model, img2text)
    return meta, results


def eval_circo_model(args, resume):
    model, img2text, preprocess = load_eval_model(args, resume)
    circo_root = REPO_ROOT / "data" / "CIRCO"
    gallery_path = circo_root / "COCO2017_unlabeled" / "unlabeled2017"
    gallery_dataset = CustomFolder(str(gallery_path), transform=preprocess)
    query_dataset = CIRCODataset(
        data_path=str(circo_root),
        split="val",
        mode="relative",
        transforms=preprocess,
        preprocess=preprocess,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    placeholder = tokenize(["*"])[0][1].item()

    gallery_features = []
    gallery_ids = []
    gallery_paths = []
    with torch.no_grad():
        for images, paths in tqdm(gallery_loader, desc=f"CIRCO gallery {Path(resume).name}", leave=False):
            images = images.cuda(args.gpu, non_blocking=True)
            gallery_features.append(normalize(model.encode_image(images)).cpu())
            for p in paths:
                gallery_paths.append(str(p))
                gallery_ids.append(int(os.path.basename(str(p)).split(".")[0]))
    gallery_features = torch.cat(gallery_features, dim=0).cuda(args.gpu, non_blocking=True)
    id_to_index = {img_id: i for i, img_id in enumerate(gallery_ids)}

    results = {}
    with torch.no_grad():
        for batch in tqdm(query_loader, desc=f"CIRCO query {Path(resume).name}", leave=False):
            ref_imgs = batch["reference_img"].cuda(args.gpu, non_blocking=True)
            captions = batch["relative_caption"]
            query_ids = batch["query_id"]
            prompts = [f"a photo of * that {cap}" for cap in captions]
            texts = tokenize(prompts).cuda(args.gpu, non_blocking=True)
            ref_feats = model.encode_image(ref_imgs)
            soft_tokens = img2text(ref_feats)
            query_feats = model.encode_text_img_retrieval(
                texts,
                soft_tokens,
                split_ind=placeholder,
                repeat=False,
            )
            query_feats = normalize(query_feats)
            sims = query_feats @ gallery_features.t()
            for i in range(sims.shape[0]):
                qid = int(query_ids[i].item()) if isinstance(query_ids[i], torch.Tensor) else int(query_ids[i])
                target = query_dataset.get_target_img_ids(qid)
                target_id = int(target["target_img_id"])
                target_index = id_to_index[target_id]
                rank, top1_index, top10_indices = rank_from_scores(sims[i], target_index)
                results[qid] = {
                    "rank": rank,
                    "top1": int(gallery_ids[top1_index]),
                    "top10": [int(gallery_ids[j]) for j in top10_indices],
                    "top1_in_gt": int(gallery_ids[top1_index]) in {int(x) for x in target["gt_img_ids"]},
                }
    meta = []
    for ann in query_dataset.annotations:
        qid = int(ann["id"])
        ref_id = int(ann["reference_img_id"])
        target_id = int(ann["target_img_id"])
        meta.append(
            {
                "dataset": "circo",
                "query_index": qid,
                "caption": ann["relative_caption"],
                "shared_concept": ann.get("shared_concept", ""),
                "semantic_aspects": ann.get("semantic_aspects", []),
                "reference": ref_id,
                "target": target_id,
                "gt_img_ids": [int(x) for x in ann.get("gt_img_ids", [])],
                "reference_path": gallery_paths[id_to_index[ref_id]],
                "target_path": gallery_paths[id_to_index[target_id]],
            }
        )
    unload(model, img2text)
    return meta, results


def eval_genecis_model(args, resume):
    model, img2text, preprocess = load_eval_model(args, resume)
    placeholder = tokenize(["*"])[0][1].item()
    json_root = "/data2/mingyu/genecis/genecis"
    out_meta = []
    out_results = {}
    for task in ["focus_attribute", "change_attribute", "focus_object", "change_object"]:
        img_root = str(REPO_ROOT / "data" / "genecis" / "VG_100K")
        if "object" in task:
            img_root = str(REPO_ROOT / "data" / "coco" / "val2017")
        dataset = GeneCISDataset(
            data_root=img_root,
            json_root=json_root,
            task=task,
            transforms=preprocess,
            tokenizer=None,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.genecis_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        offset = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"GeneCIS {task} {Path(resume).name}", leave=False):
                ref_imgs = batch["ref_img"].cuda(args.gpu, non_blocking=True)
                captions = batch["caption"]
                gallery_set = batch["gallery_set"].cuda(args.gpu, non_blocking=True)
                bsz, n_gallery, c, h, w = gallery_set.shape
                ref_feats = model.encode_image(ref_imgs)
                soft_tokens = img2text(ref_feats)
                prompts = [f"a photo of * , {cap}" for cap in captions]
                texts = tokenize(prompts).cuda(args.gpu, non_blocking=True)
                query_feats = model.encode_text_img_retrieval(
                    texts,
                    soft_tokens,
                    split_ind=placeholder,
                    repeat=False,
                )
                query_feats = normalize(query_feats)
                gallery_flat = gallery_set.view(bsz * n_gallery, c, h, w)
                gallery_feats = normalize(model.encode_image(gallery_flat)).view(bsz, n_gallery, -1)
                sims = torch.bmm(query_feats.unsqueeze(1), gallery_feats.transpose(1, 2)).squeeze(1)
                for i in range(bsz):
                    sample_idx = offset + i
                    rank, top1, top10 = rank_from_scores(sims[i], 0)
                    key = f"{task}:{sample_idx}"
                    out_results[key] = {
                        "rank": rank,
                        "top1": top1,
                        "top10": top10,
                    }
                offset += bsz
        for idx, sample in enumerate(dataset.val_samples):
            out_meta.append(
                {
                    "dataset": "genecis",
                    "task": task,
                    "query_index": f"{task}:{idx}",
                    "caption": sample.get("condition", ""),
                    "reference": sample.get("reference"),
                    "target": sample.get("target"),
                    "gallery": sample.get("gallery"),
                }
            )
    unload(model, img2text)
    return out_meta, out_results


def select_cases(meta, retrieval, joint, merged, max_cases, dataset):
    selected = []
    for item in meta:
        key = item["query_index"]
        if key not in retrieval or key not in joint or key not in merged:
            continue
        r = retrieval[key]
        j = joint[key]
        m = merged[key]
        merged_correct = m.get("rank") == 1 or m.get("subset_rank") == 1
        retrieval_wrong = r.get("rank", 999999) > 1
        joint_wrong = j.get("rank", 999999) > 1
        if not (merged_correct and retrieval_wrong and joint_wrong):
            continue
        case = dict(item)
        case.update(
            {
                "retrieval_rank": r.get("rank"),
                "joint_rank": j.get("rank"),
                "merged_rank": m.get("rank"),
                "retrieval_top1": r.get("top1"),
                "joint_top1": j.get("top1"),
                "merged_top1": m.get("top1"),
            }
        )
        if dataset == "cirr":
            case.update(
                {
                    "retrieval_subset_rank": r.get("subset_rank"),
                    "joint_subset_rank": j.get("subset_rank"),
                    "merged_subset_rank": m.get("subset_rank"),
                    "retrieval_subset_top1": r.get("subset_top1"),
                    "joint_subset_top1": j.get("subset_top1"),
                    "merged_subset_top1": m.get("subset_top1"),
                    "retrieval_top1_in_hard_group": r.get("top1") in set(item.get("group_members", [])),
                }
            )
        if dataset == "circo":
            case.update(
                {
                    "retrieval_top1_in_gt": r.get("top1_in_gt"),
                    "joint_top1_in_gt": j.get("top1_in_gt"),
                    "merged_top1_in_gt": m.get("top1_in_gt"),
                }
            )
        selected.append(case)

    def sort_key(x):
        shortcut_bonus = 0
        if x.get("retrieval_top1_in_hard_group") or x.get("retrieval_top1_in_gt"):
            shortcut_bonus = -1000
        return (
            x.get("merged_rank") or 999999,
            shortcut_bonus,
            x.get("retrieval_rank") or 999999,
            x.get("joint_rank") or 999999,
        )

    selected.sort(key=sort_key)
    return selected[:max_cases]


def write_outputs(output_dir, cases_by_dataset):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for dataset, cases in cases_by_dataset.items():
        path = output_dir / f"{dataset}_hard_cases.json"
        path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_cases.extend(cases)
    (output_dir / "hard_cases_all.json").write_text(
        json.dumps(all_cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Hard Distractor Evidence",
        "",
        "Selection rule: merged DeCIR is rank-1 correct, while retrieval-only and joint are not rank-1.",
        "For CIRR, subset rank and whether retrieval top-1 is inside the six-image hard group are also reported.",
        "For CIRCO, whether retrieval top-1 is inside the semantic ground-truth set is reported as a shortcut signal.",
        "",
    ]
    for dataset, cases in cases_by_dataset.items():
        lines.extend([f"## {dataset.upper()}", ""])
        if not cases:
            lines.extend(["No case matched the strict rule.", ""])
            continue
        rows = []
        for i, case in enumerate(cases[:10], start=1):
            rows.append(
                {
                    "#": i,
                    "query": case.get("query_index"),
                    "caption": str(case.get("caption", "")).replace("|", "/")[:100],
                    "ret_rank": case.get("retrieval_rank"),
                    "joint_rank": case.get("joint_rank"),
                    "merged_rank": case.get("merged_rank"),
                    "ret_top1": case.get("retrieval_top1"),
                    "target": case.get("target"),
                    "shortcut": case.get("retrieval_top1_in_hard_group", case.get("retrieval_top1_in_gt", "")),
                }
            )
        lines.extend(
            [
                markdown_table(
                    rows,
                    ["#", "query", "caption", "ret_rank", "joint_rank", "merged_rank", "ret_top1", "target", "shortcut"],
                ),
                "",
            ]
        )
    (output_dir / "HARD_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    selected = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    cases_by_dataset = {}

    if "cirr" in selected:
        meta, retrieval = eval_cirr_model(args, args.cirr_retrieval)
        _, joint = eval_cirr_model(args, args.cirr_joint)
        _, merged = eval_cirr_model(args, args.cirr_merged)
        cases_by_dataset["cirr"] = select_cases(meta, retrieval, joint, merged, args.max_cases, "cirr")

    if "circo" in selected:
        meta, retrieval = eval_circo_model(args, args.multi_retrieval)
        _, joint = eval_circo_model(args, args.multi_joint)
        _, merged = eval_circo_model(args, args.multi_merged)
        cases_by_dataset["circo"] = select_cases(meta, retrieval, joint, merged, args.max_cases, "circo")

    if "genecis" in selected:
        meta, retrieval = eval_genecis_model(args, args.multi_retrieval)
        _, joint = eval_genecis_model(args, args.multi_joint)
        _, merged = eval_genecis_model(args, args.multi_merged)
        cases_by_dataset["genecis"] = select_cases(meta, retrieval, joint, merged, args.max_cases, "genecis")

    write_outputs(args.output_dir, cases_by_dataset)


if __name__ == "__main__":
    main()
