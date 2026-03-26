#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
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

from eval_retrieval import load_model  # noqa: E402
from eval_utils import compute_metrics, get_metrics_fashion  # noqa: E402
from data import CIRCODataset, CustomFolder, FashionIQ, GeneCISDataset  # noqa: E402
from third_party.open_clip.clip import tokenize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on FashionIQ / CIRCO / GeneCIS.")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--genecis-batch-size", type=int, default=32)
    parser.add_argument("--name", type=str, default="multidataset_eval")
    parser.add_argument("--logs", type=str, default=str(REPO_ROOT / "logs"))
    return parser.parse_args()


def build_eval_args(args):
    return SimpleNamespace(
        model="ViT-L/14",
        middle_dim=512,
        n_layer=2,
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


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def eval_fashion_composed(model, img2text, preprocess, gpu, batch_size, workers):
    root_project = REPO_ROOT / "data"
    m = unwrap_model(model)
    id_split = tokenize(["*"])[0][1]
    out = {}
    for cloth in ["dress", "shirt", "toptee"]:
        source_dataset = FashionIQ(
            cloth=cloth,
            transforms=preprocess,
            root=str(root_project),
            is_return_target_path=True,
        )
        target_dataset = FashionIQ(
            cloth=cloth,
            transforms=preprocess,
            root=str(root_project),
            mode="imgs",
        )
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        all_target_paths = []
        all_answer_paths = []
        all_image_features = []
        all_composed_features = []

        with torch.no_grad():
            for target_images, target_paths in tqdm(target_loader, desc=f"FashionIQ-{cloth}-gallery", leave=False):
                target_images = target_images.cuda(gpu, non_blocking=True)
                image_features = normalize(m.encode_image(target_images))
                all_image_features.append(image_features)
                all_target_paths.extend(target_paths)

            for batch in tqdm(source_loader, desc=f"FashionIQ-{cloth}-query", leave=False):
                ref_images, _, target_caption, _, answer_paths, _, _ = batch
                ref_images = ref_images.cuda(gpu, non_blocking=True)
                target_caption = target_caption.cuda(gpu, non_blocking=True)
                query_image_features = m.encode_image(ref_images)
                query_image_tokens = img2text(query_image_features)
                composed_feature = m.encode_text_img_retrieval(
                    target_caption,
                    query_image_tokens,
                    split_ind=id_split,
                    repeat=False,
                )
                composed_feature = normalize(composed_feature)
                all_composed_features.append(composed_feature)
                all_answer_paths.extend(answer_paths)

        metrics = get_metrics_fashion(
            image_features=torch.cat(all_image_features, dim=0),
            ref_features=torch.cat(all_composed_features, dim=0),
            target_names=all_target_paths,
            answer_names=all_answer_paths,
        )
        out[cloth] = {k: float(v) for k, v in metrics.items()}
    return out


def eval_circo_val(model, img2text, preprocess, gpu, batch_size, workers):
    circo_root = REPO_ROOT / "data" / "CIRCO"
    gallery_path = circo_root / "COCO2017_unlabeled" / "unlabeled2017"
    gallery_dataset = CustomFolder(str(gallery_path), transform=preprocess)
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    query_dataset = CIRCODataset(
        data_path=str(circo_root),
        split="val",
        mode="relative",
        transforms=preprocess,
        preprocess=preprocess,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    m = unwrap_model(model)
    placeholder_token_id = tokenize(["*"])[0][1].item()

    gallery_features = []
    gallery_img_ids = []
    with torch.no_grad():
        for images, paths in tqdm(gallery_loader, desc="CIRCO-gallery", leave=False):
            images = images.cuda(gpu, non_blocking=True)
            image_features = normalize(m.encode_image(images))
            gallery_features.append(image_features)
            for p in paths:
                gallery_img_ids.append(int(os.path.basename(p).split(".")[0]))
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_img_ids = np.array(gallery_img_ids)

    predictions_dict = {}
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="CIRCO-query", leave=False):
            ref_imgs = batch["reference_img"].cuda(gpu, non_blocking=True)
            relative_caps = batch["relative_caption"]
            query_ids = batch["query_id"]
            prompts = [f"a photo of * that {cap}" for cap in relative_caps]
            texts = tokenize(prompts).cuda(gpu, non_blocking=True)
            ref_img_feats = m.encode_image(ref_imgs)
            soft_tokens = img2text(ref_img_feats)
            query_feats = m.encode_text_img_retrieval(
                texts,
                soft_tokens,
                split_ind=placeholder_token_id,
                repeat=False,
            )
            query_feats = normalize(query_feats)
            sim_matrix = query_feats @ gallery_features.t()
            _, topk_indices = torch.topk(sim_matrix, k=50, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            for i, q_id in enumerate(query_ids):
                q_id_int = int(q_id.item()) if isinstance(q_id, torch.Tensor) else int(q_id)
                predictions_dict[q_id_int] = gallery_img_ids[topk_indices[i]].tolist()

    map_atk, recall_atk, semantic_map_at10 = compute_metrics(
        circo_root,
        predictions_dict,
        [5, 10, 25, 50],
    )
    semantic_mean = float(np.mean(list(semantic_map_at10.values())))
    return {
        "map": {f"mAP@{k}": float(v) for k, v in map_atk.items()},
        "recall": {f"R@{k}": float(v) for k, v in recall_atk.items()},
        "semantic_map_at10": {k: float(v) for k, v in semantic_map_at10.items()},
        "semantic_map_at10_mean": semantic_mean,
    }


def eval_genecis(model, img2text, preprocess, gpu, batch_size, workers):
    img_root = str(REPO_ROOT / "data" / "genecis" / "VG_100K")
    json_root = "/data2/mingyu/genecis/genecis"
    m = unwrap_model(model)
    placeholder_token_id = tokenize(["*"])[0][1].item()
    out = {}
    for task in ["focus_attribute", "change_attribute", "focus_object", "change_object"]:
        if "object" in task:
            img_root_task = "/data2/mingyu/composed_image_retrieval/data/coco/val2017"
        else:
            img_root_task = img_root
        dataset = GeneCISDataset(
            data_root=img_root_task,
            json_root=json_root,
            task=task,
            transforms=preprocess,
            tokenizer=None,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        metrics = {1: 0, 2: 0, 3: 0}
        total_count = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"GeneCIS-{task}", leave=False):
                ref_imgs = batch["ref_img"].cuda(gpu, non_blocking=True)
                captions = batch["caption"]
                gallery_set = batch["gallery_set"].cuda(gpu, non_blocking=True)
                bsz, n_gallery, c, h, w = gallery_set.shape

                ref_feats = m.encode_image(ref_imgs)
                soft_tokens = img2text(ref_feats)
                prompts = [f"a photo of * , {cap}" for cap in captions]
                texts = tokenize(prompts).cuda(gpu, non_blocking=True)
                query_feats = m.encode_text_img_retrieval(
                    texts,
                    soft_tokens,
                    split_ind=placeholder_token_id,
                    repeat=False,
                )
                query_feats = normalize(query_feats)
                gallery_flat = gallery_set.view(bsz * n_gallery, c, h, w)
                gallery_feats = normalize(m.encode_image(gallery_flat)).view(bsz, n_gallery, -1)
                sims = torch.bmm(query_feats.unsqueeze(1), gallery_feats.transpose(1, 2)).squeeze(1)
                _, sorted_idxs = sims.sort(dim=-1, descending=True)
                for k in [1, 2, 3]:
                    top_k = sorted_idxs[:, :k]
                    hits = (top_k == 0).any(dim=1)
                    metrics[k] += hits.sum().item()
                total_count += bsz
        out[task] = {f"R@{k}": float(metrics[k] / max(total_count, 1) * 100.0) for k in [1, 2, 3]}
    return out


def main():
    cli_args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    eval_args = build_eval_args(cli_args)
    model, img2text, preprocess = load_model(eval_args)
    model.eval()
    img2text.eval()

    result = {
        "resume": cli_args.resume,
        "fashioniq": eval_fashion_composed(
            model,
            img2text,
            preprocess,
            cli_args.gpu,
            cli_args.batch_size,
            cli_args.workers,
        ),
        "circo_val": eval_circo_val(
            model,
            img2text,
            preprocess,
            cli_args.gpu,
            cli_args.batch_size,
            cli_args.workers,
        ),
        "genecis": eval_genecis(
            model,
            img2text,
            preprocess,
            cli_args.gpu,
            cli_args.genecis_batch_size,
            cli_args.workers,
        ),
    }

    output_path = Path(cli_args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
