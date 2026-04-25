import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import get_cc3m_cir_wds
from eval_retrieval import load_model
from main import seed_everything
from trainer import (
    _cosine_or_none,
    _extract_text_cproj_effective_grads,
    _project_block_vector,
    get_loss_geo_text_branch,
    get_loss_lcom_textprobe_cc3m,
)


def parse_args():
    parser = argparse.ArgumentParser("Offline Uni-X gradient conflict probe")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-L/14")
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", type=str, default="amp")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--probe-batches", type=int, default=60)
    parser.add_argument("--holdout-batches", type=int, default=20)
    parser.add_argument("--interference-step-size", type=float, default=1e-4)
    parser.add_argument("--middle-dim", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--geo-lora-r", type=int, default=64)
    parser.add_argument("--geo-lora-alpha", type=int, default=16)
    parser.add_argument("--geo-lora-dropout", type=float, default=0.0)
    parser.add_argument("--geo-weight", type=float, default=1.0)
    parser.add_argument("--wds-shards", type=str, required=True)
    parser.add_argument("--cc3m-cir-jsonl", type=str, required=True)
    parser.add_argument("--wds-image-key", type=str, default="jpg;png;jpeg;webp")
    parser.add_argument("--wds-text-key", type=str, default="txt;text;caption")
    parser.add_argument("--wds-shuffle", type=int, default=10000)
    parser.add_argument("--wds-shardshuffle", type=int, default=1000)
    parser.add_argument("--prompt-placeholder", type=str, default="*")
    parser.add_argument("--aggregate", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--dp", action="store_true", default=False)
    parser.add_argument("--openai-pretrained", action="store_true", default=False)
    parser.add_argument("--use-bn-sync", action="store_true", default=False)
    parser.add_argument("--multigpu", type=str, default=None)
    parser.add_argument("--logs", type=str, default="./logs/")
    parser.add_argument("--name", type=str, default="offline_probe")
    parser.add_argument("--checkpoint-path", type=str, default="./logs/offline_probe/checkpoints")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:6101")
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--no-lora", action="store_true", default=False)
    parser.add_argument("--pic2word-pretrained", type=str, default="/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt")
    parser.add_argument("--dataset-type", type=str, default="cc3m_cir_wds")
    parser.add_argument("--train-data", type=str, default="dummy")
    parser.add_argument("--instruction-dropout-prob", type=float, default=0.0)
    parser.add_argument("--wds-resampled", action="store_true", default=True)
    parser.add_argument("--wds-deterministic", action="store_true", default=True)
    parser.add_argument("--geo-embed-norm-eps", type=float, default=1e-6)
    parser.add_argument("--geo-delta-norm-eps", type=float, default=1e-4)
    parser.add_argument("--geo-delta-min-norm", type=float, default=1e-3)
    parser.add_argument("--geo-reverse-weight", type=float, default=0.25)
    parser.add_argument("--geo-reverse-margin", type=float, default=0.0)
    parser.add_argument("--geo-zero-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    return args


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "probe.log", encoding="utf-8"),
        ],
    )


def load_checkpoint(path, gpu):
    # Load entire checkpoint to CPU first to avoid OOM from optimizer states,
    # then let load_model move only the state_dict weights to GPU.
    return torch.load(path, map_location="cpu", weights_only=False)


def build_data_args(args):
    return SimpleNamespace(
        batch_size=args.batch_size,
        workers=args.workers,
        distributed=False,
        world_size=1,
        rank=0,
        wds_resampled=args.wds_resampled,
        wds_deterministic=args.wds_deterministic,
        seed=args.seed,
        wds_shards=args.wds_shards,
        cc3m_cir_jsonl=args.cc3m_cir_jsonl,
        cc3m_cir_reverse_jsonl=None,
        wds_image_key=args.wds_image_key,
        wds_text_key=args.wds_text_key,
        wds_shuffle=args.wds_shuffle,
        wds_shardshuffle=args.wds_shardshuffle,
        dataset_type=args.dataset_type,
        train_data=args.train_data,
        debug=False,
    )


def clone_batch(batch):
    return {
        "ref_img": batch["ref_img"].clone().cpu(),
        "instruction": list(batch["instruction"]),
        "modified_caption": list(batch["modified_caption"]),
        "src_caption": list(batch.get("src_caption", [""] * len(batch["instruction"]))),
        "reverse_instruction": list(batch.get("reverse_instruction", [""] * len(batch["instruction"]))),
    }


def move_batch_to_device(batch, gpu):
    out = clone_batch(batch)
    if torch.cuda.is_available():
        out["ref_img"] = out["ref_img"].cuda(gpu, non_blocking=True)
    return out


def select_items(seq, indices):
    return [seq[i] for i in indices]


def subset_tensor(tensor, indices):
    if len(indices) == 0:
        return tensor[:0]
    idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, idx)


def zero_grads(*modules):
    for module in modules:
        if module is not None:
            module.zero_grad(set_to_none=True)


def collect_named_lora_grads(module):
    grads = {}
    for name, param in module.named_parameters():
        if not (name.endswith(".A") or name.endswith(".B")):
            continue
        if param.grad is None:
            continue
        grads[name] = param.grad.detach().clone().float().cpu()
    return grads


def collect_text_cproj_lora_grads(module):
    grads = {}
    for name, param in module.named_parameters():
        if "transformer.resblocks." not in name or ".mlp.c_proj." not in name:
            continue
        if not (name.endswith(".A") or name.endswith(".B")):
            continue
        if param.grad is None:
            continue
        grads[name] = param.grad.detach().clone().float().cpu()
    return grads


def add_named_grads(target, grads):
    for name, grad in grads.items():
        if name not in target:
            target[name] = grad.clone()
        else:
            target[name].add_(grad)


def mean_named_grads(grad_sum, count):
    if count <= 0:
        return {}
    return {name: tensor / float(count) for name, tensor in grad_sum.items()}


def compute_joint_loss(model, img2text, batch, indices, loss_fn, args):
    images = subset_tensor(batch["ref_img"], indices)
    instructions = select_items(batch["instruction"], indices)
    modified = select_items(batch["modified_caption"], indices)
    src = select_items(batch["src_caption"], indices)
    reverse = select_items(batch["reverse_instruction"], indices)
    retrieval_loss = get_loss_lcom_textprobe_cc3m(model, img2text, images, instructions, modified, loss_fn, args)
    geo_loss, _, _ = get_loss_geo_text_branch(model, src, modified, instructions, reverse, args)
    return retrieval_loss + (float(getattr(args, "geo_weight", 1.0)) * geo_loss)


def accumulate_probe_batch(model, img2text, batch, loss_fn, args, block_sums, param_sums):
    zero_grads(model, img2text)
    ret_loss = get_loss_lcom_textprobe_cc3m(
        model,
        img2text,
        batch["ref_img"],
        batch["instruction"],
        batch["modified_caption"],
        loss_fn,
        args,
    )
    ret_loss.backward()
    ret_vectors = _extract_text_cproj_effective_grads(model)
    ret_param_grads = collect_text_cproj_lora_grads(model)

    zero_grads(model, img2text)
    edit_loss, _, _ = get_loss_geo_text_branch(
        model,
        batch["src_caption"],
        batch["modified_caption"],
        batch["instruction"],
        batch["reverse_instruction"],
        args,
    )
    edit_loss.backward()
    edit_vectors = _extract_text_cproj_effective_grads(model)
    edit_param_grads = collect_text_cproj_lora_grads(model)

    batch_size = batch["ref_img"].size(0)
    perm = torch.randperm(batch_size, device=batch["ref_img"].device).detach().cpu().tolist()
    split = max(1, batch_size // 2)
    idx_a = perm[:split]
    idx_b = perm[split:]
    if len(idx_b) == 0:
        idx_b = idx_a[-1:]
        idx_a = idx_a[:-1] or idx_b

    zero_grads(model, img2text)
    base_a_loss = get_loss_lcom_textprobe_cc3m(
        model, img2text,
        subset_tensor(batch["ref_img"], idx_a),
        select_items(batch["instruction"], idx_a),
        select_items(batch["modified_caption"], idx_a),
        loss_fn, args,
    )
    base_a_loss.backward()
    base_a_vectors = _extract_text_cproj_effective_grads(model)

    zero_grads(model, img2text)
    base_b_loss = get_loss_lcom_textprobe_cc3m(
        model, img2text,
        subset_tensor(batch["ref_img"], idx_b),
        select_items(batch["instruction"], idx_b),
        select_items(batch["modified_caption"], idx_b),
        loss_fn, args,
    )
    base_b_loss.backward()
    base_b_vectors = _extract_text_cproj_effective_grads(model)

    zero_grads(model, img2text)

    edit_proj_vectors = {}
    ret_biproj_vectors = {}
    for block_idx in ret_vectors:
        if block_idx in edit_vectors:
            edit_proj_vectors[block_idx] = _project_block_vector(ret_vectors[block_idx], edit_vectors[block_idx])
            ret_biproj_vectors[block_idx] = _project_block_vector(edit_vectors[block_idx], ret_vectors[block_idx])

    for key, vectors in (
        ("ret", ret_vectors),
        ("edit", edit_vectors),
        ("edit_proj", edit_proj_vectors),
        ("ret_biproj", ret_biproj_vectors),
        ("base_a", base_a_vectors),
        ("base_b", base_b_vectors),
    ):
        for block_idx, vec in vectors.items():
            if block_sums[key][block_idx] is None:
                block_sums[key][block_idx] = vec.clone().float()
            else:
                block_sums[key][block_idx].add_(vec.float())

    add_named_grads(param_sums["edit_raw"], edit_param_grads)
    add_named_grads(param_sums["ret_raw"], ret_param_grads)


def finalize_gc(block_sums, count):
    ret_avg = [None] * 12
    edit_avg = [None] * 12
    edit_proj_avg = [None] * 12
    ret_biproj_avg = [None] * 12
    base_a_avg = [None] * 12
    base_b_avg = [None] * 12
    s_base = [None] * 12
    s_cross = [None] * 12
    s_cross_after = [None] * 12
    s_cross_bidir = [None] * 12
    gc = [None] * 12
    gc_after = [None] * 12
    gc_bidir = [None] * 12

    for idx in range(12):
        if block_sums["ret"][idx] is not None:
            ret_avg[idx] = block_sums["ret"][idx] / float(count)
        if block_sums["edit"][idx] is not None:
            edit_avg[idx] = block_sums["edit"][idx] / float(count)
        if block_sums["edit_proj"][idx] is not None:
            edit_proj_avg[idx] = block_sums["edit_proj"][idx] / float(count)
        if block_sums["ret_biproj"][idx] is not None:
            ret_biproj_avg[idx] = block_sums["ret_biproj"][idx] / float(count)
        if block_sums["base_a"][idx] is not None:
            base_a_avg[idx] = block_sums["base_a"][idx] / float(count)
        if block_sums["base_b"][idx] is not None:
            base_b_avg[idx] = block_sums["base_b"][idx] / float(count)
        s_base[idx] = _cosine_or_none(base_a_avg[idx], base_b_avg[idx])
        s_cross[idx] = _cosine_or_none(ret_avg[idx], edit_avg[idx])
        s_cross_after[idx] = _cosine_or_none(ret_avg[idx], edit_proj_avg[idx])
        s_cross_bidir[idx] = _cosine_or_none(ret_biproj_avg[idx], edit_proj_avg[idx])
        if s_base[idx] is not None and s_cross[idx] is not None:
            gc[idx] = float(s_base[idx] - s_cross[idx])
        if s_base[idx] is not None and s_cross_after[idx] is not None:
            gc_after[idx] = float(s_base[idx] - s_cross_after[idx])
        if s_base[idx] is not None and s_cross_bidir[idx] is not None:
            gc_bidir[idx] = float(s_base[idx] - s_cross_bidir[idx])

    return {
        "ret_avg": ret_avg,
        "edit_avg": edit_avg,
        "edit_proj_avg": edit_proj_avg,
        "ret_biproj_avg": ret_biproj_avg,
        "base_a_avg": base_a_avg,
        "base_b_avg": base_b_avg,
        "s_base": s_base,
        "s_cross": s_cross,
        "s_cross_after": s_cross_after,
        "s_cross_bidir": s_cross_bidir,
        "gc": gc,
        "gc_after": gc_after,
        "gc_bidir": gc_bidir,
    }


def build_projected_param_grads(ret_avg_params, edit_avg_params):
    projected, _, _ = build_projected_param_grads_stats(ret_avg_params, edit_avg_params)
    return projected


def build_projected_param_grads_stats(ret_params, edit_params):
    projected = {}
    n_conflict = 0
    n_total = 0
    for name, edit_grad in edit_params.items():
        ret_grad = ret_params.get(name)
        if ret_grad is None:
            continue
        edit_flat = edit_grad.reshape(-1).float()
        ret_flat = ret_grad.reshape(-1).float()
        dot_val = torch.dot(edit_flat, ret_flat)
        n_total += 1
        proj = edit_flat.clone()
        if float(dot_val.item()) < 0.0:
            n_conflict += 1
            retr_norm_sq = torch.dot(ret_flat, ret_flat)
            if float(retr_norm_sq.item()) > 1e-12:
                proj = proj - (dot_val / retr_norm_sq) * ret_flat
        projected[name] = proj.reshape_as(edit_grad)
    ratio = n_conflict / n_total if n_total > 0 else 0.0
    return projected, n_conflict, ratio


def temporary_apply_grads(model, named_grads, step_size):
    applied = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in named_grads:
                continue
            delta = step_size * named_grads[name].to(device=param.device, dtype=param.dtype)
            param.add_(delta)
            applied.append((param, delta))
    return applied


def revert_applied(applied):
    with torch.no_grad():
        for param, delta in reversed(applied):
            param.sub_(delta)


def evaluate_interference(model, img2text, holdout_batches, loss_fn, args):
    raw_list = []
    proj_list = []
    proj_ratio_list = []
    for batch in holdout_batches:
        batch_gpu = move_batch_to_device(batch, args.gpu)

        zero_grads(model, img2text)
        with torch.no_grad():
            base_loss = get_loss_lcom_textprobe_cc3m(
                model, img2text,
                batch_gpu["ref_img"], batch_gpu["instruction"],
                batch_gpu["modified_caption"], loss_fn, args,
            )
            base_value = float(base_loss.detach().item())

        zero_grads(model, img2text)
        with torch.enable_grad():
            edit_loss, _, _ = get_loss_geo_text_branch(
                model, batch_gpu["src_caption"], batch_gpu["modified_caption"],
                batch_gpu["instruction"], batch_gpu["reverse_instruction"], args,
            )
            edit_loss.backward()
        raw_edit_grads = collect_named_lora_grads(model)

        zero_grads(model, img2text)
        with torch.enable_grad():
            ret_loss = get_loss_lcom_textprobe_cc3m(
                model, img2text,
                batch_gpu["ref_img"], batch_gpu["instruction"],
                batch_gpu["modified_caption"], loss_fn, args,
            )
            ret_loss.backward()
        ret_grads = collect_named_lora_grads(model)
        zero_grads(model, img2text)

        proj_edit_grads, n_conflict, ratio = build_projected_param_grads_stats(ret_grads, raw_edit_grads)
        proj_ratio_list.append(ratio)

        applied = temporary_apply_grads(model, raw_edit_grads, args.interference_step_size)
        with torch.no_grad():
            raw_after = get_loss_lcom_textprobe_cc3m(
                model, img2text,
                batch_gpu["ref_img"], batch_gpu["instruction"],
                batch_gpu["modified_caption"], loss_fn, args,
            )
            raw_list.append(float(raw_after.detach().item()) - base_value)
        revert_applied(applied)

        applied = temporary_apply_grads(model, proj_edit_grads, args.interference_step_size)
        with torch.no_grad():
            proj_after = get_loss_lcom_textprobe_cc3m(
                model, img2text,
                batch_gpu["ref_img"], batch_gpu["instruction"],
                batch_gpu["modified_caption"], loss_fn, args,
            )
            proj_list.append(float(proj_after.detach().item()) - base_value)
        revert_applied(applied)

        del batch_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "I_raw_list": raw_list,
        "I_proj_list": proj_list,
        "I_raw_mean": float(sum(raw_list) / len(raw_list)) if raw_list else None,
        "I_proj_mean": float(sum(proj_list) / len(proj_list)) if proj_list else None,
        "avg_proj_ratio": float(sum(proj_ratio_list) / len(proj_ratio_list)) if proj_ratio_list else None,
    }


def save_gc_tsv(output_dir, gc_result):
    path = output_dir / "gc_layerwise.tsv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("block\ts_base\ts_cross\ts_cross_after\ts_cross_bidir\tgc\tgc_after\tgc_bidir\n")
        for idx in range(12):
            s_base = gc_result["s_base"][idx]
            s_cross = gc_result["s_cross"][idx]
            s_cross_after = gc_result["s_cross_after"][idx]
            s_cross_bidir = gc_result["s_cross_bidir"][idx]
            gc = gc_result["gc"][idx]
            gc_after = gc_result["gc_after"][idx]
            gc_bidir = gc_result["gc_bidir"][idx]
            def _fmt(v):
                return '  ' if v is None else f'{v:.6f}'
            f.write(
                f"{idx}\t{_fmt(s_base)}\t{_fmt(s_cross)}\t{_fmt(s_cross_after)}\t"
                f"{_fmt(s_cross_bidir)}\t{_fmt(gc)}\t{_fmt(gc_after)}\t{_fmt(gc_bidir)}\n"
            )


def save_gc_plot(output_dir, gc_result):
    x = list(range(12))
    y_gc = [float("nan") if v is None else v for v in gc_result["gc"]]
    y_gc_after = [float("nan") if v is None else v for v in gc_result["gc_after"]]
    y_gc_bidir = [float("nan") if v is None else v for v in gc_result["gc_bidir"]]
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, y_gc, marker="o", linewidth=2, label="GC before projection")
    plt.plot(x, y_gc_after, marker="s", linewidth=2, linestyle="--", label="GC after (one-way PCGrad)")
    plt.plot(x, y_gc_bidir, marker="^", linewidth=2, linestyle=":", label="GC after (bidirectional PCGrad)")
    plt.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("Text ResBlock")
    plt.ylabel("Gradient Conflict")
    plt.title("Offline Uni-X Style Gradient Conflict")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gc_plot.png", dpi=220)
    plt.close()


def save_interference(output_dir, result):
    with open(output_dir / "interference.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    x = np.arange(len(result["I_raw_list"]))
    width = 0.38
    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, result["I_raw_list"], width=width, label="I(raw edit→ret)")
    plt.bar(x + width / 2, result["I_proj_list"], width=width, label="I(projected edit→ret)")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Holdout Batch Index")
    plt.ylabel("Retrieval loss delta")
    plt.title("Held-out Interference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "interference_plot.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    seed_everything(args.seed, rank_offset=0)
    logging.info("Loading checkpoint: %s", args.resume)

    model, img2text, preprocess_val = load_model(args)
    model.eval()
    img2text.eval()

    data_args = build_data_args(args)
    data_info = get_cc3m_cir_wds(data_args, preprocess_val, is_train=True)
    dataloader = data_info.dataloader
    iterator = iter(dataloader)
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda(args.gpu)

    probe_batches = []
    holdout_batches = []
    total_needed = args.probe_batches + args.holdout_batches
    logging.info("Collecting %d batches from deterministic WDS stream", total_needed)
    while len(probe_batches) + len(holdout_batches) < total_needed:
        batch = next(iterator)
        batch = clone_batch(batch)
        if len(probe_batches) < args.probe_batches:
            probe_batches.append(batch)
        else:
            holdout_batches.append(batch)

    block_sums = {
        "ret": [None] * 12,
        "edit": [None] * 12,
        "edit_proj": [None] * 12,
        "ret_biproj": [None] * 12,
        "base_a": [None] * 12,
        "base_b": [None] * 12,
    }
    param_sums = {
        "edit_raw": {},
        "ret_raw": {},
    }

    logging.info("Running offline gradient probe over %d batches", len(probe_batches))
    with torch.enable_grad():
        for idx, batch in enumerate(probe_batches, start=1):
            batch_gpu = move_batch_to_device(batch, args.gpu)
            accumulate_probe_batch(model, img2text, batch_gpu, loss_fn, args, block_sums, param_sums)
            del batch_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if idx % 10 == 0 or idx == len(probe_batches):
                logging.info("Processed probe batch %d / %d", idx, len(probe_batches))

    gc_result = finalize_gc(block_sums, len(probe_batches))

    logging.info("Evaluating held-out interference over %d batches", len(holdout_batches))
    interference = evaluate_interference(model, img2text, holdout_batches, loss_fn, args)

    summary = {
        "checkpoint": args.resume,
        "probe_batches": len(probe_batches),
        "holdout_batches": len(holdout_batches),
        "gc_macro_mean": float(sum(v for v in gc_result["gc"] if v is not None) / max(1, sum(1 for v in gc_result["gc"] if v is not None))),
        "gc_after_macro_mean": float(sum(v for v in gc_result["gc_after"] if v is not None) / max(1, sum(1 for v in gc_result["gc_after"] if v is not None))),
        "gc_bidir_macro_mean": float(sum(v for v in gc_result["gc_bidir"] if v is not None) / max(1, sum(1 for v in gc_result["gc_bidir"] if v is not None))),
        "I_raw_mean": interference["I_raw_mean"],
        "I_proj_mean": interference["I_proj_mean"],
        "avg_proj_ratio": interference["avg_proj_ratio"],
    }

    save_gc_tsv(output_dir, gc_result)
    save_gc_plot(output_dir, gc_result)
    save_interference(output_dir, interference)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Saved outputs to %s", output_dir)
    logging.info("GC macro mean = %.6f | GC_after macro mean = %.6f | GC_bidir macro mean = %.6f", summary["gc_macro_mean"], summary["gc_after_macro_mean"], summary["gc_bidir_macro_mean"])
    logging.info("I_raw_mean = %s | I_proj_mean = %s | avg_proj_ratio = %s", summary["I_raw_mean"], summary["I_proj_mean"], summary["avg_proj_ratio"])


if __name__ == "__main__":
    main()
