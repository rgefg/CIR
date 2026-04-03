#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


STEP_RAW_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$")
STEP_EMA_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)_ema\.pt$")
FINAL_RAW_RE = re.compile(r"epoch_(?P<epoch>\d+)\.pt$")
FINAL_EMA_RE = re.compile(r"epoch_(?P<epoch>\d+)_ema\.pt$")
FEATURE_RE = re.compile(r"Eval\s+(\w+)\s+Feature")
METRIC_RE = re.compile(r"(recall_[^:]+):\s*([0-9.]+)")

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = REPO_ROOT / "data" / "merge_lora_ties.py"
EVAL_SCRIPT = REPO_ROOT / "src" / "eval_retrieval.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run merged CIRR val eval for step checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--eval-gpu", type=int, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--geo-weight", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=0.9)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--min-step", type=int, default=200)
    parser.add_argument("--max-step", type=int, default=10**9)
    parser.add_argument("--base-kind", choices=["raw", "ema"], default="raw")
    parser.add_argument("--geo-kind", choices=["raw", "ema"], default="ema")
    parser.add_argument("--include-final", action="store_true", default=True)
    return parser.parse_args()


def parse_base_tag(path: Path, checkpoint_kind: str):
    name = path.name
    if checkpoint_kind == "ema":
        step_match = STEP_EMA_RE.search(name)
        final_match = FINAL_EMA_RE.search(name)
    else:
        step_match = STEP_RAW_RE.search(name)
        final_match = FINAL_RAW_RE.search(name)

    if step_match:
        epoch = int(step_match.group("epoch"))
        step = int(step_match.group("step"))
        return {
            "epoch": epoch,
            "step": step,
            "is_final": False,
            "sort_key": (epoch, step, 0),
            "tag_base": f"epoch{epoch}_step{step}",
        }
    if final_match:
        epoch = int(final_match.group("epoch"))
        return {
            "epoch": epoch,
            "step": None,
            "is_final": True,
            "sort_key": (epoch, 10**9, 1),
            "tag_base": f"epoch{epoch}_final",
        }
    return None


def geo_counterpart(base_path: Path, base_meta: dict, geo_kind: str):
    epoch = base_meta["epoch"]
    step = base_meta["step"]
    if step is None:
        suffix = "_ema" if geo_kind == "ema" else ""
        return base_path.parent / f"epoch_{epoch}_geo_lora{suffix}.pt"
    suffix = "_ema" if geo_kind == "ema" else ""
    return base_path.parent / f"epoch_{epoch}_step_{step}_geo_lora{suffix}.pt"


def build_candidates(checkpoint_dir: Path, base_kind: str, geo_kind: str, min_step: int, max_step: int):
    epoch_to_max_step = {}
    for path in checkpoint_dir.glob("*.pt"):
        if "_geo_lora" in path.name:
            continue
        meta = parse_base_tag(path, base_kind)
        if meta is None or meta["step"] is None:
            continue
        epoch_to_max_step[meta["epoch"]] = max(epoch_to_max_step.get(meta["epoch"], -1), meta["step"])

    candidates = []
    for path in checkpoint_dir.glob("*.pt"):
        if "_geo_lora" in path.name:
            continue
        meta = parse_base_tag(path, base_kind)
        if meta is None:
            continue
        if meta["is_final"] and epoch_to_max_step.get(meta["epoch"]) is not None:
            # Keep final too; it may differ from the largest stepped checkpoint.
            pass
        if meta["step"] is not None and not (min_step <= meta["step"] <= max_step):
            continue
        geo_path = geo_counterpart(path, meta, geo_kind)
        if not geo_path.exists():
            continue
        candidates.append((meta["sort_key"], path, meta, geo_path))
    candidates.sort()
    return candidates


def parse_metrics(output: str):
    metrics = {}
    current_feature = None
    for line in output.splitlines():
        feature_match = FEATURE_RE.search(line)
        if feature_match:
            current_feature = feature_match.group(1).lower()
            metrics.setdefault(current_feature, {})
        if current_feature is None:
            continue
        for metric_name, metric_val in METRIC_RE.findall(line):
            metrics[current_feature][metric_name] = float(metric_val)
    return metrics


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_subprocess(cmd, env=None):
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    return completed.returncode, completed.stdout


def main():
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("", encoding="utf-8")

    candidates = build_candidates(
        args.checkpoint_dir,
        args.base_kind,
        args.geo_kind,
        args.min_step,
        args.max_step,
    )

    for _, ckpt_path, meta, geo_path in candidates:
        with tempfile.NamedTemporaryFile(prefix=f"{meta['tag_base']}_cirr_", suffix=".pt", dir="/tmp", delete=False) as tmp_file:
            temp_merged = Path(tmp_file.name)

        merge_cmd = [
            sys.executable,
            str(MERGE_SCRIPT),
            "--ckpt-a", str(ckpt_path),
            "--ckpt-b", str(geo_path),
            "--output", str(temp_merged),
            "--weights", str(args.retrieval_weight), str(args.geo_weight),
            "--density", str(args.density),
            "--text-only",
            "--base", "a",
            "--alpha-a", str(args.lora_alpha),
            "--rank-a", str(args.lora_rank),
            "--alpha-b", str(args.lora_alpha),
            "--rank-b", str(args.lora_rank),
        ]
        merge_return, merge_output = run_subprocess(merge_cmd)
        if merge_return != 0:
            append_record(args.output_jsonl, {
                "tag": meta["tag_base"],
                "step": meta["step"],
                "status": "merge_failed",
                "merge_output": merge_output,
            })
            temp_merged.unlink(missing_ok=True)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
        eval_cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--resume", str(temp_merged),
            "--openai-pretrained",
            "--model", args.model,
            "--eval-mode", "cirr",
            "--gpu", "0",
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
        ]
        eval_return, eval_output = run_subprocess(eval_cmd, env=env)
        metrics = parse_metrics(eval_output)
        record = {
            "tag": meta["tag_base"],
            "step": meta["step"],
            "status": "ok" if eval_return == 0 and metrics else "eval_failed",
            "metrics": metrics,
        }
        if eval_return != 0 and not metrics:
            record["eval_output"] = eval_output
        append_record(args.output_jsonl, record)
        temp_merged.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
