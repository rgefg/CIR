#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


STEP_RAW_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$")
STEP_EMA_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)_ema\.pt$")
FINAL_RAW_RE = re.compile(r"epoch_(?P<epoch>\d+)\.pt$")
FINAL_EMA_RE = re.compile(r"epoch_(?P<epoch>\d+)_ema\.pt$")
METRIC_RE = re.compile(r"(recall_[^:]+):\s*([0-9.]+)")
FEATURE_RE = re.compile(r"Eval\s+(\w+)\s+Feature")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGE_SCRIPT = REPO_ROOT / "data" / "merge_lora_ties.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Watch retrieval/geo checkpoints, merge them, and run CIRR val eval.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--eval-gpu", type=int, required=True)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--geo-weight", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=0.9)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--nice", type=int, default=0)
    parser.add_argument("--cpu-affinity", type=str, default=None)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--base-kind", choices=["raw", "ema"], default="raw")
    parser.add_argument("--geo-kind", choices=["raw", "ema"], default="ema")
    parser.add_argument("--merge-script", type=Path, default=DEFAULT_MERGE_SCRIPT)
    parser.add_argument("--merge-mode", choices=["ties", "shared_b_sum_a", "hybrid_layerwise", "shared_a_sum_b"], default="ties")
    parser.add_argument("--shared-b-num-layers", dest="shared_b_num_layers", type=int, default=6)
    parser.add_argument("--shared-a-num-layers", dest="shared_b_num_layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stop-on-final", action="store_true", default=False)
    parser.add_argument("--once", action="store_true", default=False)
    return parser.parse_args()


def parse_cpu_affinity(spec: str):
    cpus = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(chunk))
    if not cpus:
        raise ValueError(f"Invalid cpu affinity spec: {spec!r}")
    return cpus


def build_limited_env(base_env, cpu_threads: int):
    env = dict(base_env)
    thread_str = str(max(1, int(cpu_threads)))
    env["OMP_NUM_THREADS"] = thread_str
    env["MKL_NUM_THREADS"] = thread_str
    env["OPENBLAS_NUM_THREADS"] = thread_str
    env["NUMEXPR_NUM_THREADS"] = thread_str
    env["VECLIB_MAXIMUM_THREADS"] = thread_str
    env["BLIS_NUM_THREADS"] = thread_str
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_command(cmd, timeout, env=None):
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    return completed.returncode, completed.stdout


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


def load_processed(path: Path):
    processed = set()
    if not path.exists():
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = record.get("tag")
            if tag:
                processed.add(tag)
    return processed


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def build_epoch_max_step_map(checkpoint_dir: Path, kind: str):
    epoch_to_max_step = {}
    for path in checkpoint_dir.glob("*.pt"):
        if "_geo_lora" in path.name:
            continue
        meta = parse_base_tag(path, kind)
        if meta is None or meta["step"] is None:
            continue
        epoch = meta["epoch"]
        step = meta["step"]
        prev = epoch_to_max_step.get(epoch)
        if prev is None or step > prev:
            epoch_to_max_step[epoch] = step
    return epoch_to_max_step


def main():
    args = parse_args()
    if args.cpu_affinity:
        os.sched_setaffinity(0, parse_cpu_affinity(args.cpu_affinity))
    if args.nice > 0:
        os.nice(args.nice)

    limited_env = build_limited_env(os.environ, args.cpu_threads)
    processed = load_processed(args.output_jsonl)

    while True:
        epoch_to_max_step = build_epoch_max_step_map(args.checkpoint_dir, args.base_kind)
        candidates = []
        for path in args.checkpoint_dir.glob("*.pt"):
            if "_geo_lora" in path.name:
                continue
            meta = parse_base_tag(path, args.base_kind)
            if meta is None:
                continue
            if meta["is_final"] and epoch_to_max_step.get(meta["epoch"]) is not None:
                continue
            if meta["step"] is not None and meta["step"] < args.min_step:
                continue
            geo_path = geo_counterpart(path, meta, args.geo_kind)
            if not geo_path.exists():
                continue
            tag = f"{meta['tag_base']}_{args.base_kind}_plus_geo{args.geo_kind}_merged"
            if tag in processed:
                continue
            candidates.append((meta["sort_key"], path, meta, tag, geo_path))

        candidates.sort()
        processed_any = False
        for _, ckpt_path, meta, tag, geo_path in candidates:
            processed_any = True
            with tempfile.NamedTemporaryFile(prefix=f"{tag}_cirr_", suffix=".pt", dir="/tmp", delete=False) as tmp_file:
                temp_merged = Path(tmp_file.name)

            merge_cmd = [
                sys.executable,
                str(args.merge_script),
                "--ckpt-a",
                str(ckpt_path),
                "--ckpt-b",
                str(geo_path),
                "--output",
                str(temp_merged),
                "--weights",
                str(args.retrieval_weight),
                str(args.geo_weight),
                "--merge-mode",
                str(args.merge_mode),
                "--shared-b-num-layers",
                str(args.shared_b_num_layers),
                "--density",
                str(args.density),
                "--text-only",
                "--base",
                "a",
                "--include-b-only",
                "--alpha-a",
                str(args.lora_alpha),
                "--rank-a",
                str(args.lora_rank),
                "--alpha-b",
                str(args.lora_alpha),
                "--rank-b",
                str(args.lora_rank),
                "--slim-output",
            ]
            merge_code, merge_output = run_command(merge_cmd, timeout=args.timeout, env=limited_env)
            if merge_code != 0:
                append_record(
                    args.output_jsonl,
                    {
                        "tag": tag,
                        "step": meta["step"],
                        "status": "merge_failed",
                        "base_kind": args.base_kind,
                        "geo_kind": args.geo_kind,
                        "merge_output": merge_output[-4000:],
                        "created_at": time.time(),
                    },
                )
                processed.add(tag)
                temp_merged.unlink(missing_ok=True)
                if meta["is_final"] and args.stop_on_final:
                    return
                continue

            eval_env = dict(limited_env)
            eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
            eval_cmd = [
                sys.executable,
                str(REPO_ROOT / "src" / "eval_retrieval.py"),
                "--resume",
                str(temp_merged),
                "--openai-pretrained",
                "--model",
                "ViT-L/14",
                "--eval-mode",
                "cirr",
                "--gpu",
                "0",
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ]
            eval_code, eval_output = run_command(eval_cmd, timeout=args.timeout, env=eval_env)
            metrics = parse_metrics(eval_output)
            append_record(
                args.output_jsonl,
                {
                    "tag": tag,
                    "step": meta["step"],
                    "status": "ok" if eval_code == 0 else "eval_failed",
                    "base_kind": args.base_kind,
                    "geo_kind": args.geo_kind,
                    "metrics": metrics,
                    "merge_summary": merge_output[-2000:],
                    "eval_output": eval_output[-4000:] if eval_code != 0 else "",
                    "created_at": time.time(),
                },
            )
            processed.add(tag)
            temp_merged.unlink(missing_ok=True)
            if meta["is_final"] and args.stop_on_final:
                return

        if args.once:
            return
        if not processed_any and args.stop_on_final:
            time.sleep(max(1, args.poll_interval))
            continue
        time.sleep(max(1, args.poll_interval))


if __name__ == "__main__":
    main()
