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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGE_SCRIPT = REPO_ROOT / "data" / "merge_lora_ties.py"
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "data" / "eval_multidataset_suite.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch checkpoints and run compact standalone or merged FashionIQ + GeneCIS evaluation."
    )
    parser.add_argument("--mode", choices=["standalone", "merged"], default="standalone")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--eval-gpu", type=int, required=True)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--genecis-batch-size", type=int, default=32)
    parser.add_argument("--nice", type=int, default=0)
    parser.add_argument("--cpu-affinity", type=str, default=None)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--datasets", type=str, default="fashioniq,genecis")
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--stop-on-final", action="store_true", default=False)
    parser.add_argument("--once", action="store_true", default=False)

    parser.add_argument("--checkpoint-kind", choices=["raw", "ema"], default="raw")

    parser.add_argument("--base-kind", choices=["raw", "ema"], default="raw")
    parser.add_argument("--geo-kind", choices=["raw", "ema"], default="ema")
    parser.add_argument("--merge-script", type=Path, default=DEFAULT_MERGE_SCRIPT)
    parser.add_argument("--merge-mode", choices=["ties", "shared_a_sum_b"], default="ties")
    parser.add_argument("--merge-weight-a", type=float, default=0.5)
    parser.add_argument("--merge-weight-b", type=float, default=0.5)
    parser.add_argument("--merge-density", type=float, default=0.9)
    parser.add_argument("--merge-alpha-a", type=float, default=16.0)
    parser.add_argument("--merge-rank-a", type=int, default=64)
    parser.add_argument("--merge-alpha-b", type=float, default=16.0)
    parser.add_argument("--merge-rank-b", type=int, default=64)
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


def build_eval_record(tag: str, step, mode: str, metrics: dict, extra: dict | None = None):
    record = {
        "tag": tag,
        "step": step,
        "mode": mode,
    }
    if extra:
        record.update(extra)
    record.update(metrics)
    return record


def maybe_compact_metrics(metrics: dict):
    out = {}
    if "fashioniq" in metrics:
        out["fashioniq"] = metrics["fashioniq"]
    if "genecis" in metrics:
        out["genecis"] = metrics["genecis"]
    if "circo_val" in metrics:
        out["circo_val"] = metrics["circo_val"]
    return out


def evaluate_checkpoint(eval_script: Path, resume_path: Path, args, limited_env, eval_name: str):
    with tempfile.NamedTemporaryFile(prefix=f"{eval_name}_", suffix=".json", dir="/tmp", delete=False) as tmp_file:
        tmp_output = Path(tmp_file.name)
    eval_env = dict(limited_env)
    eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--resume",
        str(resume_path),
        "--output-json",
        str(tmp_output),
        "--gpu",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--genecis-batch-size",
        str(args.genecis_batch_size),
        "--datasets",
        args.datasets,
        "--name",
        eval_name,
    ]
    code, output = run_command(eval_cmd, timeout=args.timeout, env=eval_env)
    metrics = None
    if code == 0 and tmp_output.exists():
        with open(tmp_output, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    tmp_output.unlink(missing_ok=True)
    return code, output, metrics


def main():
    args = parse_args()
    if args.cpu_affinity:
        os.sched_setaffinity(0, parse_cpu_affinity(args.cpu_affinity))
    if args.nice > 0:
        os.nice(args.nice)

    limited_env = build_limited_env(os.environ, args.cpu_threads)
    processed = load_processed(args.output_jsonl)

    while True:
        if args.mode == "standalone":
            epoch_to_max_step = build_epoch_max_step_map(args.checkpoint_dir, args.checkpoint_kind)
        else:
            epoch_to_max_step = build_epoch_max_step_map(args.checkpoint_dir, args.base_kind)
        candidates = []
        for path in args.checkpoint_dir.glob("*.pt"):
            if args.mode == "standalone":
                if "_geo_lora" in path.name:
                    continue
                meta = parse_base_tag(path, args.checkpoint_kind)
                if meta is None:
                    continue
                if meta["is_final"] and epoch_to_max_step.get(meta["epoch"]) is not None:
                    continue
                if meta["step"] is not None and meta["step"] < args.min_step:
                    continue
                tag = f"{meta['tag_base']}_{args.checkpoint_kind}_standalone"
                if tag in processed:
                    continue
                candidates.append((meta["sort_key"], path, meta, tag, None))
            else:
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
            eval_target = ckpt_path
            temp_merged = None
            if args.mode == "merged":
                with tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".pt", dir="/tmp", delete=False) as tmp_file:
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
                    str(args.merge_weight_a),
                    str(args.merge_weight_b),
                    "--merge-mode",
                    str(args.merge_mode),
                    "--density",
                    str(args.merge_density),
                    "--text-only",
                    "--base",
                    "a",
                    "--include-b-only",
                    "--alpha-a",
                    str(args.merge_alpha_a),
                    "--rank-a",
                    str(args.merge_rank_a),
                    "--alpha-b",
                    str(args.merge_alpha_b),
                    "--rank-b",
                    str(args.merge_rank_b),
                    "--slim-output",
                ]
                merge_code, merge_output = run_command(merge_cmd, timeout=args.timeout, env=limited_env)
                if merge_code != 0:
                    print(f"[watch_multidataset_eval] merge failed for {tag}\n{merge_output[-4000:]}", flush=True)
                    processed.add(tag)
                    temp_merged.unlink(missing_ok=True)
                    if meta["is_final"] and args.stop_on_final:
                        raise SystemExit(1)
                    continue
                eval_target = temp_merged

            eval_code, eval_output, metrics = evaluate_checkpoint(
                DEFAULT_EVAL_SCRIPT,
                eval_target,
                args,
                limited_env,
                f"multidataset_{tag}",
            )
            if eval_code != 0 or metrics is None:
                print(f"[watch_multidataset_eval] eval failed for {tag}\n{eval_output[-4000:]}", flush=True)
                processed.add(tag)
                if temp_merged is not None:
                    temp_merged.unlink(missing_ok=True)
                if meta["is_final"] and args.stop_on_final:
                    raise SystemExit(1)
                continue

            compact_metrics = maybe_compact_metrics(metrics)
            if args.mode == "standalone":
                extra = {"checkpoint_kind": args.checkpoint_kind}
            else:
                extra = {
                    "base_kind": args.base_kind,
                    "geo_kind": args.geo_kind,
                }
            record = build_eval_record(tag, meta["step"], args.mode, compact_metrics, extra)
            append_record(args.output_jsonl, record)
            processed.add(tag)
            if temp_merged is not None:
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
