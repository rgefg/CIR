#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


STEP_RAW_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$")
STEP_EMA_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)_ema\.pt$")
FINAL_RAW_RE = re.compile(r"epoch_(?P<epoch>\d+)\.pt$")
FINAL_EMA_RE = re.compile(r"epoch_(?P<epoch>\d+)_ema\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch checkpoints and run periodic FashionIQ + GeneCIS evaluation."
    )
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
    parser.add_argument("--checkpoint-kind", choices=["raw", "ema"], default="ema")
    parser.add_argument("--stop-on-final", action="store_true", default=False)
    return parser.parse_args()


def parse_tag(path: Path, checkpoint_kind: str):
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
            "tag": f"epoch{epoch}_step{step}_{checkpoint_kind}",
            "sort_key": (epoch, step, 0),
            "is_final": False,
            "checkpoint_kind": checkpoint_kind,
        }
    if final_match:
        epoch = int(final_match.group("epoch"))
        return {
            "epoch": epoch,
            "step": None,
            "tag": f"epoch{epoch}_final_{checkpoint_kind}",
            "sort_key": (epoch, 10**9, 1),
            "is_final": True,
            "checkpoint_kind": checkpoint_kind,
        }
    return None


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


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def main():
    args = parse_args()
    if args.cpu_affinity:
        os.sched_setaffinity(0, parse_cpu_affinity(args.cpu_affinity))
    if args.nice > 0:
        os.nice(args.nice)

    limited_env = build_limited_env(os.environ, args.cpu_threads)
    processed = load_processed(args.output_jsonl)

    while True:
        candidates = []
        for path in args.checkpoint_dir.glob("*.pt"):
            if "_geo_lora" in path.name:
                continue
            tag = parse_tag(path, args.checkpoint_kind)
            if tag is None or tag["tag"] in processed:
                continue
            candidates.append((tag["sort_key"], path, tag))
        candidates.sort()

        for _, ckpt_path, tag in candidates:
            with tempfile.NamedTemporaryFile(
                prefix=f"{tag['tag']}_multieval_",
                suffix=".json",
                dir="/tmp",
                delete=False,
            ) as tmp_file:
                tmp_output = Path(tmp_file.name)

            eval_env = dict(limited_env)
            eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
            eval_cmd = [
                "python",
                "data/eval_multidataset_suite.py",
                "--resume",
                str(ckpt_path),
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
                "fashioniq,genecis",
                "--name",
                f"multidataset_eval_{tag['tag']}",
            ]
            eval_code, eval_output = run_command(eval_cmd, timeout=args.timeout, env=eval_env)
            record = {
                **tag,
                "checkpoint": str(ckpt_path),
                "created_at": time.time(),
                "status": "ok" if eval_code == 0 else "eval_failed",
                "eval_output_tail": eval_output[-4000:],
            }
            if eval_code == 0 and tmp_output.exists():
                with open(tmp_output, "r", encoding="utf-8") as f:
                    record["metrics"] = json.load(f)
            append_record(args.output_jsonl, record)
            processed.add(tag["tag"])
            tmp_output.unlink(missing_ok=True)

            if tag["is_final"] and args.stop_on_final:
                return

        time.sleep(max(1, args.poll_interval))


if __name__ == "__main__":
    main()
