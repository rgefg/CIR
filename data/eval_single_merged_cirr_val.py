#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = REPO_ROOT / "data" / "merge_lora_ties.py"
EVAL_SCRIPT = REPO_ROOT / "src" / "eval_retrieval.py"
FEATURE_RE = re.compile(r"Eval\s+(\w+)\s+Feature")
METRIC_RE = re.compile(r"([A-Za-z0-9_@]+):\s*([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge one retrieval/geo checkpoint pair and run CIRR val.")
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--eval-gpu", type=int, required=True)
    parser.add_argument("--model", type=str, default="ViT-L/14")
    parser.add_argument("--img2text-arch", type=str, default="phi")
    parser.add_argument("--middle-dim", type=int, default=3072)
    parser.add_argument("--img2text-pretrained", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retrieval-prompt-connector", type=str, default="and")
    parser.add_argument("--merge-weight-a", type=float, default=0.5)
    parser.add_argument("--merge-weight-b", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=0.9)
    parser.add_argument("--merge-mode", type=str, default="ties")
    parser.add_argument("--shared-b-num-layers", type=int, default=12)
    parser.add_argument("--svd-topk-rank", type=int, default=32)
    parser.add_argument("--svd-rescale", action="store_true", default=False)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    return parser.parse_args()


def run_subprocess(cmd, env=None):
    env = dict(os.environ if env is None else env)
    repo_paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(repo_paths + ([existing] if existing else []))
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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


def main():
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="cirr_single_merge_", suffix=".pt", dir="/tmp", delete=False) as tmp:
        merged_path = Path(tmp.name)

    try:
        merge_cmd = [
            sys.executable,
            str(MERGE_SCRIPT),
            "--ckpt-a", str(args.ckpt_a),
            "--ckpt-b", str(args.ckpt_b),
            "--output", str(merged_path),
            "--weights", str(args.merge_weight_a), str(args.merge_weight_b),
            "--density", str(args.density),
            "--merge-mode", str(args.merge_mode),
            "--shared-b-num-layers", str(args.shared_b_num_layers),
            "--svd-topk-rank", str(args.svd_topk_rank),
            *(["--svd-rescale"] if args.svd_rescale else []),
            "--text-only",
            "--base", "a",
            "--alpha-a", str(args.lora_alpha),
            "--rank-a", str(args.lora_rank),
            "--alpha-b", str(args.lora_alpha),
            "--rank-b", str(args.lora_rank),
        ]
        merge_rc, merge_out = run_subprocess(merge_cmd)
        if merge_rc != 0:
            payload = {
                "status": "merge_failed",
                "merge_output": merge_out,
            }
            args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
        eval_cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--resume", str(merged_path),
            "--openai-pretrained",
            "--model", args.model,
            "--img2text-arch", args.img2text_arch,
            "--middle_dim", str(args.middle_dim),
            "--eval-mode", "cirr",
            "--gpu", "0",
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--retrieval-prompt-connector", args.retrieval_prompt_connector,
        ]
        if args.img2text_pretrained:
            eval_cmd.extend(["--img2text-pretrained", args.img2text_pretrained])
        eval_rc, eval_out = run_subprocess(eval_cmd, env=env)
        metrics = parse_metrics(eval_out)
        payload = {
            "status": "ok" if eval_rc == 0 and metrics else "eval_failed",
            "merge": {
                "ckpt_a": str(args.ckpt_a),
                "ckpt_b": str(args.ckpt_b),
                "weights": [args.merge_weight_a, args.merge_weight_b],
                "density": args.density,
                "mode": args.merge_mode,
            },
            "metrics": metrics,
        }
        if eval_rc != 0 and not metrics:
            payload["eval_output"] = eval_out
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        merged_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
