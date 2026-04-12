#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_RETRIEVAL = REPO_ROOT / "src" / "eval_retrieval.py"
EVAL_SUITE = REPO_ROOT / "data" / "eval_multidataset_suite.py"
FEATURE_RE = re.compile(r"Eval\s+(\w+)\s+Feature")
METRIC_RE = re.compile(r"([A-Za-z0-9_@]+):\s*([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval-only or geo-only single checkpoints across CIRR/CIRCO/GeneCIS.")
    parser.add_argument("--resume", type=Path, required=True, help="Full retrieval checkpoint.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--variant", choices=["retrieval_only", "geo_only"], required=True)
    parser.add_argument("--geo-lora", type=Path, default=None, help="Required for geo_only.")
    parser.add_argument("--cirr-gpu", type=int, default=0)
    parser.add_argument("--suite-gpu", type=int, default=1)
    parser.add_argument("--model", type=str, default="ViT-L/14")
    parser.add_argument("--img2text-arch", type=str, default="phi")
    parser.add_argument("--middle-dim", type=int, default=3072)
    parser.add_argument("--img2text-pretrained", type=str, default=None)
    parser.add_argument("--cirr-batch-size", type=int, default=48)
    parser.add_argument("--suite-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--genecis-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-prompt-connector", choices=["and", "that"], default="that")
    parser.add_argument("--datasets", type=str, default="cirr,circo,genecis")
    parser.add_argument("--name", type=str, default="branch_ablation_eval")
    return parser.parse_args()


def _strip_module_prefix(name: str) -> str:
    return name[len("module."):] if name.startswith("module.") else name


def _is_text_lora_key(name: str) -> bool:
    core = _strip_module_prefix(name)
    return (core.endswith(".A") or core.endswith(".B")) and not core.startswith("visual.")


def _safe_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def build_geo_only_checkpoint(full_ckpt_path: Path, geo_lora_path: Path) -> Path:
    ckpt = _safe_load(full_ckpt_path)
    geo_lora = _safe_load(geo_lora_path)
    if not isinstance(geo_lora, dict):
        raise ValueError(f"Unsupported geo LoRA checkpoint format: {geo_lora_path}")
    state_dict = ckpt["state_dict"]
    norm_to_full = {
        _strip_module_prefix(k): k
        for k in state_dict.keys()
        if _is_text_lora_key(k)
    }
    replaced = 0
    missing = []
    for key, value in geo_lora.items():
        if not _is_text_lora_key(key):
            continue
        actual_key = norm_to_full.get(_strip_module_prefix(key))
        if actual_key is None:
            missing.append(key)
            continue
        state_dict[actual_key] = value
        replaced += 1
    if replaced == 0:
        raise RuntimeError(f"No text LoRA tensors were replaced from {geo_lora_path}")
    if missing:
        print(f"[geo-only] warning: {len(missing)} geo keys were not matched to retrieval checkpoint", flush=True)
    ckpt["state_dict"] = state_dict
    ckpt["branch_variant"] = "geo_only"
    with tempfile.NamedTemporaryFile(prefix="geo_only_", suffix=".pt", dir="/tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
    torch.save(ckpt, temp_path)
    return temp_path


def parse_cirr_metrics(output: str):
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


def main():
    args = parse_args()
    selected = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    eval_resume = args.resume
    temp_resume = None
    try:
        if args.variant == "geo_only":
            if args.geo_lora is None:
                raise ValueError("--geo-lora is required for geo_only")
            temp_resume = build_geo_only_checkpoint(args.resume, args.geo_lora)
            eval_resume = temp_resume

        result = {
            "variant": args.variant,
            "resume": str(args.resume),
            "eval_resume": str(eval_resume),
        }

        if "cirr" in selected:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.cirr_gpu)
            cirr_cmd = [
                sys.executable,
                str(EVAL_RETRIEVAL),
                "--resume", str(eval_resume),
                "--openai-pretrained",
                "--model", args.model,
                "--img2text-arch", args.img2text_arch,
                "--middle_dim", str(args.middle_dim),
                "--eval-mode", "cirr",
                "--gpu", "0",
                "--batch-size", str(args.cirr_batch_size),
                "--workers", str(args.workers),
            ]
            if args.img2text_pretrained:
                cirr_cmd.extend(["--img2text-pretrained", args.img2text_pretrained])
            rc, out = run_subprocess(cirr_cmd, env=env)
            cirr_metrics = parse_cirr_metrics(out)
            result["cirr"] = {
                "status": "ok" if rc == 0 and cirr_metrics else "failed",
                "metrics": cirr_metrics,
            }
            if rc != 0 and not cirr_metrics:
                result["cirr"]["output"] = out

        suite_selected = [x for x in ("circo", "genecis") if x in selected]
        if suite_selected:
            with tempfile.NamedTemporaryFile(prefix="suite_eval_", suffix=".json", dir="/tmp", delete=False) as tmp:
                suite_out_path = Path(tmp.name)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.suite_gpu)
            suite_cmd = [
                sys.executable,
                str(EVAL_SUITE),
                "--resume", str(eval_resume),
                "--output-json", str(suite_out_path),
                "--gpu", "0",
                "--batch-size", str(args.suite_batch_size),
                "--workers", str(args.workers),
                "--genecis-batch-size", str(args.genecis_batch_size),
                "--model", args.model,
                "--img2text-arch", args.img2text_arch,
                "--middle-dim", str(args.middle_dim),
                "--retrieval-prompt-connector", args.retrieval_prompt_connector,
                "--datasets", ",".join(suite_selected),
                "--name", args.name,
            ]
            if args.img2text_pretrained:
                suite_cmd.extend(["--img2text-pretrained", args.img2text_pretrained])
            rc, out = run_subprocess(suite_cmd, env=env)
            if suite_out_path.exists():
                result.update(json.loads(suite_out_path.read_text(encoding="utf-8")))
                suite_out_path.unlink(missing_ok=True)
            else:
                result["suite_status"] = "failed"
                result["suite_output"] = out
            if rc != 0 and "suite_output" not in result:
                result["suite_output"] = out

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if temp_resume is not None:
            temp_resume.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
