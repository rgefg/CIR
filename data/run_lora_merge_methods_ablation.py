#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_TIES = REPO_ROOT / "data" / "merge_lora_ties.py"
MERGE_ROBUST = REPO_ROOT / "data" / "merge_lora_robust.py"
EVAL_RETRIEVAL = REPO_ROOT / "src" / "eval_retrieval.py"
EVAL_MULTI = REPO_ROOT / "data" / "eval_multidataset_suite.py"

DEFAULT_CIRR_DIR = REPO_ROOT / "logs" / "DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp" / "checkpoints"
DEFAULT_MULTI_DIR = REPO_ROOT / "logs" / "DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp" / "checkpoints"
SEARLE_VITL = "/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt"

FEATURE_RE = re.compile(r"Eval\s+(\w+)\s+Feature")
METRIC_RE = re.compile(r"([A-Za-z0-9_@]+):\s*([0-9.]+)")


METHODS = {
    "task_arithmetic": {"script": "ties", "mode": "task_arithmetic"},
    "magnitude_prune": {"script": "ties", "mode": "magnitude_prune"},
    "ties_frequency": {"script": "ties", "mode": "ties", "majority": "frequency"},
    "dare_linear": {"script": "ties", "mode": "dare_linear"},
    "dare_ties": {"script": "ties", "mode": "dare_ties"},
    "breadcrumbs": {"script": "ties", "mode": "breadcrumbs"},
    "robust": {"script": "robust", "mode": None},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoRA merge-method ablations on the fixed Shared-B step1400 setup.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", type=str, default="task_arithmetic,magnitude_prune,ties_frequency,dare_linear,dare_ties,breadcrumbs,robust")
    parser.add_argument("--gpu-list", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--idle-mem-mb", type=int, default=200)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--step", type=int, default=1400)
    parser.add_argument("--eval-scope", type=str, default="all", choices=["all", "cirr", "multi"])
    parser.add_argument("--density", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--multi-batch-size", type=int, default=56)
    parser.add_argument("--genecis-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--merge-on-gpu",
        action="store_true",
        default=False,
        help="Run dense LoRA merge/SVD on the selected physical GPU instead of CPU.",
    )
    parser.add_argument("--cirr-ckpt-dir", type=Path, default=DEFAULT_CIRR_DIR)
    parser.add_argument("--multi-ckpt-dir", type=Path, default=DEFAULT_MULTI_DIR)
    return parser.parse_args()


def run(cmd, *, log_path=None):
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if log_path is not None:
        Path(log_path).write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, completed.stdout


def query_gpu_memory():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    out = {}
    for line in completed.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            out[int(parts[0])] = {"mem": int(parts[1]), "util": int(parts[2])}
    return out


def wait_for_gpu(gpu_list, idle_mem_mb, poll_seconds, status_path):
    while True:
        mem = query_gpu_memory()
        for gpu in gpu_list:
            if gpu in mem and mem[gpu]["mem"] <= idle_mem_mb:
                status_path.write_text(
                    f"{datetime.now().isoformat()} selected_gpu={gpu} mem={mem[gpu]['mem']} util={mem[gpu]['util']}\n",
                    encoding="utf-8",
                )
                return gpu
        status_path.write_text(
            f"{datetime.now().isoformat()} waiting; "
            + " ".join([f"gpu{g}:mem={mem.get(g, {}).get('mem', 'na')}" for g in gpu_list])
            + "\n",
            encoding="utf-8",
        )
        time.sleep(poll_seconds)


def parse_cirr_metrics(output):
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


def avg_genecis_r1(genecis):
    vals = []
    for task in ["focus_attribute", "change_attribute", "focus_object", "change_object"]:
        if task in genecis and "R@1" in genecis[task]:
            vals.append(float(genecis[task]["R@1"]))
    return sum(vals) / max(len(vals), 1)


def checkpoint_pair(ckpt_dir, step):
    return ckpt_dir / f"epoch_0_step_{step}.pt", ckpt_dir / f"epoch_0_step_{step}_geo_lora_ema.pt"


def merge_checkpoint(method_name, method_cfg, ckpt_a, ckpt_b, output, density, seed, svd_device, log_path):
    if method_cfg["script"] == "robust":
        cmd = [
            sys.executable,
            str(MERGE_ROBUST),
            "--ckpt-a", str(ckpt_a),
            "--ckpt-b", str(ckpt_b),
            "--output", str(output),
            "--weights", "0.5", "0.5",
            "--density", str(density),
            "--text-only",
            "--base", "a",
            "--alpha-a", "16",
            "--rank-a", "64",
            "--alpha-b", "16",
            "--rank-b", "64",
            "--svd-device", svd_device,
        ]
    else:
        cmd = [
            sys.executable,
            str(MERGE_TIES),
            "--ckpt-a", str(ckpt_a),
            "--ckpt-b", str(ckpt_b),
            "--output", str(output),
            "--weights", "0.5", "0.5",
            "--density", str(density),
            "--merge-mode", method_cfg["mode"],
            "--majority-sign-method", method_cfg.get("majority", "total"),
            "--shared-b-num-layers", "12",
            "--svd-topk-rank", "32",
            "--text-only",
            "--base", "a",
            "--alpha-a", "16",
            "--rank-a", "64",
            "--alpha-b", "16",
            "--rank-b", "64",
            "--seed", str(seed),
            "--svd-device", svd_device,
        ]
    rc, stdout = run(cmd, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f"merge failed for {method_name}\n{stdout}")


def eval_cirr(merged_ckpt, gpu, output_json, log_path, batch_size, workers):
    cmd = [
        sys.executable,
        str(EVAL_RETRIEVAL),
        "--resume", str(merged_ckpt),
        "--openai-pretrained",
        "--model", "ViT-L/14",
        "--img2text-arch", "phi",
        "--middle_dim", "3072",
        "--img2text-pretrained", SEARLE_VITL,
        "--eval-mode", "cirr",
        "--gpu", str(gpu),
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--retrieval-prompt-connector", "and",
    ]
    rc, stdout = run(cmd, log_path=log_path)
    metrics = parse_cirr_metrics(stdout)
    payload = {"status": "ok" if rc == 0 and metrics else "eval_failed", "metrics": metrics}
    if payload["status"] != "ok":
        payload["returncode"] = rc
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def eval_multi(merged_ckpt, gpu, output_json, log_path, batch_size, genecis_batch_size, workers):
    cmd = [
        sys.executable,
        str(EVAL_MULTI),
        "--resume", str(merged_ckpt),
        "--output-json", str(output_json),
        "--gpu", str(gpu),
        "--batch-size", str(batch_size),
        "--genecis-batch-size", str(genecis_batch_size),
        "--workers", str(workers),
        "--model", "ViT-L/14",
        "--img2text-arch", "phi",
        "--middle-dim", "3072",
        "--img2text-pretrained", SEARLE_VITL,
        "--retrieval-prompt-connector", "that",
        "--datasets", "circo,genecis",
    ]
    rc, stdout = run(cmd, log_path=log_path)
    if rc != 0 or not output_json.exists():
        return {"status": "eval_failed", "returncode": rc}
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload["status"] = "ok"
    return payload


def summarize(results):
    lines = [
        "# LoRA Merge Method Ablation",
        "",
        "Fixed setup: ViT-L/14 Shared-B12 step1400. CIRR uses the no-drop CIRR run; CIRCO/GeneCIS use the dropout=0.5 CIRCO/GeneCIS run. All merges are text-only, retrieval:geo = 0.5:0.5, density = 0.9.",
        "",
        "| Method | CIRR R_subset@1 | CIRCO mAP@50 | GeneCIS avg R@1 | Status |",
        "|---|---:|---:|---:|---|",
        "| TIES reference | 63.29 | 26.70 | 16.59 | existing baseline |",
    ]
    for item in results:
        status = item.get("status", "unknown")
        cirr = item.get("cirr", {}).get("metrics", {}).get("composed", {})
        multi = item.get("multi", {})
        circo = multi.get("circo_val", {}).get("map", {})
        genecis = multi.get("genecis", {})
        rs1 = cirr.get("R_subset@1")
        map50 = circo.get("mAP@50")
        avg = avg_genecis_r1(genecis) if genecis else None
        map50_pct = None if map50 is None else float(map50) * 100.0
        lines.append(
            f"| {item['method']} | "
            f"{rs1:.2f} | " if rs1 is not None else f"| {item['method']} | n/a | "
        )
        row = lines.pop()
        row += f"{map50_pct:.2f} | " if map50_pct is not None else "n/a | "
        row += f"{avg:.2f} | " if avg is not None else "n/a | "
        row += f"{status} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / "logs" / f"lora_merge_methods_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    gpu_list = [int(x) for x in args.gpu_list.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = [x for x in methods if x not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")

    cirr_a, cirr_b = checkpoint_pair(args.cirr_ckpt_dir, args.step)
    multi_a, multi_b = checkpoint_pair(args.multi_ckpt_dir, args.step)
    for path in [cirr_a, cirr_b, multi_a, multi_b]:
        if not path.exists():
            raise FileNotFoundError(path)

    results = []
    result_jsonl = output_dir / "results.jsonl"
    with result_jsonl.open("w", encoding="utf-8") as result_f:
        for method in methods:
            cfg = METHODS[method]
            record = {"method": method, "started_at": datetime.now().isoformat(), "status": "running"}
            try:
                if args.eval_scope in {"all", "cirr"}:
                    gpu = wait_for_gpu(gpu_list, args.idle_mem_mb, args.poll_seconds, output_dir / "gpu_wait_status.txt")
                    svd_device = f"cuda:{gpu}" if args.merge_on_gpu else "cpu"
                    cirr_merged = tmp_dir / f"{method}_cirr.pt"
                    merge_checkpoint(
                        method,
                        cfg,
                        cirr_a,
                        cirr_b,
                        cirr_merged,
                        args.density,
                        args.seed,
                        svd_device,
                        output_dir / "logs" / f"{method}_cirr_merge.log",
                    )
                    gpu = wait_for_gpu(gpu_list, args.idle_mem_mb, args.poll_seconds, output_dir / "gpu_wait_status.txt")
                    record["cirr"] = eval_cirr(
                        cirr_merged,
                        gpu,
                        output_dir / "records" / f"{method}_cirr_result.json",
                        output_dir / "logs" / f"{method}_cirr_eval.log",
                        args.batch_size,
                        args.workers,
                    )
                    cirr_merged.unlink(missing_ok=True)

                if args.eval_scope in {"all", "multi"}:
                    gpu = wait_for_gpu(gpu_list, args.idle_mem_mb, args.poll_seconds, output_dir / "gpu_wait_status.txt")
                    svd_device = f"cuda:{gpu}" if args.merge_on_gpu else "cpu"
                    multi_merged = tmp_dir / f"{method}_multi.pt"
                    merge_checkpoint(
                        method,
                        cfg,
                        multi_a,
                        multi_b,
                        multi_merged,
                        args.density,
                        args.seed,
                        svd_device,
                        output_dir / "logs" / f"{method}_multi_merge.log",
                    )
                    gpu = wait_for_gpu(gpu_list, args.idle_mem_mb, args.poll_seconds, output_dir / "gpu_wait_status.txt")
                    record["multi"] = eval_multi(
                        multi_merged,
                        gpu,
                        output_dir / "records" / f"{method}_multi_result.json",
                        output_dir / "logs" / f"{method}_multi_eval.log",
                        args.multi_batch_size,
                        args.genecis_batch_size,
                        args.workers,
                    )
                    multi_merged.unlink(missing_ok=True)
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
            record["finished_at"] = datetime.now().isoformat()
            results.append(record)
            result_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_f.flush()
            (output_dir / "SUMMARY.md").write_text(summarize(results), encoding="utf-8")
            print(json.dumps(record, ensure_ascii=False), flush=True)

    (output_dir / "SUMMARY.md").write_text(summarize(results), encoding="utf-8")
    print(f"summary: {output_dir / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
