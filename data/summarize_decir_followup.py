#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def cirr_metrics(result):
    metrics = ((result or {}).get("cirr") or {}).get("metrics") or {}
    composed = metrics.get("composed") or {}
    return {
        "FeatureR@1": composed.get("FeatureR@1"),
        "R_subset@1": composed.get("R_subset@1"),
        "R@5": composed.get("R@5"),
        "R@10": composed.get("R@10"),
    }


def suite_metrics(result):
    circo = ((result or {}).get("circo_val") or {}).get("map") or {}
    genecis = (result or {}).get("genecis") or {}
    r1s = [
        values.get("R@1")
        for values in genecis.values()
        if isinstance(values, dict) and values.get("R@1") is not None
    ]
    return {
        "CIRCO_mAP@50": circo.get("mAP@50"),
        "CIRCO_mAP@10": circo.get("mAP@10"),
        "GeneCIS_avg_R@1": (sum(r1s) / len(r1s)) if r1s else None,
    }


def read_status_rows(status_path):
    if not status_path.exists():
        return []
    with status_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def latest_rows_by_job(rows):
    latest = {}
    for row in rows:
        latest[row.get("job_id", "")] = row
    return latest


def result_for(records_dir, job_id):
    return load_json(records_dir / f"{job_id}_result.json")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def summarize_rank(records_dir):
    cirr_rows = []
    suite_rows = []
    for rank in [16, 32, 48, 64]:
        result = result_for(records_dir, f"rank_cirr_k{rank}")
        metrics = cirr_metrics(result)
        cirr_rows.append(
            {
                "rank": rank,
                "FeatureR@1": fmt(metrics["FeatureR@1"]),
                "R_subset@1": fmt(metrics["R_subset@1"]),
                "R@5": fmt(metrics["R@5"]),
                "R@10": fmt(metrics["R@10"]),
            }
        )
        result = result_for(records_dir, f"rank_suite_k{rank}")
        metrics = suite_metrics(result)
        suite_rows.append(
            {
                "rank": rank,
                "CIRCO mAP@50": fmt(metrics["CIRCO_mAP@50"]),
                "CIRCO mAP@10": fmt(metrics["CIRCO_mAP@10"]),
                "GeneCIS avg R@1": fmt(metrics["GeneCIS_avg_R@1"]),
            }
        )
    return cirr_rows, suite_rows


def summarize_loss(records_dir):
    rows = []
    jobs = [
        ("CIRR", "fwd_only", "loss_cirr_fwd_only_rerun", "cirr"),
        ("CIRR", "fwd_rev_nozero", "loss_cirr_fwd_rev_nozero", "cirr"),
        ("CIRCO+GeneCIS", "fwd_only", "loss_multi_fwd_only", "suite"),
        ("CIRCO+GeneCIS", "fwd_rev_nozero", "loss_multi_fwd_rev_nozero", "suite"),
    ]
    for dataset, variant, job_id, kind in jobs:
        result = result_for(records_dir, job_id)
        if kind == "cirr":
            metrics = cirr_metrics(result)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "primary": "CIRR R_subset@1",
                    "score": fmt(metrics["R_subset@1"]),
                    "secondary": f"FeatureR@1={fmt(metrics['FeatureR@1'])}",
                    "job_id": job_id,
                }
            )
        else:
            metrics = suite_metrics(result)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "primary": "CIRCO mAP@50 / GeneCIS avg R@1",
                    "score": f"{fmt(metrics['CIRCO_mAP@50'])} / {fmt(metrics['GeneCIS_avg_R@1'])}",
                    "secondary": f"CIRCO mAP@10={fmt(metrics['CIRCO_mAP@10'])}",
                    "job_id": job_id,
                }
            )
    return rows


def summarize_hard(result_dir):
    hard_dir = result_dir / "hard_analysis"
    rows = []
    for dataset in ["cirr", "circo", "genecis"]:
        cases = load_json(hard_dir / f"{dataset}_hard_cases.json")
        rows.append(
            {
                "dataset": dataset.upper(),
                "strict merged-win cases": len(cases) if isinstance(cases, list) else "",
                "file": str(hard_dir / f"{dataset}_hard_cases.json") if isinstance(cases, list) else "",
            }
        )
    return rows


def write_summary(result_dir, output_path):
    records_dir = result_dir / "records"
    rows = read_status_rows(result_dir / "status.tsv")
    latest = latest_rows_by_job(rows)
    rank_cirr, rank_suite = summarize_rank(records_dir)
    loss_rows = summarize_loss(records_dir)
    hard_rows = summarize_hard(result_dir)

    failed = [r for r in rows if r.get("status") == "failed"]
    current_loss_ids = {
        "loss_cirr_fwd_only_rerun",
        "loss_cirr_fwd_rev_nozero",
        "loss_multi_fwd_only",
        "loss_multi_fwd_rev_nozero",
    }
    current_loss_status = [latest.get(job_id, {}) for job_id in sorted(current_loss_ids)]
    hard_status = latest.get("hard_analysis_recovery") or latest.get("hard_analysis") or {}

    lines = [
        "# DeCIR Follow-up Results",
        "",
        f"- result_dir: `{result_dir}`",
        f"- status_file: `{result_dir / 'status.tsv'}`",
        f"- results_jsonl: `{result_dir / 'results.jsonl'}`",
        "",
        "## Rank Ablation",
        "",
        "CIRR validation, composed feature:",
        "",
        md_table(["rank", "FeatureR@1", "R_subset@1", "R@5", "R@10"], rank_cirr),
        "",
        "CIRCO validation and GeneCIS validation:",
        "",
        md_table(["rank", "CIRCO mAP@50", "CIRCO mAP@10", "GeneCIS avg R@1"], rank_suite),
        "",
        "## Transition-loss Ablation",
        "",
        md_table(["dataset", "variant", "primary", "score", "secondary", "job_id"], loss_rows),
        "",
        "## Hard Distractor Analysis",
        "",
        md_table(["dataset", "strict merged-win cases", "file"], hard_rows),
        "",
        f"- hard_analysis_status: `{hard_status.get('status', 'missing')}`",
        f"- hard_analysis_md: `{result_dir / 'hard_analysis' / 'HARD_ANALYSIS.md'}`",
        "",
        "## Run Status",
        "",
        md_table(
            ["job_id", "status", "started_at", "finished_at"],
            [
                {
                    "job_id": r.get("job_id", ""),
                    "status": r.get("status", ""),
                    "started_at": r.get("started_at", ""),
                    "finished_at": r.get("finished_at", ""),
                }
                for r in current_loss_status
            ],
        ),
        "",
        f"- failed_status_rows_total: {len(failed)}",
        "- note: earlier repeated `hard_analysis` failures came from the killed accelerator loop; the recovery result is `hard_analysis_recovery`.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize DeCIR follow-up experiment records.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (args.result_dir / "FOLLOWUP_RESULTS_SUMMARY.md")
    write_summary(args.result_dir, output)


if __name__ == "__main__":
    main()
