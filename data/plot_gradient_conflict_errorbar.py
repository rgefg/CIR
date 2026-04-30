#!/usr/bin/env python3
"""Aggregate offline gradient-conflict probe runs and plot error bars."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("gc", "gc_after", "gc_bidir")
LABELS = {
    "gc": "GC before projection",
    "gc_after": "GC after (one-way PCGrad)",
    "gc_bidir": "GC after (bidirectional PCGrad)",
}
STYLES = {
    "gc": {"marker": "o", "linestyle": "-"},
    "gc_after": {"marker": "s", "linestyle": "--"},
    "gc_bidir": {"marker": "^", "linestyle": ":"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="Run spec as seed:path_to_probe_output_dir. Can be repeated.")
    parser.add_argument("--error", choices=("sem", "std"), default="sem")
    parser.add_argument("--title", default="Offline Gradient Conflict")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_layerwise(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            block = int(row["block"])
            rows[block] = {
                metric: float(row[metric])
                for metric in METRICS
                if row.get(metric, "").strip()
            }
    return rows


def read_summary(path):
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_runs(run_specs):
    runs = []
    for spec in run_specs:
        if ":" not in spec:
            raise ValueError(f"Run spec must be seed:path, got {spec!r}")
        seed, path_str = spec.split(":", 1)
        path = Path(path_str)
        tsv_path = path / "gc_layerwise.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(tsv_path)
        runs.append({
            "seed": seed,
            "path": str(path),
            "layerwise": read_layerwise(tsv_path),
            "summary": read_summary(path),
        })
    return runs


def aggregate(runs, error_kind):
    records = []
    for block in range(12):
        record = {"block": block}
        for metric in METRICS:
            vals = np.array([
                run["layerwise"][block][metric]
                for run in runs
                if block in run["layerwise"] and metric in run["layerwise"][block]
            ], dtype=np.float64)
            if vals.size == 0:
                record[f"{metric}_mean"] = np.nan
                record[f"{metric}_std"] = np.nan
                record[f"{metric}_sem"] = np.nan
                record[f"{metric}_n"] = 0
                continue
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            sem = float(std / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            record[f"{metric}_mean"] = float(vals.mean())
            record[f"{metric}_std"] = std
            record[f"{metric}_sem"] = sem
            record[f"{metric}_n"] = int(vals.size)
            record[f"{metric}_err"] = sem if error_kind == "sem" else std
        records.append(record)
    return records


def write_outputs(output_dir, runs, records, error_kind, title, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "gc_errorbar_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "error": error_kind,
            "num_runs": len(runs),
            "runs": [
                {"seed": run["seed"], "path": run["path"], "summary": run["summary"]}
                for run in runs
            ],
        }, f, indent=2, ensure_ascii=False)

    fieldnames = ["block"]
    for metric in METRICS:
        fieldnames.extend([
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_sem",
            f"{metric}_n",
        ])
    with open(output_dir / "gc_layerwise_errorbar.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    x = np.arange(12)
    plt.figure(figsize=(8, 4.8))
    for metric in METRICS:
        y = np.array([record[f"{metric}_mean"] for record in records], dtype=np.float64)
        yerr = np.array([record[f"{metric}_{error_kind}"] for record in records], dtype=np.float64)
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            linewidth=2,
            capsize=3,
            elinewidth=1,
            label=LABELS[metric],
            **STYLES[metric],
        )
    plt.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("Text ResBlock")
    plt.ylabel("Gradient Conflict")
    plt.title(title)
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gc_plot_errorbar.png", dpi=dpi)
    plt.savefig(output_dir / "gc_plot_errorbar.pdf")
    plt.close()

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.errorbar(
            x,
            np.array([record["gc_mean"] for record in records], dtype=np.float64),
            yerr=np.array([record[f"gc_{error_kind}"] for record in records], dtype=np.float64),
            marker="o",
            markersize=8,
            linewidth=2.8,
            capsize=3.5,
            elinewidth=1.4,
            color="tab:blue",
            label="w/o PCGrad (Ours)",
        )
        ax.errorbar(
            x,
            np.array([record["gc_bidir_mean"] for record in records], dtype=np.float64),
            yerr=np.array([record[f"gc_bidir_{error_kind}"] for record in records], dtype=np.float64),
            marker="^",
            markersize=8,
            linewidth=2.8,
            linestyle=":",
            capsize=3.5,
            elinewidth=1.4,
            color="tab:green",
            label="w/ PCGrad (Ours)",
        )
        ax.axhline(0.0, color="gray", linewidth=1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Text Transformer Block Index")
        ax.set_ylabel("Gradient Conflict")
        ax.set_xticks([0, 5, 10])
        ax.set_xlim(-0.4, 11.4)
        ax.legend(loc="upper left", framealpha=0.95)
        fig.tight_layout(pad=0.5)
        fig.subplots_adjust(left=0.15, bottom=0.16)
        fig.savefig(output_dir / "gc_plot_iclr_errorbar.png", dpi=dpi)
        fig.savefig(output_dir / "gc_plot_iclr_errorbar.pdf")
        plt.close(fig)

    macros = {}
    for metric in METRICS:
        mean_key = f"{metric}_mean"
        err_key = f"{metric}_{error_kind}"
        macros[metric] = {
            "mean_over_layers": float(np.nanmean([record[mean_key] for record in records])),
            f"{error_kind}_over_layers_mean": float(np.nanmean([record[err_key] for record in records])),
        }

    lines = [
        "# Gradient Conflict Error-Bar Probe",
        "",
        f"- runs: {len(runs)} seeds",
        f"- error bar: per-layer {error_kind.upper()} across seeds",
        "",
        "## Macro Means",
        "",
        "| metric | mean over layers | mean error bar |",
        "| --- | ---: | ---: |",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {macros[metric]['mean_over_layers']:.6f} | "
            f"{macros[metric][f'{error_kind}_over_layers_mean']:.6f} |"
        )
    lines.extend([
        "",
        "## Runs",
        "",
        "| seed | path | GC macro | GC after macro | GC bidir macro |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for run in runs:
        summary = run["summary"]
        lines.append(
            f"| {run['seed']} | `{run['path']}` | "
            f"{summary.get('gc_macro_mean', float('nan')):.6f} | "
            f"{summary.get('gc_after_macro_mean', float('nan')):.6f} | "
            f"{summary.get('gc_bidir_macro_mean', float('nan')):.6f} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `gc_plot_errorbar.png`",
        "- `gc_plot_errorbar.pdf`",
        "- `gc_plot_iclr_errorbar.png`",
        "- `gc_plot_iclr_errorbar.pdf`",
        "- `gc_layerwise_errorbar.tsv`",
        "- `gc_errorbar_summary.json`",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    runs = load_runs(args.run)
    records = aggregate(runs, args.error)
    write_outputs(Path(args.output_dir), runs, records, args.error, args.title, args.dpi)


if __name__ == "__main__":
    main()
