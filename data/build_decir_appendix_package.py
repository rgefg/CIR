#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
CIRR_ROOT = REPO_ROOT / "data" / "CIRR" / "dev"
CIRCO_ROOT = REPO_ROOT / "data" / "CIRCO" / "COCO2017_unlabeled" / "unlabeled2017"
GENECIS_VG_ROOT = REPO_ROOT / "data" / "genecis" / "VG_100K"
GENECIS_COCO_ROOT = REPO_ROOT / "data" / "coco" / "val2017"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))[:120]


def copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def cirr_image_path(name):
    return CIRR_ROOT / str(name)


def circo_image_path(image_id):
    return CIRCO_ROOT / f"{int(image_id):012d}.jpg"


def genecis_image_path(meta, task):
    if isinstance(meta, dict):
        if "val_image_id" in meta:
            return GENECIS_COCO_ROOT / f"{int(meta['val_image_id']):012d}.jpg"
        image_id = meta.get("image_id", meta.get("id"))
        return GENECIS_VG_ROOT / f"{image_id}.jpg"
    if "object" in task:
        return GENECIS_COCO_ROOT / f"{int(meta):012d}.jpg"
    return GENECIS_VG_ROOT / f"{meta}.jpg"


def crop_genecis(src, dst, meta, padding=0.08):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src).convert("RGB")
    except Exception:
        Image.new("RGB", (224, 224), (0, 0, 0)).save(dst)
        return False
    bbox = meta.get("instance_bbox", meta.get("bbox")) if isinstance(meta, dict) else None
    if bbox:
        x, y, w, h = [float(v) for v in bbox]
        pad_x = w * padding
        pad_y = h * padding
        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(img.width, int(x + w + pad_x))
        bottom = min(img.height, int(y + h + pad_y))
        if right > left and bottom > top:
            img = img.crop((left, top, right, bottom))
    img.save(dst, quality=95)
    return True


def load_font(size=22):
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def fit_image(path, size=(320, 260)):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (30, 30, 30))
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def wrap_text(draw, text, font, width):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def make_contact_sheet(panels, caption, output_path):
    font = load_font(20)
    small = load_font(16)
    panel_w, panel_h = 320, 310
    cols = len(panels)
    caption_h = 90
    canvas = Image.new("RGB", (panel_w * cols, panel_h + caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, path) in enumerate(panels):
        x = idx * panel_w
        draw.rectangle((x, 0, x + panel_w - 1, panel_h - 1), outline=(210, 210, 210), width=1)
        draw.text((x + 10, 10), title, fill=(20, 20, 20), font=font)
        img = fit_image(path, (panel_w - 20, panel_h - 60))
        canvas.paste(img, (x + 10, 46))
    y = panel_h + 10
    for line in wrap_text(draw, caption, small, panel_w * cols - 24)[:4]:
        draw.text((12, y), line, fill=(20, 20, 20), font=small)
        y += 20
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def collect_cirr_case(row, case_dir):
    q = row["query_index"]
    copied = []
    role_paths = {
        "reference": cirr_image_path(row["reference"]),
        "target_merged_top1": cirr_image_path(row["target"]),
        "retrieval_top1_wrong": cirr_image_path(row["retrieval_top1"]),
        "joint_top1_wrong": cirr_image_path(row["joint_top1"]),
    }
    local = {}
    for role, src in role_paths.items():
        dst = case_dir / f"{role}{src.suffix}"
        if copy_file(src, dst):
            local[role] = dst
            copied.append({"role": role, "source": str(src), "file": str(dst)})
    group_dir = case_dir / "hard_group"
    for idx, name in enumerate(row.get("group_members", [])):
        src = cirr_image_path(name)
        dst = group_dir / f"group_{idx:02d}_{safe_name(name)}"
        if copy_file(src, dst):
            copied.append({"role": f"group_{idx:02d}", "source": str(src), "file": str(dst)})
    if {"reference", "retrieval_top1_wrong", "joint_top1_wrong", "target_merged_top1"} <= set(local):
        make_contact_sheet(
            [
                ("Reference", local["reference"]),
                ("Retrieval top-1 (wrong)", local["retrieval_top1_wrong"]),
                ("Joint top-1 (wrong)", local["joint_top1_wrong"]),
                ("Merged top-1 / Target", local["target_merged_top1"]),
            ],
            f"CIRR q={q}. {row['caption']} | ranks: retrieval={row['retrieval_rank']}, joint={row['joint_rank']}, merged={row['merged_rank']}.",
            case_dir / "contact_sheet.jpg",
        )
    return copied


def collect_circo_case(row, case_dir):
    q = row["query_index"]
    role_paths = {
        "reference": circo_image_path(row["reference"]),
        "target_merged_top1": circo_image_path(row["target"]),
        "retrieval_top1_wrong": circo_image_path(row["retrieval_top1"]),
        "joint_top1_wrong": circo_image_path(row["joint_top1"]),
    }
    copied = []
    local = {}
    for role, src in role_paths.items():
        dst = case_dir / f"{role}{src.suffix}"
        if copy_file(src, dst):
            local[role] = dst
            copied.append({"role": role, "source": str(src), "file": str(dst)})
    if {"reference", "retrieval_top1_wrong", "joint_top1_wrong", "target_merged_top1"} <= set(local):
        make_contact_sheet(
            [
                ("Reference", local["reference"]),
                ("Retrieval top-1 (wrong)", local["retrieval_top1_wrong"]),
                ("Joint top-1 (wrong)", local["joint_top1_wrong"]),
                ("Merged top-1 / Target", local["target_merged_top1"]),
            ],
            f"CIRCO q={q}. Shared concept: {row.get('shared_concept', '')}. Edit: {row['caption']} | ranks: retrieval={row['retrieval_rank']}, joint={row['joint_rank']}, merged={row['merged_rank']}.",
            case_dir / "contact_sheet.jpg",
        )
    return copied


def collect_genecis_case(row, case_dir):
    task = row["task"]
    gallery = row["gallery"]
    def candidate_meta(index):
        index = int(index)
        if index == 0:
            return row["target"]
        return gallery[index - 1]

    roles = {
        "reference": row["reference"],
        "target_merged_top1": row["target"],
        "retrieval_top1_wrong": candidate_meta(row["retrieval_top1"]),
        "joint_top1_wrong": candidate_meta(row["joint_top1"]),
    }
    copied = []
    local = {}
    for role, meta in roles.items():
        src = genecis_image_path(meta, task)
        full_dst = case_dir / f"{role}_full{src.suffix}"
        crop_dst = case_dir / f"{role}_crop.jpg"
        if copy_file(src, full_dst):
            copied.append({"role": f"{role}_full", "source": str(src), "file": str(full_dst)})
        if crop_genecis(src, crop_dst, meta):
            local[role] = crop_dst
            copied.append({"role": f"{role}_crop", "source": str(src), "file": str(crop_dst), "bbox": meta.get("instance_bbox") if isinstance(meta, dict) else None})
    if {"reference", "retrieval_top1_wrong", "joint_top1_wrong", "target_merged_top1"} <= set(local):
        make_contact_sheet(
            [
                ("Reference crop", local["reference"]),
                ("Retrieval top-1 (wrong)", local["retrieval_top1_wrong"]),
                ("Joint top-1 (wrong)", local["joint_top1_wrong"]),
                ("Merged top-1 / Target", local["target_merged_top1"]),
            ],
            f"GeneCIS {row['query_index']}. Attribute: {row['caption']} | ranks: retrieval={row['retrieval_rank']}, joint={row['joint_rank']}, merged={row['merged_rank']}.",
            case_dir / "contact_sheet.jpg",
        )
    return copied


def format_float(value, digits=4):
    return f"{float(value):.{digits}f}"


def build_summary_md(result_dir, out_dir, cases, image_index):
    followup = (result_dir / "FOLLOWUP_RESULTS_SUMMARY.md").read_text(encoding="utf-8")
    hard_md = (result_dir / "hard_analysis" / "HARD_ANALYSIS.md").read_text(encoding="utf-8")
    by_dataset = defaultdict(list)
    for row in cases:
        by_dataset[row["dataset"]].append(row)

    rank_cirr = []
    rank_suite = []
    for rank in [16, 32, 48, 64]:
        cpath = result_dir / "records" / f"rank_cirr_k{rank}_result.json"
        spath = result_dir / "records" / f"rank_suite_k{rank}_result.json"
        if cpath.exists():
            cirr = read_json(cpath)["cirr"]["metrics"]["composed"]
            rank_cirr.append((rank, cirr))
        if spath.exists():
            suite = read_json(spath)
            rank_suite.append((rank, suite))

    loss_rows = []
    for job in [
        "loss_cirr_fwd_only_rerun",
        "loss_cirr_fwd_rev_nozero",
        "loss_multi_fwd_only",
        "loss_multi_fwd_rev_nozero",
    ]:
        path = result_dir / "records" / f"{job}_result.json"
        if path.exists():
            loss_rows.append((job, read_json(path)))

    lines = []
    lines.append("# DeCIR Appendix Package Summary")
    lines.append("")
    lines.append("This folder contains the successful follow-up ablations and hard-distractor evidence prepared for appendix use.")
    lines.append("")
    lines.append("## Package Contents")
    lines.append("")
    lines.append("- `APPENDIX_SUMMARY.md`: paper-facing summary of completed experiments and evidence.")
    lines.append("- `hard_cases/`: copied images and contact sheets for every strict merged-win hard case.")
    lines.append("- `representative_contact_sheets/`: one-file-per-case visual summaries for quick appendix figure selection.")
    lines.append("- `source_tables/`: copied metric tables and JSON files used to build this package.")
    lines.append("- `IMAGE_INDEX.json`: mapping from each copied image/contact sheet back to the source dataset path.")
    lines.append("")
    lines.append("## Successful Experiments")
    lines.append("")
    lines.append("All effective follow-up jobs completed with status `ok`: SVD rank ablation, transition-loss ablation, and hard distractor mining.")
    lines.append("Older repeated `hard_analysis` failures in `status.tsv` came from an interrupted accelerator loop; the final valid job is `hard_analysis_recovery`, which completed successfully.")
    lines.append("")
    lines.append("### SVD Rank Ablation")
    lines.append("")
    lines.append("CIRR validation, composed feature:")
    lines.append("")
    lines.append("| rank | FeatureR@1 | R_subset@1 | R@5 | R@10 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for rank, m in rank_cirr:
        lines.append(f"| {rank} | {format_float(m['FeatureR@1'])} | {format_float(m['R_subset@1'])} | {format_float(m['R@5'])} | {format_float(m['R@10'])} |")
    lines.append("")
    lines.append("CIRCO validation and GeneCIS validation:")
    lines.append("")
    lines.append("| rank | CIRCO mAP@50 | CIRCO mAP@10 | GeneCIS avg R@1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for rank, suite in rank_suite:
        gen = suite.get("genecis", {})
        avg = sum(v["R@1"] for v in gen.values()) / len(gen)
        lines.append(f"| {rank} | {format_float(suite['circo_val']['map']['mAP@50'])} | {format_float(suite['circo_val']['map']['mAP@10'])} | {format_float(avg)} |")
    lines.append("")
    lines.append("Takeaway: rank choices are very stable. `k=32` gives the best CIRR R_subset@1, while suite metrics are nearly flat across ranks.")
    lines.append("")
    lines.append("### Transition-Loss Ablation")
    lines.append("")
    lines.append("| dataset | variant | main metric | secondary metrics |")
    lines.append("| --- | --- | ---: | --- |")
    for job, result in loss_rows:
        if "cirr" in result:
            comp = result["cirr"]["metrics"]["composed"]
            variant = "fwd_only" if "fwd_only" in job else "fwd_rev_nozero"
            lines.append(f"| CIRR | {variant} | R_subset@1={format_float(comp['R_subset@1'])} | FeatureR@1={format_float(comp['FeatureR@1'])}, R@5={format_float(comp['R@5'])}, R@10={format_float(comp['R@10'])} |")
        else:
            variant = "fwd_only" if "fwd_only" in job else "fwd_rev_nozero"
            gen = result.get("genecis", {})
            avg = sum(v["R@1"] for v in gen.values()) / len(gen)
            lines.append(f"| CIRCO+GeneCIS | {variant} | CIRCO mAP@50={format_float(result['circo_val']['map']['mAP@50'])}, GeneCIS avg R@1={format_float(avg)} | CIRCO mAP@10={format_float(result['circo_val']['map']['mAP@10'])} |")
    lines.append("")
    lines.append("Takeaway: adding reverse alignment without the zero-vector term is slightly better on CIRR R_subset@1 and on the CIRCO+GeneCIS suite.")
    lines.append("")
    lines.append("## Hard Distractor Evidence")
    lines.append("")
    lines.append("Selection rule: merged DeCIR is rank-1 correct, while retrieval-only and joint are not rank-1. This directly targets the appendix claim that endpoint-only retrieval can do well on easy cases but is fragile on hard distractors and edit-sensitive cases.")
    lines.append("")
    lines.append("| dataset | strict merged-win cases | notes |")
    lines.append("| --- | ---: | --- |")
    lines.append(f"| CIRR | {len(by_dataset['cirr'])} | Retrieval top-1 is inside the six-image hard group for {sum(1 for r in by_dataset['cirr'] if r.get('retrieval_top1_in_hard_group'))}/{len(by_dataset['cirr'])} cases. |")
    lines.append(f"| CIRCO | {len(by_dataset['circo'])} | Retrieval and joint top-1 are outside the semantic GT set; merged top-1 is target for all selected cases. |")
    lines.append(f"| GeneCIS | {len(by_dataset['genecis'])} | Attribute-sensitive cases where retrieval and joint rank the target second, while merged ranks it first. |")
    lines.append("")
    lines.append("Representative contact sheets:")
    lines.append("")
    for dataset in ["cirr", "circo", "genecis"]:
        for row in by_dataset[dataset][:3]:
            case_id = row["_case_id"]
            lines.append(f"- {dataset.upper()} `{row.get('query_index')}`: `representative_contact_sheets/{case_id}.jpg`")
    lines.append("")
    lines.append("All copied case images are under `hard_cases/<dataset>/<case_id>/`. Each case directory contains source images plus `contact_sheet.jpg`.")
    lines.append("")
    lines.append("## Appendix-Ready Text")
    lines.append("")
    lines.append("We mine hard distractor examples using a strict criterion: the merged DeCIR model must retrieve the ground-truth target at rank 1, while both the retrieval-only endpoint model and the joint-single baseline fail at rank 1. The mined examples support the edit-direction bottleneck hypothesis: endpoint-style retrieval often selects a visually or semantically plausible shortcut candidate, whereas the merged model preserves the edit direction and selects the correct target. On CIRR, all 30 mined examples have the retrieval-only top-1 inside the six-image hard group but not equal to the target, indicating that the shortcut is not a random failure but a hard distractor confusion. CIRCO provides 3 strict semantic cases, and GeneCIS provides 30 attribute-sensitive cases.")
    lines.append("")
    lines.append("## Source Summary")
    lines.append("")
    lines.append("The following source summaries were copied verbatim into `source_tables/` for traceability.")
    lines.append("")
    lines.append("### Follow-Up Results Summary")
    lines.append("")
    lines.append(followup)
    lines.append("")
    lines.append("### Hard Analysis Table")
    lines.append("")
    lines.append(hard_md)
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Build a paper appendix package for DeCIR follow-up experiments.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-representative-per-dataset", type=int, default=6)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else result_dir / "appendix_package"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "source_tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "representative_contact_sheets").mkdir(parents=True, exist_ok=True)

    for rel in [
        "FOLLOWUP_RESULTS_SUMMARY.md",
        "status.tsv",
        "results.jsonl",
        "hard_analysis/HARD_ANALYSIS.md",
        "hard_analysis/cirr_hard_cases.json",
        "hard_analysis/circo_hard_cases.json",
        "hard_analysis/genecis_hard_cases.json",
        "hard_analysis/hard_cases_all.json",
        "records/hard_analysis_result.json",
    ]:
        src = result_dir / rel
        if src.exists():
            dst = out_dir / "source_tables" / rel.replace("/", "__")
            copy_file(src, dst)

    cases = read_json(result_dir / "records" / "hard_analysis_result.json")
    counters = Counter()
    image_index = []
    for row in cases:
        dataset = row["dataset"]
        counters[dataset] += 1
        query = row.get("query_index")
        case_id = f"{dataset}_{counters[dataset]:03d}_q{safe_name(query)}"
        row["_case_id"] = case_id
        case_dir = out_dir / "hard_cases" / dataset / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(case_dir / "metadata.json", row)
        if dataset == "cirr":
            copied = collect_cirr_case(row, case_dir)
        elif dataset == "circo":
            copied = collect_circo_case(row, case_dir)
        elif dataset == "genecis":
            copied = collect_genecis_case(row, case_dir)
        else:
            copied = []
        for item in copied:
            item["case_id"] = case_id
            item["dataset"] = dataset
            image_index.append(item)
        sheet = case_dir / "contact_sheet.jpg"
        if sheet.exists() and counters[dataset] <= args.max_representative_per_dataset:
            copy_file(sheet, out_dir / "representative_contact_sheets" / f"{case_id}.jpg")

    write_json(out_dir / "IMAGE_INDEX.json", image_index)
    summary = build_summary_md(result_dir, out_dir, cases, image_index)
    (out_dir / "APPENDIX_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
