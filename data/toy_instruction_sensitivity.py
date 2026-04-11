import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from real_text_geom_stats import (
    build_retrieval_model,
    collect_real_samples,
    encode_image_anchor_query,
    encode_texts,
)


STOPWORDS = {
    "a", "an", "the", "to", "with", "and", "of", "in", "on", "for", "from", "into",
    "at", "by", "up", "down", "left", "right", "more", "less", "very", "not", "no",
    "but", "change", "make", "turn", "replace", "add", "remove", "is", "be", "as",
    "it", "its", "this", "that", "these", "those", "photo", "image", "object",
}


def parse_args():
    parser = argparse.ArgumentParser("Toy instruction sensitivity on real samples")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wds-shards", type=str, required=True)
    parser.add_argument("--cc3m-cir-jsonl", type=str, required=True)
    parser.add_argument("--wds-image-key", type=str, default="jpg;png;jpeg;webp")
    parser.add_argument("--wds-text-key", type=str, default="txt;text;caption")
    parser.add_argument("--wds-shuffle", type=int, default=10000)
    parser.add_argument("--wds-shardshuffle", type=int, default=1000)
    parser.add_argument("--wds-resampled", action="store_true", default=True)
    parser.add_argument("--wds-deterministic", action="store_true", default=True)
    return parser.parse_args()


def mask_instruction(text: str) -> str:
    pieces = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    out = []
    for p in pieces:
        if p.isalpha():
            if p.lower() in STOPWORDS:
                out.append(p)
            else:
                out.append("something")
        else:
            out.append(p)
    masked = "".join(out)
    masked = re.sub(r"(something(?:\s+something)+)", "something", masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked if masked else "something"


def build_prompts(instructions, connector):
    original = [f"a photo of * {connector} {ins}".strip() for ins in instructions]
    masked = [f"a photo of * {connector} {mask_instruction(ins)}".strip() for ins in instructions]
    swapped_ins = instructions[1:] + instructions[:1]
    swapped = [f"a photo of * {connector} {ins}".strip() for ins in swapped_ins]
    return original, masked, swapped, swapped_ins


def summarize_retrieval(query_feats, target_feats):
    sim = query_feats @ target_feats.t()
    n = sim.shape[0]
    diag = sim.diag()
    eye = torch.eye(n, dtype=torch.bool)
    neg = sim.masked_fill(eye, float("-inf"))
    hard_neg, _ = neg.max(dim=1)
    mean_neg = sim.masked_fill(eye, 0.0).sum(dim=1) / max(n - 1, 1)
    top1 = (sim.argmax(dim=1) == torch.arange(n)).float().mean()
    return {
        "mean_true_cos": float(diag.mean().item()),
        "mean_hard_negative_cos": float(hard_neg.mean().item()),
        "mean_negative_cos": float(mean_neg.mean().item()),
        "mean_margin_to_mean_negative": float((diag - mean_neg).mean().item()),
        "mean_margin_to_hard_negative": float((diag - hard_neg).mean().item()),
        "top1_acc": float(top1.item()),
    }


def evaluate_model(name, model, img2text, sample_pack, gpu, connector):
    images = sample_pack["images"]
    target_feats = encode_texts(model, sample_pack["tgt"], gpu)
    original_prompts, masked_prompts, swapped_prompts, swapped_ins = build_prompts(sample_pack["fwd"], connector)

    q_orig = encode_image_anchor_query(model, img2text, images, original_prompts, gpu)
    q_mask = encode_image_anchor_query(model, img2text, images, masked_prompts, gpu)
    q_swap = encode_image_anchor_query(model, img2text, images, swapped_prompts, gpu)

    cos_orig = (q_orig * target_feats).sum(dim=-1)
    cos_mask = (q_mask * target_feats).sum(dim=-1)
    cos_swap = (q_swap * target_feats).sum(dim=-1)

    stats_orig = summarize_retrieval(q_orig, target_feats)
    stats_mask = summarize_retrieval(q_mask, target_feats)
    stats_swap = summarize_retrieval(q_swap, target_feats)

    return {
        "mean_cos_original": float(cos_orig.mean().item()),
        "mean_cos_masked": float(cos_mask.mean().item()),
        "mean_cos_swapped": float(cos_swap.mean().item()),
        "mean_drop_original_to_masked": float((cos_orig - cos_mask).mean().item()),
        "mean_drop_original_to_swapped": float((cos_orig - cos_swap).mean().item()),
        "retrieval_original": stats_orig,
        "retrieval_masked": stats_mask,
        "retrieval_swapped": stats_swap,
        "margin_drop_original_to_masked": float(
            stats_orig["mean_margin_to_mean_negative"] - stats_mask["mean_margin_to_mean_negative"]
        ),
        "margin_drop_original_to_swapped": float(
            stats_orig["mean_margin_to_mean_negative"] - stats_swap["mean_margin_to_mean_negative"]
        ),
        "top1_drop_original_to_masked": float(stats_orig["top1_acc"] - stats_mask["top1_acc"]),
        "top1_drop_original_to_swapped": float(stats_orig["top1_acc"] - stats_swap["top1_acc"]),
        "samples": [
            {
                "target": sample_pack["tgt"][i],
                "instruction": sample_pack["fwd"][i],
                "masked_instruction": mask_instruction(sample_pack["fwd"][i]),
                "swapped_instruction": swapped_ins[i],
                "cos_original": float(cos_orig[i].item()),
                "cos_masked": float(cos_mask[i].item()),
                "cos_swapped": float(cos_swap[i].item()),
            }
            for i in range(len(sample_pack["tgt"]))
        ],
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Pic2Word / circobest-style ViT-L
    pic_args = argparse.Namespace(
        model="ViT-L/14",
        img2text_arch="im2text",
        middle_dim=512,
        n_layer=2,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.0,
        geo_lora_r=64,
        geo_lora_alpha=16,
        geo_lora_dropout=0.0,
        gpu=args.gpu,
        retrieval_prompt_connector="that",
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        wds_shards=args.wds_shards,
        cc3m_cir_jsonl=args.cc3m_cir_jsonl,
        wds_image_key=args.wds_image_key,
        wds_text_key=args.wds_text_key,
        wds_shuffle=args.wds_shuffle,
        wds_shardshuffle=args.wds_shardshuffle,
        wds_resampled=args.wds_resampled,
        wds_deterministic=args.wds_deterministic,
        use_image_anchor=True,
    )
    pic_samples = collect_real_samples(pic_args)
    pic_model, pic_img2text = build_retrieval_model(pic_args, ckpt_path="checkpoint/pic2word_model.pt")

    # Our merged model
    ours_args = argparse.Namespace(
        model="ViT-B/32",
        img2text_arch="phi",
        middle_dim=2048,
        n_layer=2,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.0,
        geo_lora_r=64,
        geo_lora_alpha=16,
        geo_lora_dropout=0.0,
        gpu=args.gpu,
        retrieval_prompt_connector="and",
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        wds_shards=args.wds_shards,
        cc3m_cir_jsonl=args.cc3m_cir_jsonl,
        wds_image_key=args.wds_image_key,
        wds_text_key=args.wds_text_key,
        wds_shuffle=args.wds_shuffle,
        wds_shardshuffle=args.wds_shardshuffle,
        wds_resampled=args.wds_resampled,
        wds_deterministic=args.wds_deterministic,
        use_image_anchor=True,
    )
    ours_samples = collect_real_samples(ours_args)
    ours_model, ours_img2text = build_retrieval_model(
        ours_args,
        ckpt_path="/tmp/vitb_sharedb_cirr_step1400_ties_realstats.pt",
    )
    ours_retrieval_model, ours_retrieval_img2text = build_retrieval_model(
        ours_args,
        ckpt_path=str(ROOT / "logs" / "DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_And_NoDrop_SharedB_CIRR_MergeCmp" / "checkpoints" / "epoch_0_step_1400.pt"),
    )

    if pic_samples["src"] != ours_samples["src"] or pic_samples["fwd"] != ours_samples["fwd"] or pic_samples["tgt"] != ours_samples["tgt"]:
        raise RuntimeError("Pic2Word and ours toy samples diverged; sample order must match for fair comparison.")

    results = {
        "sample_info": {
            "num_samples": args.num_samples,
            "pic2word_connector": "that",
            "ours_connector": "and",
        },
        "pic2word_vitl14": evaluate_model("pic2word_vitl14", pic_model, pic_img2text, pic_samples, args.gpu, "that"),
        "ours_retrieval_vitb32": evaluate_model("ours_retrieval_vitb32", ours_retrieval_model, ours_retrieval_img2text, ours_samples, args.gpu, "and"),
        "ours_merged_vitb32": evaluate_model("ours_merged_vitb32", ours_model, ours_img2text, ours_samples, args.gpu, "and"),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved stats to %s", out)


if __name__ == "__main__":
    main()
