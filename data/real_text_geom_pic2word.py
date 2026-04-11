import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from real_text_geom_stats import (
    build_retrieval_model,
    collect_real_samples,
    compute_image_anchor_stats,
    compute_stats,
)


def parse_args():
    parser = argparse.ArgumentParser("Pic2Word real caption/instruction geometry statistics")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-L/14")
    parser.add_argument("--img2text-arch", type=str, default="im2text")
    parser.add_argument("--middle-dim", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--retrieval-prompt-connector", type=str, default="and")
    parser.add_argument("--wds-shards", type=str, required=True)
    parser.add_argument("--cc3m-cir-jsonl", type=str, required=True)
    parser.add_argument("--wds-image-key", type=str, default="jpg;png;jpeg;webp")
    parser.add_argument("--wds-text-key", type=str, default="txt;text;caption")
    parser.add_argument("--wds-shuffle", type=int, default=10000)
    parser.add_argument("--wds-shardshuffle", type=int, default=1000)
    parser.add_argument("--wds-resampled", action="store_true", default=True)
    parser.add_argument("--wds-deterministic", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args.use_image_anchor = False
    sample_pack_text = collect_real_samples(args)
    model, img2text = build_retrieval_model(args, ckpt_path=args.checkpoint)

    args.use_image_anchor = True
    sample_pack_image = collect_real_samples(args)

    results = {
        "sample_info": {
            "requested_num_samples": sample_pack_text["requested_num_samples"],
            "valid_num_samples": sample_pack_text["valid_num_samples"],
            "model": args.model,
            "checkpoint": args.checkpoint,
            "retrieval_prompt_connector": args.retrieval_prompt_connector,
        }
    }

    _, text_metrics = compute_stats("pic2word_pretrained", model, sample_pack_text, args.gpu)
    results["text_anchor_stats"] = text_metrics

    _, image_metrics = compute_image_anchor_stats(
        "pic2word_pretrained",
        model,
        img2text,
        sample_pack_image,
        args.gpu,
        args.retrieval_prompt_connector,
    )
    results["image_anchor_stats"] = image_metrics

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved stats to %s", output_path)


if __name__ == "__main__":
    main()
