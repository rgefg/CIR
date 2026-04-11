import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from data import get_cc3m_cir_wds
from main import TextEncoderBranch, apply_lora_to_linear_layers
from model.clip import load as load_clip
from model.model import IM2TEXT, Phi
from third_party.open_clip.clip import tokenize


def parse_args():
    parser = argparse.ArgumentParser("Real caption/instruction geometry statistics")
    parser.add_argument("--retrieval-ckpt", type=str, required=True)
    parser.add_argument("--merged-ckpt", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--img2text-arch", type=str, default="phi")
    parser.add_argument("--img2text-pretrained", type=str, default="")
    parser.add_argument("--middle-dim", type=int, default=2048)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--geo-lora-r", type=int, default=64)
    parser.add_argument("--geo-lora-alpha", type=int, default=16)
    parser.add_argument("--geo-lora-dropout", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
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
    parser.add_argument("--use-image-anchor", action="store_true", default=False)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not first_key.startswith("module."):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def build_data_args(args):
    return SimpleNamespace(
        batch_size=args.batch_size,
        workers=args.workers,
        distributed=False,
        world_size=1,
        rank=0,
        wds_resampled=args.wds_resampled,
        wds_deterministic=args.wds_deterministic,
        seed=args.seed,
        wds_shards=args.wds_shards,
        cc3m_cir_jsonl=args.cc3m_cir_jsonl,
        cc3m_cir_reverse_jsonl=None,
        wds_image_key=args.wds_image_key,
        wds_text_key=args.wds_text_key,
        wds_shuffle=args.wds_shuffle,
        wds_shardshuffle=args.wds_shardshuffle,
        dataset_type="cc3m_cir_wds",
        train_data="unused",
        debug=False,
    )


def tiny_preprocess(_img):
    return torch.zeros(1, dtype=torch.float32)


def collect_real_samples(args):
    data_args = build_data_args(args)
    preprocess = tiny_preprocess
    if args.use_image_anchor:
        _, _, preprocess = load_clip(args.model, jit=False)
    dataloader = get_cc3m_cir_wds(data_args, preprocess, is_train=True).dataloader
    iterator = iter(dataloader)
    src, tgt, fwd, rev = [], [], [], []
    imgs = []
    while len(src) < args.num_samples:
        batch = next(iterator)
        src.extend([str(x) for x in batch["src_caption"]])
        tgt.extend([str(x) for x in batch["modified_caption"]])
        fwd.extend([str(x) for x in batch["instruction"]])
        rev.extend([str(x) for x in batch["reverse_instruction"]])
        if args.use_image_anchor:
            imgs.append(batch["ref_img"].half().cpu())
    src = src[: args.num_samples]
    tgt = tgt[: args.num_samples]
    fwd = fwd[: args.num_samples]
    rev = rev[: args.num_samples]
    valid = [
        i for i, (s, t, fi, ri) in enumerate(zip(src, tgt, fwd, rev))
        if s.strip() and t.strip() and fi.strip() and ri.strip()
    ]
    out = {
        "src": [src[i] for i in valid],
        "tgt": [tgt[i] for i in valid],
        "fwd": [fwd[i] for i in valid],
        "rev": [rev[i] for i in valid],
        "requested_num_samples": args.num_samples,
        "valid_num_samples": len(valid),
    }
    if args.use_image_anchor:
        all_imgs = torch.cat(imgs, dim=0)[: args.num_samples]
        valid_idx = torch.as_tensor(valid, dtype=torch.long)
        out["images"] = all_imgs.index_select(0, valid_idx)
    return out


def build_img2text(base_model, args):
    if args.img2text_arch == "phi":
        return Phi(
            input_dim=base_model.embed_dim,
            hidden_dim=args.middle_dim,
            output_dim=base_model.token_embedding.weight.shape[1],
            dropout=0.5,
        )
    if args.img2text_arch == "im2text":
        return IM2TEXT(
            embed_dim=base_model.embed_dim,
            middle_dim=args.middle_dim,
            output_dim=base_model.token_embedding.weight.shape[1],
            n_layer=args.n_layer,
            dropout=0.1,
        )
    raise ValueError(f"Unsupported img2text_arch: {args.img2text_arch}")


def build_retrieval_model(args, ckpt_path=None):
    model, _, _ = load_clip(args.model, jit=False)
    apply_lora_to_linear_layers(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    img2text = build_img2text(model, args)
    if ckpt_path is not None:
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")
        state_dict = strip_module_prefix(ckpt["state_dict"])
        model.load_state_dict(state_dict, strict=False)
        if "state_dict_img2text" in ckpt and ckpt["state_dict_img2text"] is not None:
            img2text.load_state_dict(strip_module_prefix(ckpt["state_dict_img2text"]), strict=False)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device).float().eval()
    img2text = img2text.to(device).float().eval()
    return model, img2text


def build_geo_model(args, retrieval_ckpt_path):
    base_model, _, _ = load_clip(args.model, jit=False)
    apply_lora_to_linear_layers(
        base_model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    geo_model = TextEncoderBranch(base_model)
    apply_lora_to_linear_layers(
        geo_model,
        r=args.geo_lora_r,
        alpha=args.geo_lora_alpha,
        dropout=args.geo_lora_dropout,
    )
    ckpt = safe_torch_load(retrieval_ckpt_path, map_location="cpu")
    geo_model.load_state_dict(strip_module_prefix(ckpt["state_dict_geo_text"]), strict=False)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    geo_model = geo_model.to(device).float().eval()
    return geo_model


@torch.no_grad()
def encode_texts(text_encoder, texts, gpu, chunk_size=256):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    outs = []
    for start in range(0, len(texts), chunk_size):
        batch = texts[start:start + chunk_size]
        tokens = tokenize(batch, truncate=True).to(device)
        feats = text_encoder.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        outs.append(feats.float().cpu())
    return torch.cat(outs, dim=0)


def compute_stats(name, text_encoder, sample_pack, gpu):
    src = sample_pack["src"]
    tgt = sample_pack["tgt"]
    fwd = sample_pack["fwd"]
    rev = sample_pack["rev"]
    composed = [f"{s}, but {ins}" for s, ins in zip(src, fwd)]

    z_src = encode_texts(text_encoder, src, gpu)
    z_tgt = encode_texts(text_encoder, tgt, gpu)
    z_fwd = encode_texts(text_encoder, fwd, gpu)
    z_rev = encode_texts(text_encoder, rev, gpu)
    z_cmp = encode_texts(text_encoder, composed, gpu)

    delta_fwd = z_tgt - z_src
    delta_fwd = delta_fwd / delta_fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta_rev = z_src - z_tgt
    delta_rev = delta_rev / delta_rev.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    add_fwd = z_src + z_fwd
    add_fwd = add_fwd / add_fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    metrics = {
        "mean_cos_compose_to_target": float((z_cmp * z_tgt).sum(dim=-1).mean().item()),
        "mean_cos_delta_to_forward_instruction": float((delta_fwd * z_fwd).sum(dim=-1).mean().item()),
        "mean_cos_reverse_delta_to_reverse_instruction": float((delta_rev * z_rev).sum(dim=-1).mean().item()),
        "mean_cos_additive_to_target": float((add_fwd * z_tgt).sum(dim=-1).mean().item()),
        "std_cos_compose_to_target": float((z_cmp * z_tgt).sum(dim=-1).std(unbiased=False).item()),
        "std_cos_delta_to_forward_instruction": float((delta_fwd * z_fwd).sum(dim=-1).std(unbiased=False).item()),
        "std_cos_reverse_delta_to_reverse_instruction": float((delta_rev * z_rev).sum(dim=-1).std(unbiased=False).item()),
        "std_cos_additive_to_target": float((add_fwd * z_tgt).sum(dim=-1).std(unbiased=False).item()),
    }
    return name, metrics


@torch.no_grad()
def encode_image_anchor_query(model, img2text, images_cpu, prompts, gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    placeholder_token_id = int(tokenize(["*"])[0][1].item())
    outs = []
    for start in range(0, len(prompts), 128):
        batch_prompts = prompts[start:start + 128]
        batch_images = images_cpu[start:start + 128].to(device=device, dtype=torch.float32, non_blocking=True)
        image_features = model.encode_image(batch_images)
        token_features = img2text(image_features)
        text_tokens = tokenize(batch_prompts, truncate=True).to(device)
        feats = model.encode_text_img_vis(text_tokens, token_features, split_ind=placeholder_token_id)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        outs.append(feats.float().cpu())
    return torch.cat(outs, dim=0)


def compute_image_anchor_stats(name, model, img2text, sample_pack, gpu, connector):
    src = sample_pack["src"]
    tgt = sample_pack["tgt"]
    fwd = sample_pack["fwd"]
    rev = sample_pack["rev"]
    images = sample_pack["images"]

    target_feats = encode_texts(model, tgt, gpu)
    fwd_feats = encode_texts(model, fwd, gpu)
    rev_feats = encode_texts(model, rev, gpu)

    anchor_prompts = ["a photo of *"] * len(tgt)
    composed_prompts = [f"a photo of * {connector} {ins}" for ins in fwd]

    anchor_feats = encode_image_anchor_query(model, img2text, images, anchor_prompts, gpu)
    composed_feats = encode_image_anchor_query(model, img2text, images, composed_prompts, gpu)

    delta_fwd = target_feats - anchor_feats
    delta_fwd = delta_fwd / delta_fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta_rev = anchor_feats - target_feats
    delta_rev = delta_rev / delta_rev.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    add_fwd = anchor_feats + fwd_feats
    add_fwd = add_fwd / add_fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    metrics = {
        "mean_cos_compose_to_target": float((composed_feats * target_feats).sum(dim=-1).mean().item()),
        "mean_cos_delta_to_forward_instruction": float((delta_fwd * fwd_feats).sum(dim=-1).mean().item()),
        "mean_cos_reverse_delta_to_reverse_instruction": float((delta_rev * rev_feats).sum(dim=-1).mean().item()),
        "mean_cos_additive_to_target": float((add_fwd * target_feats).sum(dim=-1).mean().item()),
        "std_cos_compose_to_target": float((composed_feats * target_feats).sum(dim=-1).std(unbiased=False).item()),
        "std_cos_delta_to_forward_instruction": float((delta_fwd * fwd_feats).sum(dim=-1).std(unbiased=False).item()),
        "std_cos_reverse_delta_to_reverse_instruction": float((delta_rev * rev_feats).sum(dim=-1).std(unbiased=False).item()),
        "std_cos_additive_to_target": float((add_fwd * target_feats).sum(dim=-1).std(unbiased=False).item()),
    }
    return name, metrics


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    torch.manual_seed(args.seed)

    sample_pack = collect_real_samples(args)
    logging.info("Collected %d valid real samples", sample_pack["valid_num_samples"])

    pretrained_model, pretrained_img2text = build_retrieval_model(args, ckpt_path=None)
    retrieval_model, retrieval_img2text = build_retrieval_model(args, ckpt_path=args.retrieval_ckpt)
    merged_model, merged_img2text = build_retrieval_model(args, ckpt_path=args.merged_ckpt)
    geo_model = build_geo_model(args, args.retrieval_ckpt)

    results = {
        "sample_info": {
            "requested_num_samples": sample_pack["requested_num_samples"],
            "valid_num_samples": sample_pack["valid_num_samples"],
            "model": args.model,
            "retrieval_ckpt": args.retrieval_ckpt,
            "merged_ckpt": args.merged_ckpt,
            "retrieval_prompt_connector": args.retrieval_prompt_connector,
            "use_image_anchor": bool(args.use_image_anchor),
        }
    }

    for name, text_encoder in (
        ("pretrained_openai", pretrained_model),
        ("retrieval_branch", retrieval_model),
        ("geo_branch", geo_model),
        ("merged_ties", merged_model),
    ):
        logging.info("Computing stats for %s", name)
        key, metrics = compute_stats(name, text_encoder, sample_pack, args.gpu)
        results[key] = metrics

    if args.use_image_anchor:
        image_anchor_results = {}
        for name, model_obj, img2text_obj in (
            ("retrieval_branch", retrieval_model, retrieval_img2text),
            ("merged_ties", merged_model, merged_img2text),
        ):
            logging.info("Computing image-anchor stats for %s", name)
            key, metrics = compute_image_anchor_stats(
                name,
                model_obj,
                img2text_obj,
                sample_pack,
                args.gpu,
                args.retrieval_prompt_connector,
            )
            image_anchor_results[key] = metrics
        results["image_anchor_stats"] = image_anchor_results

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Saved stats to %s", output_path)


if __name__ == "__main__":
    main()
