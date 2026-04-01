# trainer.py
# Copyright 2022 Google LLC
# Licensed under the Apache License, Version 2.0

import os
import time
import re
import random
from contextlib import contextmanager, nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import wandb
import json
from pathlib import Path

from torch.cuda.amp import autocast
from third_party.open_clip.clip import tokenize
from eval_utils import evaluate_cirr, evaluate_fashion, evaluate_genecis
from utils import is_master, use_ema_weights


SAFE_TOKENS = {
    # operations
    "change", "add", "remove", "make", "more", "less", "replace", "swap",
    "switch", "turn", "move", "keep", "increase", "decrease",
    # relations / directions / background
    "left", "right", "background", "foreground", "front", "back",
    "behind", "between", "around", "near", "next", "beside", "over", "under",
    "inside", "outside", "top", "bottom", "middle", "center", "centre",
    # common function words to avoid dropping
    "a", "an", "the", "and", "or", "of", "to", "from", "with", "without",
    "in", "on", "at", "by", "for", "into", "onto", "off", "up", "down",
    "is", "are", "be", "this", "that", "these", "those", "it", "its",
}
RISK_TOKEN_SET = None


def _tokenize_simple(text):
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _load_or_build_risk_tokens(args):
    global RISK_TOKEN_SET
    if RISK_TOKEN_SET is not None:
        return RISK_TOKEN_SET

    jsonl_path = getattr(args, "cc3m_cir_jsonl", None)
    if not jsonl_path or (not os.path.exists(jsonl_path)):
        if is_master(args):
            logging.warning("⚠️ Risk token mining skipped: cc3m_cir_jsonl not found.")
        RISK_TOKEN_SET = set()
        return RISK_TOKEN_SET

    min_count = 1000
    top_pct = 0.10
    cache_path = jsonl_path + f".risk_tokens.top{int(top_pct*100)}.min{min_count}.txt"

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            RISK_TOKEN_SET = set([line.strip() for line in f if line.strip()])
        if is_master(args):
            logging.info(f"✅ Loaded risk tokens from cache: {cache_path} (n={len(RISK_TOKEN_SET)})")
        return RISK_TOKEN_SET

    if is_master(args):
        logging.info("🔎 Mining risk tokens from cleaned jsonl (data-driven). This runs once.")
        count_t = {}
        count_co = {}
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("instruction", "")
                m = obj.get("modified_caption", "")
                t_set = set(_tokenize_simple(t))
                m_set = set(_tokenize_simple(m))
                for w in t_set:
                    count_t[w] = count_t.get(w, 0) + 1
                    if w in m_set:
                        count_co[w] = count_co.get(w, 0) + 1
        eligible = []
        for w, ct in count_t.items():
            if ct >= min_count:
                co = count_co.get(w, 0)
                rate = co / ct
                eligible.append((rate, w))
        eligible.sort(reverse=True)
        top_k = max(1, int(len(eligible) * top_pct))
        risk_tokens = [w for _, w in eligible[:top_k]]
        with open(cache_path, "w") as f:
            for w in risk_tokens:
                f.write(w + "\n")
        RISK_TOKEN_SET = set(risk_tokens)
        logging.info(
            f"✅ Mined risk tokens: eligible={len(eligible)} top_k={top_k} saved={cache_path}"
        )
    if args.distributed and dist.is_initialized():
        dist.barrier()
        if RISK_TOKEN_SET is None and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                RISK_TOKEN_SET = set([line.strip() for line in f if line.strip()])
    if RISK_TOKEN_SET is None:
        RISK_TOKEN_SET = set()
    return RISK_TOKEN_SET


def _is_high_risk_token(token, risk_set):
    core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token).lower()
    if not core:
        return False
    if core in SAFE_TOKENS:
        return False
    if core in risk_set:
        return True
    return False


def _safe_l2_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _build_retrieval_prompt(instruction, placeholder, args):
    inst_str = _to_text(instruction).strip()
    if not inst_str:
        return f"a photo of {placeholder}"
    return f"a photo of {placeholder} that {inst_str}"


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ["text", "value", "en", "caption", "modified_caption", "instruction", "reverse_instruction"]:
            if k in x and isinstance(x[k], str):
                return x[k]
        return ""
    if isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, str):
                return v
        for v in x:
            if isinstance(v, dict):
                vv = _to_text(v)
                if vv:
                    return vv
        return ""
    return str(x)


def get_loss(model, images, texts, loss_img, loss_txt, args, data_identifier=-1):
    # original CLIP loss (kept)
    if data_identifier == 1:
        image_features, text_features, logit_scale = model(images, texts, extra=True)
    else:
        image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat([image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1 :])
        all_text_features = torch.cat([text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1 :])

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss


def get_text_features(model, token_features, args):
    # original helper (kept)
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1).repeat(token_features.size(0), 1)
    text_features = model.encode_text_img(text, token_features)
    return text_features


def get_loss_img2text(model, img2text, images, loss_img, loss_txt, args, memory=None):
    # original img2text CLIP-style training (kept, but NOT used for CC3M CIR dict batches)
    with torch.no_grad():
        image_features = model.encode_image(images)
    token_features = img2text(image_features)
    text_features = get_text_features(model, token_features, args)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp().mean()

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat([image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1 :])
        all_text_features = torch.cat([text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1 :])

        ground_truth = torch.arange(len(all_image_features)).long().to(images.device)
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        loss_txt_val = loss_txt(logits_per_image.t(), ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long().to(images.device)
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        loss_txt_val = loss_txt((logit_scale * text_features @ image_features.t()), ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss


def get_loss_lcom_cc3m(model, img2text, ref_images, instructions, modified_captions, loss_fn, args):
    """
    Lcom-only (query->modified_caption) using '*' placeholder injection.

    query: h(i,t) = encode_text_img_vis(prompt_with_*, img2text(encode_image(i)), split_ind=4)
    pos/neg: h_m = encode_text(modified_caption)
    """
    device = ref_images.device

    # IMPORTANT: no torch.no_grad here, so LoRA in visual tower can learn
    
    # ============================================================
    # 🔍 图片加载检查（仅在第一次调用时打印）
    # ============================================================
    if not hasattr(get_loss_lcom_cc3m, '_checked_images'):
        img_min = ref_images.min().item()
        img_max = ref_images.max().item()
        img_mean = ref_images.mean().item()
        img_std = ref_images.std().item()
        logging.info(f"🔍 [Input Images] Shape: {ref_images.shape}, Range: [{img_min:.4f}, {img_max:.4f}], "
                    f"Mean: {img_mean:.4f}, Std: {img_std:.4f}")
        if torch.allclose(ref_images, torch.zeros_like(ref_images), atol=1e-6):
            logging.warning("🚨 WARNING: All input images are ZERO! Images may not be loaded correctly!")
        get_loss_lcom_cc3m._checked_images = True
    # ============================================================
    
    image_features = model.encode_image(ref_images)          # [B, embed_dim]
    
    # ============================================================
    # 🔍 Encoder输出检查（仅在第一次调用时打印）
    # ============================================================
    if not hasattr(get_loss_lcom_cc3m, '_checked_features'):
        feat_min = image_features.min().item()
        feat_max = image_features.max().item()
        feat_mean = image_features.mean().item()
        feat_std = image_features.std().item()
        feat_norm_mean = image_features.norm(dim=-1).mean().item()
        feat_norm_std = image_features.norm(dim=-1).std().item()
        
        logging.info(f"🔍 [Image Features] Shape: {image_features.shape}, Range: [{feat_min:.4f}, {feat_max:.4f}], "
                    f"Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")
        logging.info(f"🔍 [Image Features Norm] Mean: {feat_norm_mean:.4f}, Std: {feat_norm_std:.4f}")
        
        # 检查是否全零
        if torch.allclose(image_features, torch.zeros_like(image_features), atol=1e-6):
            logging.error("🚨 ERROR: All image features are ZERO! Encoder may not be working!")
        # 检查是否有NaN或Inf
        elif torch.isnan(image_features).any():
            logging.error("🚨 ERROR: Image features contain NaN values!")
        elif torch.isinf(image_features).any():
            logging.error("🚨 ERROR: Image features contain Inf values!")
        # 检查特征是否过于相似（可能encoder没有学到有效特征）
        elif feat_std < 0.01:
            logging.warning(f"⚠️  WARNING: Image features have very low std ({feat_std:.6f}), features may be too similar!")
        # 检查特征范数是否合理（CLIP特征通常归一化后范数接近1）
        elif feat_norm_mean < 0.1 or feat_norm_mean > 10.0:
            logging.warning(f"⚠️  WARNING: Image features norm is unusual ({feat_norm_mean:.4f}), expected ~1.0 for normalized features")
        else:
            logging.info("✅ Image features look normal")
        
        get_loss_lcom_cc3m._checked_features = True
    # ============================================================
    
    token_features = img2text(image_features)               # [B, transformer_width]
    
    # ============================================================
    # 🔍 IM2TEXT输出检查（仅在第一次调用时打印）
    # ============================================================
    if not hasattr(get_loss_lcom_cc3m, '_checked_tokens'):
        token_min = token_features.min().item()
        token_max = token_features.max().item()
        token_mean = token_features.mean().item()
        token_std = token_features.std().item()
        token_norm_mean = token_features.norm(dim=-1).mean().item()
        
        logging.info(f"🔍 [Token Features] Shape: {token_features.shape}, Range: [{token_min:.4f}, {token_max:.4f}], "
                    f"Mean: {token_mean:.4f}, Std: {token_std:.4f}, Norm: {token_norm_mean:.4f}")
        
        if torch.allclose(token_features, torch.zeros_like(token_features), atol=1e-6):
            logging.error("🚨 ERROR: All token features are ZERO! IM2TEXT may not be working!")
        elif torch.isnan(token_features).any():
            logging.error("🚨 ERROR: Token features contain NaN values!")
        elif torch.isinf(token_features).any():
            logging.error("🚨 ERROR: Token features contain Inf values!")
        else:
            logging.info("✅ Token features look normal")
        
        get_loss_lcom_cc3m._checked_tokens = True
    # ============================================================

    # prompt with placeholder token, and pass its TOKEN ID into encode_text_img_vis
    placeholder = getattr(args, "prompt_placeholder", "*")
    # tokenize(['*']) -> [SOS, <placeholder_token_id>, EOS, PAD...]
    placeholder_token_id = int(tokenize([placeholder])[0][1].item())
    
    # Handle empty instructions (from instruction dropout)
    # When instruction is empty, use simple prompt to force model to rely on visual features
    prompts = [_build_retrieval_prompt(inst, placeholder, args) for inst in instructions]
    
    # Use truncate=True to automatically truncate prompts that exceed context length (77 tokens)
    prompt_tokens = tokenize(prompts, truncate=True).to(device, non_blocking=True)

    # composed embedding (trainable through text-tower LoRA too)
    query_features = model.encode_text_img_vis(prompt_tokens, token_features, split_ind=placeholder_token_id)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    
    # ==============================================
    # ==============
    # 🔍 Query Features检查（仅在第一次调用时打印）
    # ============================================================
    if not hasattr(get_loss_lcom_cc3m, '_checked_query'):
        query_min = query_features.min().item()
        query_max = query_features.max().item()
        query_mean = query_features.mean().item()
        query_std = query_features.std().item()
        query_norm_mean = query_features.norm(dim=-1).mean().item()
        query_norm_std = query_features.norm(dim=-1).std().item()
        
        logging.info(f"🔍 [Query Features] Shape: {query_features.shape}, Range: [{query_min:.4f}, {query_max:.4f}], "
                    f"Mean: {query_mean:.4f}, Std: {query_std:.4f}")
        logging.info(f"🔍 [Query Features Norm] Mean: {query_norm_mean:.4f}, Std: {query_norm_std:.4f} (should be ~1.0 after normalization)")
        
        if torch.allclose(query_features, torch.zeros_like(query_features), atol=1e-6):
            logging.error("🚨 ERROR: All query features are ZERO!")
        elif torch.isnan(query_features).any():
            logging.error("🚨 ERROR: Query features contain NaN values!")
        elif torch.isinf(query_features).any():
            logging.error("🚨 ERROR: Query features contain Inf values!")
        elif abs(query_norm_mean - 1.0) > 0.1:
            logging.warning(f"⚠️  WARNING: Query features norm is {query_norm_mean:.4f}, expected ~1.0 after normalization")
        else:
            logging.info("✅ Query features look normal (normalized)")
        
        get_loss_lcom_cc3m._checked_query = True
    # ============================================================

    # modified caption embedding
    # CRITICAL: Extract modified_caption text with smart fallback, NEVER include brainstorming or instruction
    def _to_str(x):
        """
        Safely convert to string with smart fallback strategy.
        CRITICAL: Prioritize "modified_caption" key, NEVER include "brainstorming" or "instruction" content.
        This prevents contamination that could cause token truncation issues (CLIP max 77 tokens).
        
        Fallback strategy (in order):
        1. "modified_caption" key (highest priority)
        2. "caption" key (common alias)
        3. Other text keys (text, value, en) - but EXCLUDE "brainstorming" and "instruction"
        4. First string element in list
        5. Empty string (last resort)
        """
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        # Handle dict: smart fallback with exclusion of problematic keys
        if isinstance(x, dict):
            # Priority 1: "modified_caption" key (highest priority)
            if "modified_caption" in x and isinstance(x["modified_caption"], str):
                return x["modified_caption"]
            # Priority 2: "caption" key (common alias)
            if "caption" in x and isinstance(x["caption"], str):
                return x["caption"]
            # Priority 3: Other text keys, but EXCLUDE "brainstorming" and "instruction"
            # This provides fallback for data format variations while avoiding contamination
            excluded_keys = {"brainstorming", "instruction"}  # NEVER include these
            for k in ["text", "value", "en", "description", "content"]:
                if k in x and k not in excluded_keys and isinstance(x[k], str):
                    return x[k]
            # Last resort: return empty string (don't stringify entire dict to avoid contamination)
            return ""
        # Handle list/tuple: smart extraction with priority
        if isinstance(x, (list, tuple)):
            # Priority 1: Try to find a dict with "modified_caption" key
            for v in x:
                if isinstance(v, dict):
                    extracted = _to_str(v)
                    if extracted:  # Only return if we found something
                        return extracted
            # Priority 2: Try to find first string element (assume it's the caption)
            # This handles cases like ["caption text", {...}] or ["caption text"]
            for v in x:
                if isinstance(v, str):
                    return v
            # Last resort: return empty string (don't join all elements to avoid contamination)
            return ""
        return str(x)

    # Use truncate=True to automatically truncate captions that exceed context length
    cap_tokens = tokenize([_to_str(x) for x in modified_captions], truncate=True).to(device, non_blocking=True)
    cap_features = model.encode_text(cap_tokens)
    cap_features = cap_features / cap_features.norm(dim=-1, keepdim=True)

    # ============================================================
    # 🔒 Logit Scale 检查与限制
    # ============================================================
    # CLIP标准做法：clamp logit_scale 到最大值 100 (log(100) ≈ 4.605)
    # 如果logit_scale太大，会导致模型过度自信，loss异常低
    logit_scale_raw = model.logit_scale.exp()
    # Clamp to max 100 (standard CLIP practice)
    logit_scale = torch.clamp(logit_scale_raw, max=100.0).mean()
    # ============================================================

    # gather across GPUs to enlarge negatives
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gathered_q = [torch.zeros_like(query_features) for _ in range(world_size)]
        gathered_c = [torch.zeros_like(cap_features) for _ in range(world_size)]
        dist.all_gather(gathered_q, query_features)
        dist.all_gather(gathered_c, cap_features)

        all_q = torch.cat([query_features] + gathered_q[:rank] + gathered_q[rank + 1 :])
        all_c = torch.cat([cap_features] + gathered_c[:rank] + gathered_c[rank + 1 :])
    else:
        all_q, all_c = query_features, cap_features

    logits_per_query = logit_scale * (all_q @ all_c.t())
    logits_per_caption = logits_per_query.t() 

    # 生成标签 (对角线为正样本)
    targets = torch.arange(all_q.size(0), device=all_q.device, dtype=torch.long)
    
    # 计算双向 Loss
    loss_q2c = loss_fn(logits_per_query, targets) # Query 找 Caption
    loss_c2q = loss_fn(logits_per_caption, targets) # Caption 找 Query

    local_batch_size = query_features.size(0)
    local_targets = torch.arange(local_batch_size, device=all_q.device, dtype=torch.long)
    local_q2c = F.cross_entropy(logits_per_query[:local_batch_size], local_targets, reduction="none")
    local_c2q = F.cross_entropy(logits_per_caption[:local_batch_size], local_targets, reduction="none")
    per_sample_retrieval_loss = 0.5 * (local_q2c + local_c2q)
    
    loss = (loss_q2c + loss_c2q) / 2
    # --- 修改结束 ---
    
    stats = {
        "loss_q2c": float(loss_q2c.detach().item()),
        "loss_c2q": float(loss_c2q.detach().item()),
        "retrieval_hardness_mean": float(per_sample_retrieval_loss.detach().mean().item()),
        "retrieval_hardness_max": float(per_sample_retrieval_loss.detach().max().item()),
    }
    aux = {
        "per_sample_retrieval_loss": per_sample_retrieval_loss.detach(),
        "retrieval_local_logits": logits_per_query[:local_batch_size].detach(),
    }
    return loss, stats, aux


def _select_batch_items(items, indices):
    if indices is None:
        return list(items)
    return [items[i] for i in indices]


def _zero_geo_loss(text_model):
    first_param = next(text_model.parameters(), None)
    if first_param is None:
        return torch.tensor(0.0)
    return first_param.sum() * 0.0


def select_geo_subset(
    src_captions,
    modified_captions,
    forward_instructions,
    reverse_instructions,
    per_sample_retrieval_loss,
    args,
):
    mode = str(getattr(args, "geo_sampling_mode", "all")).lower()
    topk = int(getattr(args, "geo_topk", 0))
    batch_size = len(modified_captions)

    if batch_size == 0:
        return [], {
            "geo_candidate_ratio": 0.0,
            "geo_candidate_count": 0.0,
            "geo_selected_ratio": 0.0,
            "geo_selected_count": 0.0,
            "geo_selected_hardness_mean": 0.0,
            "geo_sampling_is_hard": 0.0,
            "geo_sampling_is_random": 0.0,
        }

    device = per_sample_retrieval_loss.device if per_sample_retrieval_loss is not None else "cpu"
    has_src = torch.tensor([bool(_to_text(x).strip()) for x in src_captions], device=device, dtype=torch.bool)
    has_tgt = torch.tensor([bool(_to_text(x).strip()) for x in modified_captions], device=device, dtype=torch.bool)
    has_fwd = torch.tensor([bool(_to_text(x).strip()) for x in forward_instructions], device=device, dtype=torch.bool)
    has_rev = torch.tensor([bool(_to_text(x).strip()) for x in reverse_instructions], device=device, dtype=torch.bool)
    candidate_mask = has_src & has_tgt & has_fwd & has_rev
    candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()

    if candidate_indices.numel() == 0:
        return [], {
            "geo_candidate_ratio": 0.0,
            "geo_candidate_count": 0.0,
            "geo_selected_ratio": 0.0,
            "geo_selected_count": 0.0,
            "geo_selected_hardness_mean": 0.0,
            "geo_sampling_is_hard": 1.0 if mode == "hard" else 0.0,
            "geo_sampling_is_random": 1.0 if mode == "random" else 0.0,
        }

    if mode == "all" or topk <= 0 or candidate_indices.numel() <= topk:
        selected_indices = candidate_indices
    elif mode == "hard" and per_sample_retrieval_loss is not None:
        candidate_scores = per_sample_retrieval_loss[candidate_indices]
        _, order = torch.topk(candidate_scores, k=topk, largest=True, sorted=False)
        selected_indices = candidate_indices[order]
    elif mode == "random":
        perm = torch.randperm(candidate_indices.numel(), device=candidate_indices.device)
        selected_indices = candidate_indices[perm[:topk]]
    else:
        selected_indices = candidate_indices

    selected_indices = selected_indices.sort().values
    selected_hardness_mean = 0.0
    if per_sample_retrieval_loss is not None and selected_indices.numel() > 0:
        selected_hardness_mean = float(per_sample_retrieval_loss[selected_indices].mean().detach().item())

    stats = {
        "geo_candidate_ratio": float(candidate_indices.numel()) / float(batch_size),
        "geo_candidate_count": float(candidate_indices.numel()),
        "geo_selected_ratio": float(selected_indices.numel()) / float(batch_size),
        "geo_selected_count": float(selected_indices.numel()),
        "geo_selected_hardness_mean": selected_hardness_mean,
        "geo_sampling_is_hard": 1.0 if mode == "hard" else 0.0,
        "geo_sampling_is_random": 1.0 if mode == "random" else 0.0,
    }
    return selected_indices.detach().cpu().tolist(), stats


def get_loss_geo_text_branch(
    text_model,
    src_captions,
    modified_captions,
    forward_instructions,
    reverse_instructions,
    args,
):
    device = next(text_model.parameters()).device

    csrc = [_to_text(x) for x in src_captions]
    ctgt = [_to_text(x) for x in modified_captions]
    fwd = [_to_text(x) for x in forward_instructions] if forward_instructions else [""] * len(ctgt)
    rev = [_to_text(x) for x in reverse_instructions] if reverse_instructions else [""] * len(ctgt)

    has_src = torch.tensor([bool(x.strip()) for x in csrc], device=device, dtype=torch.bool)
    has_tgt = torch.tensor([bool(x.strip()) for x in ctgt], device=device, dtype=torch.bool)
    has_fwd = torch.tensor([bool(x.strip()) for x in fwd], device=device, dtype=torch.bool)
    has_rev = torch.tensor([bool(x.strip()) for x in rev], device=device, dtype=torch.bool)
    valid_mask = has_src & has_tgt & has_fwd & has_rev

    embed_norm_eps = float(getattr(args, "geo_embed_norm_eps", 1e-6))
    delta_norm_eps = float(getattr(args, "geo_delta_norm_eps", 1e-4))
    delta_min_norm = float(getattr(args, "geo_delta_min_norm", 1e-3))

    csrc_tokens = tokenize(csrc, truncate=True).to(device, non_blocking=True)
    ctgt_tokens = tokenize(ctgt, truncate=True).to(device, non_blocking=True)
    fwd_tokens = tokenize(fwd, truncate=True).to(device, non_blocking=True)
    rev_tokens = tokenize(rev, truncate=True).to(device, non_blocking=True)

    z_src = _safe_l2_normalize(text_model.encode_text(csrc_tokens), dim=-1, eps=embed_norm_eps)
    z_tgt = _safe_l2_normalize(text_model.encode_text(ctgt_tokens), dim=-1, eps=embed_norm_eps)
    z_fwd = _safe_l2_normalize(text_model.encode_text(fwd_tokens), dim=-1, eps=embed_norm_eps)
    z_rev = _safe_l2_normalize(text_model.encode_text(rev_tokens), dim=-1, eps=embed_norm_eps)

    delta_raw = z_tgt - z_src
    delta_raw_norm = delta_raw.norm(dim=-1)
    delta_valid = delta_raw_norm > delta_min_norm
    valid_mask = valid_mask & delta_valid

    delta_fwd = delta_raw / delta_raw_norm.unsqueeze(-1).clamp_min(delta_norm_eps)
    delta_rev = -delta_fwd

    fwd_align = (z_fwd * delta_fwd).sum(dim=-1)
    rev_align = (z_rev * delta_rev).sum(dim=-1)
    fwd_rev_cos = (z_fwd * z_rev).sum(dim=-1)

    valid_count = int(valid_mask.sum().item())
    reverse_weight = float(getattr(args, "geo_reverse_weight", 0.25))
    reverse_margin = float(getattr(args, "geo_reverse_margin", 0.0))
    zero_loss_weight = float(getattr(args, "geo_zero_loss_weight", 0.0))

    if valid_count > 0:
        valid_fwd_align = fwd_align[valid_mask]
        valid_rev_align = rev_align[valid_mask]
        valid_fwd_rev_cos = fwd_rev_cos[valid_mask]
        valid_zero_residual = torch.norm((z_fwd + z_rev)[valid_mask], dim=-1)

        loss_fwd = (1.0 - valid_fwd_align).mean()
        loss_rev = (1.0 - valid_rev_align).mean()
        loss_reverse = F.relu(valid_fwd_rev_cos + reverse_margin).mean()
        loss_zero = valid_zero_residual.mean()
        geom_loss = loss_fwd + loss_rev + (reverse_weight * loss_reverse) + (zero_loss_weight * loss_zero)

        mean_fwd_align = float(valid_fwd_align.mean().detach().item())
        mean_rev_align = float(valid_rev_align.mean().detach().item())
        mean_fwd_rev_cos = float(valid_fwd_rev_cos.mean().detach().item())
        mean_zero_residual = float(valid_zero_residual.mean().detach().item())
    else:
        geom_loss = (z_src.sum() + z_tgt.sum() + z_fwd.sum() + z_rev.sum()) * 0.0
        loss_fwd = geom_loss.detach()
        loss_rev = geom_loss.detach()
        loss_reverse = geom_loss.detach()
        loss_zero = geom_loss.detach()
        mean_fwd_align = 0.0
        mean_rev_align = 0.0
        mean_fwd_rev_cos = 0.0
        mean_zero_residual = 0.0

    valid_ratio = float(valid_mask.float().mean().detach().item()) if len(ctgt) > 0 else 0.0
    missing_src_ratio = float((~has_src).float().mean().detach().item()) if len(ctgt) > 0 else 0.0
    small_delta_ratio = float((~delta_valid).float().mean().detach().item()) if len(ctgt) > 0 else 0.0
    delta_norm_mean = float(delta_raw_norm.mean().detach().item()) if len(ctgt) > 0 else 0.0

    valid_geo_logits_unit = None
    if valid_count > 0:
        valid_geo_logits_unit = (z_fwd[valid_mask] @ z_tgt[valid_mask].t()).detach()

    stats = {
        "loss_fwd": float(loss_fwd.detach().item()),
        "loss_rev": float(loss_rev.detach().item()),
        "loss_reverse_consistency": float(loss_reverse.detach().item()),
        "loss_zero_regularizer": float(loss_zero.detach().item()),
        "loss_geom_total": float(geom_loss.detach().item()),
        "z_fwd_align": mean_fwd_align,
        "z_rev_align": mean_rev_align,
        "z_fwd_rev_cos": mean_fwd_rev_cos,
        "z_fwd_rev_zero_norm": mean_zero_residual,
        "geo_valid_ratio": valid_ratio,
        "geo_valid_count": valid_count,
        "geo_missing_src_ratio": missing_src_ratio,
        "geo_small_delta_ratio": small_delta_ratio,
        "geo_delta_norm_mean": delta_norm_mean,
    }
    aux = {
        "geo_logits_unit": valid_geo_logits_unit,
    }
    return geom_loss, stats, aux


def _normalized_lora_name(name):
    if name.startswith("module."):
        name = name[len("module."):]
    return name


_TEXT_BLOCK_RE = re.compile(r"(?:^|\.)transformer\.resblocks\.(\d+)\.")
_TEXT_CPROJ_RE = re.compile(r"(?:^|\.)transformer\.resblocks\.(\d+)\.mlp\.c_proj\.(A|B)$")


def _extract_text_block_index(name):
    match = _TEXT_BLOCK_RE.search(_normalized_lora_name(name))
    if not match:
        return None
    return int(match.group(1))


def _extract_text_cproj_effective_grads(model):
    named_modules = {_normalized_lora_name(name): module for name, module in model.named_modules()}
    entries = {}
    for name, param in model.named_parameters():
        norm_name = _normalized_lora_name(name)
        match = _TEXT_CPROJ_RE.search(norm_name)
        if not match:
            continue
        block_idx = int(match.group(1))
        part = match.group(2)
        prefix = norm_name.rsplit(".", 1)[0]
        entry = entries.setdefault(block_idx, {"prefix": prefix, "scaling": 1.0})
        entry[part] = param.detach()
        if param.grad is not None:
            entry[f"d{part}"] = param.grad.detach()
        module = named_modules.get(prefix)
        if module is not None and hasattr(module, "scaling"):
            entry["scaling"] = float(module.scaling)

    block_vectors = {}
    for block_idx, entry in entries.items():
        if not all(key in entry for key in ("A", "B", "dA", "dB")):
            continue
        a = entry["A"].float()
        b = entry["B"].float()
        da = entry["dA"].float()
        db = entry["dB"].float()
        delta_grad = (db @ a) + (b @ da)
        scaling = float(entry.get("scaling", 1.0))
        if scaling != 1.0:
            delta_grad = delta_grad * scaling
        block_vectors[block_idx] = delta_grad.reshape(-1).detach().cpu()
    return block_vectors


def _cosine_or_none(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return None
    a = vec_a.float()
    b = vec_b.float()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float(torch.dot(a, b).div(denom).item())


def _project_block_vector(retrieval_vec, edit_vec):
    if retrieval_vec is None or edit_vec is None:
        return None
    proj = edit_vec.clone().float()
    retr = retrieval_vec.float()
    dot_val = torch.dot(proj, retr)
    if float(dot_val.item()) < 0.0:
        retr_norm_sq = torch.dot(retr, retr)
        if float(retr_norm_sq.item()) > 1e-12:
            proj = proj - (dot_val / retr_norm_sq) * retr
    return proj


def _subset_images(images, indices):
    if indices is None:
        return images
    if len(indices) == 0:
        return images[:0]
    index_tensor = torch.as_tensor(indices, device=images.device, dtype=torch.long)
    return images.index_select(0, index_tensor)


def _probe_modified_caption_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("modified_caption", "caption", "text", "value", "en", "description", "content"):
            if key in x and isinstance(x[key], str):
                return x[key]
        return ""
    if isinstance(x, (list, tuple)):
        for value in x:
            extracted = _probe_modified_caption_text(value)
            if extracted:
                return extracted
        return ""
    return str(x)


def get_loss_lcom_textprobe_cc3m(model, img2text, ref_images, instructions, modified_captions, loss_fn, args):
    device = ref_images.device
    with torch.no_grad():
        image_features = model.encode_image(ref_images)
        token_features = img2text(image_features).detach()

    placeholder = getattr(args, "prompt_placeholder", "*")
    placeholder_token_id = int(tokenize([placeholder])[0][1].item())
    prompts = [_build_retrieval_prompt(inst, placeholder, args) for inst in instructions]
    prompt_tokens = tokenize(prompts, truncate=True).to(device, non_blocking=True)
    query_features = model.encode_text_img_vis(prompt_tokens, token_features, split_ind=placeholder_token_id)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)

    cap_tokens = tokenize([_probe_modified_caption_text(x) for x in modified_captions], truncate=True).to(device, non_blocking=True)
    cap_features = model.encode_text(cap_tokens)
    cap_features = cap_features / cap_features.norm(dim=-1, keepdim=True)
    logit_scale = torch.clamp(model.logit_scale.exp(), max=100.0).mean()

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered_q = [torch.zeros_like(query_features) for _ in range(world_size)]
        gathered_c = [torch.zeros_like(cap_features) for _ in range(world_size)]
        dist.all_gather(gathered_q, query_features)
        dist.all_gather(gathered_c, cap_features)
        all_q = torch.cat([query_features] + gathered_q[:rank] + gathered_q[rank + 1 :])
        all_c = torch.cat([cap_features] + gathered_c[:rank] + gathered_c[rank + 1 :])
    else:
        all_q, all_c = query_features, cap_features

    logits_per_query = logit_scale * (all_q @ all_c.t())
    logits_per_caption = logits_per_query.t()
    targets = torch.arange(all_q.size(0), device=all_q.device, dtype=torch.long)
    loss_q2c = loss_fn(logits_per_query, targets)
    loss_c2q = loss_fn(logits_per_caption, targets)
    return (loss_q2c + loss_c2q) / 2.0


def _mean_softmax_entropy(logits):
    if logits is None or logits.numel() == 0:
        return None
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return float(entropy.detach().item())


def _compute_eteg(retrieval_aux, geo_aux, logit_scale):
    retrieval_logits = retrieval_aux.get("retrieval_local_logits") if retrieval_aux else None
    geo_logits_unit = geo_aux.get("geo_logits_unit") if geo_aux else None
    retrieval_entropy = _mean_softmax_entropy(retrieval_logits)
    geo_entropy = None
    if geo_logits_unit is not None and geo_logits_unit.numel() > 0:
        geo_entropy = _mean_softmax_entropy(float(logit_scale) * geo_logits_unit.float())
    eteg = None
    if retrieval_entropy is not None and geo_entropy is not None:
        eteg = float(geo_entropy - retrieval_entropy)
    return {
        "retrieval_entropy": retrieval_entropy,
        "geo_entropy": geo_entropy,
        "eteg": eteg,
    }


def _init_conflict_probe_state(args):
    if not bool(getattr(args, "conflict_probe", False)):
        return None
    history_path = getattr(args, "conflict_probe_output", None)
    if not history_path:
        history_path = os.path.join(args.logs, args.name, "gradient_conflict_history.jsonl")
    layerwise_path = os.path.join(os.path.dirname(history_path), "gradient_conflict_layerwise.tsv")
    summary_path = os.path.join(os.path.dirname(history_path), "gradient_conflict_summary.json")
    plot_path = getattr(args, "conflict_probe_plot", None)
    Path(os.path.dirname(history_path)).mkdir(parents=True, exist_ok=True)
    return {
        "history_path": history_path,
        "layerwise_path": layerwise_path,
        "summary_path": summary_path,
        "plot_path": plot_path,
        "count": 0,
        "ret_sum": [None] * 12,
        "edit_sum": [None] * 12,
        "edit_proj_sum": [None] * 12,
        "base_a_sum": [None] * 12,
        "base_b_sum": [None] * 12,
        "ret_count": [0] * 12,
        "edit_count": [0] * 12,
        "edit_proj_count": [0] * 12,
        "base_a_count": [0] * 12,
        "base_b_count": [0] * 12,
    }


def _append_conflict_probe_record(probe_state, record):
    history_record = {k: v for k, v in record.items() if not k.endswith("_vectors")}
    with open(probe_state["history_path"], "a", encoding="utf-8") as f:
        f.write(json.dumps(history_record, ensure_ascii=False) + "\n")
    probe_state["count"] += 1
    vector_groups = (
        ("ret_sum", "ret_count", record.get("ret_vectors", {})),
        ("edit_sum", "edit_count", record.get("edit_vectors", {})),
        ("edit_proj_sum", "edit_proj_count", record.get("edit_proj_vectors", {})),
        ("base_a_sum", "base_a_count", record.get("base_a_vectors", {})),
        ("base_b_sum", "base_b_count", record.get("base_b_vectors", {})),
    )
    for sum_key, count_key, vectors in vector_groups:
        for idx, vec in vectors.items():
            if vec is None:
                continue
            vec_cpu = vec.detach().cpu().float()
            if probe_state[sum_key][idx] is None:
                probe_state[sum_key][idx] = vec_cpu.clone()
            else:
                probe_state[sum_key][idx].add_(vec_cpu)
            probe_state[count_key][idx] += 1


def _finalize_conflict_probe(probe_state):
    if probe_state is None or probe_state["count"] <= 0:
        return
    s_base_vals = []
    s_cross_before_vals = []
    s_cross_after_vals = []
    gc_before_vals = []
    gc_after_vals = []
    with open(probe_state["layerwise_path"], "w", encoding="utf-8") as f:
        f.write("block\ts_base\ts_cross_before\ts_cross_after\tgc_before\tgc_after\n")
        for idx in range(12):
            ret_avg = None
            edit_avg = None
            edit_proj_avg = None
            base_a_avg = None
            base_b_avg = None
            if probe_state["ret_sum"][idx] is not None and probe_state["ret_count"][idx] > 0:
                ret_avg = probe_state["ret_sum"][idx] / float(probe_state["ret_count"][idx])
            if probe_state["edit_sum"][idx] is not None and probe_state["edit_count"][idx] > 0:
                edit_avg = probe_state["edit_sum"][idx] / float(probe_state["edit_count"][idx])
            if probe_state["edit_proj_sum"][idx] is not None and probe_state["edit_proj_count"][idx] > 0:
                edit_proj_avg = probe_state["edit_proj_sum"][idx] / float(probe_state["edit_proj_count"][idx])
            if probe_state["base_a_sum"][idx] is not None and probe_state["base_a_count"][idx] > 0:
                base_a_avg = probe_state["base_a_sum"][idx] / float(probe_state["base_a_count"][idx])
            if probe_state["base_b_sum"][idx] is not None and probe_state["base_b_count"][idx] > 0:
                base_b_avg = probe_state["base_b_sum"][idx] / float(probe_state["base_b_count"][idx])

            s_base = _cosine_or_none(base_a_avg, base_b_avg)
            s_cross_before = _cosine_or_none(ret_avg, edit_avg)
            s_cross_after = _cosine_or_none(ret_avg, edit_proj_avg)
            gc_before = None if (s_base is None or s_cross_before is None) else float(s_base - s_cross_before)
            gc_after = None if (s_base is None or s_cross_after is None) else float(s_base - s_cross_after)

            s_base_vals.append(s_base)
            s_cross_before_vals.append(s_cross_before)
            s_cross_after_vals.append(s_cross_after)
            gc_before_vals.append(gc_before)
            gc_after_vals.append(gc_after)

            def _fmt(val):
                return "" if val is None else f"{val:.6f}"

            f.write(
                f"{idx}\t{_fmt(s_base)}\t{_fmt(s_cross_before)}\t{_fmt(s_cross_after)}\t{_fmt(gc_before)}\t{_fmt(gc_after)}\n"
            )

    summary = {
        "num_probe_steps": int(probe_state["count"]),
        "s_base_mean": [None if v is None else float(v) for v in s_base_vals],
        "s_cross_before_mean": [None if v is None else float(v) for v in s_cross_before_vals],
        "s_cross_after_mean": [None if v is None else float(v) for v in s_cross_after_vals],
        "gc_before_mean": [None if v is None else float(v) for v in gc_before_vals],
        "gc_after_mean": [None if v is None else float(v) for v in gc_after_vals],
        "gc_before_macro_mean": float(sum(v for v in gc_before_vals if v is not None) / max(1, sum(1 for v in gc_before_vals if v is not None))),
        "gc_after_macro_mean": float(sum(v for v in gc_after_vals if v is not None) / max(1, sum(1 for v in gc_after_vals if v is not None))),
    }
    with open(probe_state["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not probe_state.get("plot_path"):
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = list(range(12))
        y_before = [float("nan") if v is None else v for v in gc_before_vals]
        y_after = [float("nan") if v is None else v for v in gc_after_vals]
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, y_before, marker="o", label="Before Projection", linewidth=2)
        plt.plot(x, y_after, marker="o", label="After Projection", linewidth=2)
        plt.xticks(x)
        plt.xlabel("Text ResBlock")
        plt.ylabel("Gradient Conflict")
        plt.title("Layer-wise FFN c_proj Gradient Conflict")
        plt.legend()
        plt.tight_layout()
        plt.savefig(probe_state["plot_path"], dpi=200)
        plt.close()
    except Exception as exc:
        logging.warning(f"Failed to render gradient conflict plot: {exc}")


def _run_unix_conflict_probe(
    model,
    img2text,
    ref_images,
    instructions,
    modified_captions,
    src_captions,
    reverse_instructions,
    loss_fn,
    args,
    step,
):
    if ref_images is None or ref_images.size(0) < 2:
        return None

    probe_seed = int(getattr(args, "seed", 0)) + 1000003 + int(step)
    with torch.random.fork_rng(devices=_branch_cuda_devices(args), enabled=True):
        torch.manual_seed(probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(probe_seed)

        model.zero_grad(set_to_none=True)
        img2text.zero_grad(set_to_none=True)

        ret_loss = get_loss_lcom_textprobe_cc3m(
            model,
            img2text,
            ref_images,
            instructions,
            modified_captions,
            loss_fn,
            args,
        )
        ret_loss.backward()
        ret_vectors = _extract_text_cproj_effective_grads(model)

        model.zero_grad(set_to_none=True)
        img2text.zero_grad(set_to_none=True)

        edit_loss, _, _ = get_loss_geo_text_branch(
            model,
            src_captions,
            modified_captions,
            instructions,
            reverse_instructions,
            args,
        )
        edit_loss.backward()
        edit_vectors = _extract_text_cproj_effective_grads(model)

        model.zero_grad(set_to_none=True)
        img2text.zero_grad(set_to_none=True)

        batch_size = ref_images.size(0)
        perm = torch.randperm(batch_size, device=ref_images.device).detach().cpu().tolist()
        split = max(1, batch_size // 2)
        idx_a = perm[:split]
        idx_b = perm[split:]
        if len(idx_b) == 0:
            idx_b = idx_a[-1:]
            idx_a = idx_a[:-1]
            if len(idx_a) == 0:
                idx_a = idx_b

        def _joint_loss(indices):
            subset_images = _subset_images(ref_images, indices)
            subset_instructions = _select_batch_items(instructions, indices)
            subset_modified = _select_batch_items(modified_captions, indices)
            subset_src = _select_batch_items(src_captions, indices)
            subset_reverse = _select_batch_items(reverse_instructions, indices)
            retrieval_loss = get_loss_lcom_textprobe_cc3m(
                model,
                img2text,
                subset_images,
                subset_instructions,
                subset_modified,
                loss_fn,
                args,
            )
            geo_loss, _, _ = get_loss_geo_text_branch(
                model,
                subset_src,
                subset_modified,
                subset_instructions,
                subset_reverse,
                args,
            )
            return retrieval_loss + (float(getattr(args, "geo_weight", 1.0)) * geo_loss)

        joint_loss_a = _joint_loss(idx_a)
        joint_loss_a.backward()
        base_a_vectors = _extract_text_cproj_effective_grads(model)

        model.zero_grad(set_to_none=True)
        img2text.zero_grad(set_to_none=True)

        joint_loss_b = _joint_loss(idx_b)
        joint_loss_b.backward()
        base_b_vectors = _extract_text_cproj_effective_grads(model)

        model.zero_grad(set_to_none=True)
        img2text.zero_grad(set_to_none=True)

    edit_proj_vectors = {}
    for block_idx in range(12):
        retr_vec = ret_vectors.get(block_idx)
        edit_vec = edit_vectors.get(block_idx)
        if retr_vec is None or edit_vec is None:
            continue
        edit_proj_vectors[block_idx] = _project_block_vector(retr_vec, edit_vec)

    s_base = []
    s_cross_before = []
    s_cross_after = []
    gc_before = []
    gc_after = []
    for block_idx in range(12):
        s_base_val = _cosine_or_none(base_a_vectors.get(block_idx), base_b_vectors.get(block_idx))
        s_cross_before_val = _cosine_or_none(ret_vectors.get(block_idx), edit_vectors.get(block_idx))
        s_cross_after_val = _cosine_or_none(ret_vectors.get(block_idx), edit_proj_vectors.get(block_idx))
        gc_before_val = None if (s_base_val is None or s_cross_before_val is None) else float(s_base_val - s_cross_before_val)
        gc_after_val = None if (s_base_val is None or s_cross_after_val is None) else float(s_base_val - s_cross_after_val)
        s_base.append(s_base_val)
        s_cross_before.append(s_cross_before_val)
        s_cross_after.append(s_cross_after_val)
        gc_before.append(gc_before_val)
        gc_after.append(gc_after_val)

    valid_before = [v for v in gc_before if v is not None]
    valid_after = [v for v in gc_after if v is not None]
    return {
        "step": int(step),
        "s_base": s_base,
        "s_cross_before": s_cross_before,
        "s_cross_after": s_cross_after,
        "gc_before": gc_before,
        "gc_after": gc_after,
        "gc_before_mean": float(sum(valid_before) / len(valid_before)) if valid_before else None,
        "gc_after_mean": float(sum(valid_after) / len(valid_after)) if valid_after else None,
        "ret_vectors": ret_vectors,
        "edit_vectors": edit_vectors,
        "edit_proj_vectors": edit_proj_vectors,
        "base_a_vectors": base_a_vectors,
        "base_b_vectors": base_b_vectors,
    }


def project_geo_gradients(retrieval_model, geo_text_model):
    retrieval_grads = {}
    for name, param in retrieval_model.named_parameters():
        norm_name = _normalized_lora_name(name)
        if norm_name.startswith("visual."):
            continue
        if not (norm_name.endswith(".A") or norm_name.endswith(".B")):
            continue
        if param.grad is None:
            continue
        retrieval_grads[norm_name] = param.grad.detach()

    matched = 0
    conflicts = 0
    projected = 0
    negative_dot_sum = 0.0
    for name, param in geo_text_model.named_parameters():
        norm_name = _normalized_lora_name(name)
        if not (norm_name.endswith(".A") or norm_name.endswith(".B")):
            continue
        if param.grad is None:
            continue
        retr_grad = retrieval_grads.get(norm_name)
        if retr_grad is None:
            continue

        geo_grad = param.grad
        matched += 1
        geo_flat = geo_grad.reshape(-1).float()
        retr_flat = retr_grad.reshape(-1).float()
        dot_val = torch.dot(geo_flat, retr_flat)
        if dot_val < 0:
            conflicts += 1
            negative_dot_sum += float(dot_val.detach().item())
            retr_norm_sq = torch.dot(retr_flat, retr_flat)
            if retr_norm_sq > 1e-12:
                scale = (dot_val / retr_norm_sq).to(dtype=geo_grad.dtype)
                geo_grad.sub_(scale * retr_grad)
                projected += 1

    ratio = float(conflicts) / float(matched) if matched > 0 else 0.0
    return {
        "geo_grad_pairs": matched,
        "geo_conflict_count": conflicts,
        "geo_projected_count": projected,
        "geo_conflict_ratio": ratio,
        "geo_negative_dot_sum": negative_dot_sum,
    }


def _capture_named_grads(module, target_names):
    if not target_names:
        return {}
    target_names = set(target_names)
    saved = {}
    for name, param in module.named_parameters():
        if name not in target_names:
            continue
        saved[name] = None if param.grad is None else param.grad.detach().clone()
    return saved


def _restore_named_grads(module, saved_grads):
    if not saved_grads:
        return 0
    restored = 0
    for name, param in module.named_parameters():
        if name not in saved_grads:
            continue
        saved_grad = saved_grads[name]
        if saved_grad is None:
            param.grad = None
        else:
            if param.grad is None:
                param.grad = saved_grad.clone()
            else:
                param.grad.copy_(saved_grad)
        restored += 1
    return restored


def _branch_cuda_devices(args):
    if not torch.cuda.is_available():
        return []
    if getattr(args, "gpu", None) is not None:
        return [int(args.gpu)]
    return [torch.cuda.current_device()]


class _TorchBranchRNGState:
    def __init__(self, seed, cuda_devices):
        self.seed = int(seed)
        self.cuda_devices = list(cuda_devices)
        with torch.random.fork_rng(devices=self.cuda_devices, enabled=True):
            torch.manual_seed(self.seed)
            if self.cuda_devices:
                torch.cuda.manual_seed_all(self.seed)
            self.cpu_state = torch.random.get_rng_state()
            self.cuda_states = {
                device: torch.cuda.get_rng_state(device) for device in self.cuda_devices
            }


@contextmanager
def _geo_rng_context(branch_rng_state, enabled):
    if (not enabled) or (branch_rng_state is None):
        yield
        return

    saved_cpu_state = torch.random.get_rng_state()
    saved_cuda_states = {
        device: torch.cuda.get_rng_state(device) for device in branch_rng_state.cuda_devices
    }
    torch.random.set_rng_state(branch_rng_state.cpu_state)
    for device, state in branch_rng_state.cuda_states.items():
        torch.cuda.set_rng_state(state, device)

    try:
        yield
    finally:
        branch_rng_state.cpu_state = torch.random.get_rng_state()
        branch_rng_state.cuda_states = {
            device: torch.cuda.get_rng_state(device) for device in branch_rng_state.cuda_devices
        }
        torch.random.set_rng_state(saved_cpu_state)
        for device, state in saved_cuda_states.items():
            torch.cuda.set_rng_state(state, device)


def _run_periodic_multidataset_eval(model, img2text, args, data):
    results = {"fashioniq": {}, "genecis": {}}

    fashion_eval_loaders = data.get("fashion_eval_loaders", {})
    for cloth, loaders in fashion_eval_loaders.items():
        metrics = evaluate_fashion(
            model,
            img2text,
            args,
            loaders["query"],
            loaders["target"],
        )
        results["fashioniq"][cloth] = metrics or {}

    genecis_eval_loaders = data.get("genecis_eval_loaders", {})
    original_genecis_task = getattr(args, "genecis_task", None)
    try:
        for task, loader in genecis_eval_loaders.items():
            args.genecis_task = task
            metrics = evaluate_genecis(model, img2text, args, loader)
            results["genecis"][task] = {
                f"R@{rank}": float(value) for rank, value in (metrics or {}).items()
            }
    finally:
        args.genecis_task = original_genecis_task

    return results


def train(
    model,
    img2text,
    data,
    epoch,
    optimizer,
    retrieval_scaler,
    scheduler,
    args,
    tb_writer=None,
    geo_text_model=None,
    geo_optimizer=None,
    geo_scheduler=None,
    geo_scaler=None,
    retrieval_model_ema=None,
    img2text_ema=None,
    geo_text_ema=None,
):
    os.environ["WDS_EPOCH"] = str(epoch)

    # IMPORTANT: train mode so LoRA can learn (and dropout consistent)
    model.train()
    img2text.train()
    if geo_text_model is not None:
        geo_text_model.train()

    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    loss_ce = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_ce = loss_ce.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    if args.precision == "amp" and retrieval_scaler is None:
        raise ValueError("AMP training requires a retrieval GradScaler.")
    if args.precision == "amp" and geo_optimizer is not None and geo_scaler is None:
        raise ValueError("AMP geo training requires a dedicated geo GradScaler.")

    geo_rng_state = None
    if geo_text_model is not None and geo_optimizer is not None:
        rank_offset = int(getattr(args, "rank", 0) or 0)
        geo_rng_seed = int(getattr(args, "geo_seed", getattr(args, "seed", 0))) + rank_offset
        geo_rng_state = _TorchBranchRNGState(geo_rng_seed, _branch_cuda_devices(args))
        if is_master(args):
            logging.info(
                f"Geo branch uses an independent torch RNG stream with seed={geo_rng_seed}."
            )

    shared_a_param_names = set(getattr(args, "shared_a_param_names", []) or [])
    shared_a_retrieval_only = bool(
        getattr(args, "shared_a_lora", False) and getattr(args, "shared_a_retrieval_only_update", False)
    )
    if shared_a_retrieval_only and is_master(args):
        logging.info(
            f"Shared-A retrieval-only update active for {len(shared_a_param_names)} text LoRA A tensors."
        )

    ema_pairs = [
        (model, retrieval_model_ema),
        (img2text, img2text_ema),
        (geo_text_model, geo_text_ema),
    ]
    ema_enabled = any(ema is not None for _, ema in ema_pairs)
    ema_eval_enabled = bool(getattr(args, "ema_eval", False) and ema_enabled)
    conflict_probe_state = _init_conflict_probe_state(args) if is_master(args) else None
    if ema_enabled and is_master(args):
        logging.info(f"EMA enabled for training. ema_eval={ema_eval_enabled}")

    # handle streaming dataset (num_batches=None)
    num_batches_per_epoch = getattr(dataloader, "num_batches", None)
    if num_batches_per_epoch is None:
        num_batches_per_epoch = getattr(args, "wds_epoch_steps", 100000)

    accum_steps = max(int(getattr(args, "accum_steps", 1)), 1)
    # We interpret `num_batches_per_epoch` as the number of optimizer updates per epoch when accumulating.
    num_updates_per_epoch = num_batches_per_epoch
    num_microbatches = num_updates_per_epoch * accum_steps

    optimizer.zero_grad(set_to_none=True)
    if geo_optimizer is not None:
        geo_optimizer.zero_grad(set_to_none=True)

    # Create iterator once; re-creating iter(dataloader) repeatedly is very expensive (and can stall).
    dl_it = iter(dataloader)

    end = time.time()
    last_log_time = end
    micro_idx = 0
    update_idx = 0
    while True:
        if micro_idx >= num_microbatches:
            break

        try:
            batch = next(dl_it)
        except StopIteration:
            dl_it = iter(dataloader)
            batch = next(dl_it)

        data_time = time.time() - end
        m = model.module if args.distributed or args.dp else model
        i2t = img2text.module if (args.distributed or args.dp) else img2text
        g = geo_text_model.module if hasattr(geo_text_model, "module") else geo_text_model
        loss_stats = None

        # ---- CC3M CIR dict batch -> Lcom-only ----
        if isinstance(batch, dict):
            images = batch["ref_img"]
            instructions = batch["instruction"]
            modified_captions = batch["modified_caption"]
            src_captions = batch.get("src_caption", [""] * len(modified_captions))
            reverse_instructions = batch.get("reverse_instruction", [""] * len(modified_captions))
            geo_forward_instructions = list(instructions)

            # ============================================================
            # 🔒 Whole-instruction Dropout: drop the entire instruction
            # ============================================================
            instruction_dropout_prob = getattr(args, "instruction_dropout_prob", 0.0)
            num_dropped = 0
            num_eligible = 0
            if instruction_dropout_prob > 0.0 and model.training:
                dropped_instructions = []
                for inst in instructions:
                    inst_str = str(inst)
                    if inst_str.strip():
                        num_eligible += 1
                        if random.random() < instruction_dropout_prob:
                            num_dropped += 1
                            dropped_instructions.append("")
                            continue
                    dropped_instructions.append(inst_str)
                instructions = dropped_instructions
            # ============================================================

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            geo_weight = float(getattr(args, "geo_weight", 0.0))
            geo_enabled = (g is not None) and (geo_weight > 0.0)

            if args.precision == "amp":
                with autocast():
                    retrieval_loss, retrieval_stats, retrieval_aux = get_loss_lcom_cc3m(
                        m,
                        i2t,
                        images,
                        instructions,
                        modified_captions,
                        loss_ce,
                        args,
                    )
                    total_loss = retrieval_loss / float(accum_steps)
                retrieval_scaler.scale(total_loss).backward()
                shared_a_saved_grads = None
                if geo_enabled and shared_a_retrieval_only and shared_a_param_names:
                    shared_a_saved_grads = _capture_named_grads(m, shared_a_param_names)
                if geo_enabled:
                    geo_indices, geo_sampling_stats = select_geo_subset(
                        src_captions,
                        modified_captions,
                        geo_forward_instructions,
                        reverse_instructions,
                        retrieval_aux.get("per_sample_retrieval_loss"),
                        args,
                    )
                    geo_src = _select_batch_items(src_captions, geo_indices)
                    geo_tgt = _select_batch_items(modified_captions, geo_indices)
                    geo_fwd = _select_batch_items(geo_forward_instructions, geo_indices)
                    geo_rev = _select_batch_items(reverse_instructions, geo_indices)
                    with _geo_rng_context(geo_rng_state, enabled=True):
                        with autocast():
                            if geo_tgt:
                                geo_loss, geo_stats, geo_aux = get_loss_geo_text_branch(
                                    g,
                                    geo_src,
                                    geo_tgt,
                                    geo_fwd,
                                    geo_rev,
                                    args,
                                )
                            else:
                                geo_loss = _zero_geo_loss(g)
                                geo_stats = {
                                    "loss_fwd": 0.0,
                                    "loss_rev": 0.0,
                                    "loss_reverse_consistency": 0.0,
                                    "loss_zero_regularizer": 0.0,
                                    "loss_geom_total": 0.0,
                                    "z_fwd_align": 0.0,
                                    "z_rev_align": 0.0,
                                    "z_fwd_rev_cos": 0.0,
                                    "z_fwd_rev_zero_norm": 0.0,
                                    "geo_valid_ratio": 0.0,
                                    "geo_valid_count": 0,
                                    "geo_missing_src_ratio": 0.0,
                                    "geo_small_delta_ratio": 0.0,
                                    "geo_delta_norm_mean": 0.0,
                                }
                                geo_aux = {"geo_logits_unit": None}
                            weighted_geo_loss = geo_loss * geo_weight
                        if torch.isfinite(weighted_geo_loss.detach()).all():
                            geo_scaler.scale(weighted_geo_loss / float(accum_steps)).backward()
                            if shared_a_saved_grads is not None:
                                restored_count = _restore_named_grads(m, shared_a_saved_grads)
                                if loss_stats is None:
                                    loss_stats = {}
                                loss_stats["shared_a_restored"] = float(restored_count)
                        else:
                            if geo_stats is None:
                                geo_stats = {}
                            geo_stats["geo_skipped_nonfinite"] = 1.0
                            weighted_geo_loss = None
                else:
                    geo_loss, geo_stats, geo_aux, weighted_geo_loss = None, None, None, None
            else:
                retrieval_loss, retrieval_stats, retrieval_aux = get_loss_lcom_cc3m(
                    m,
                    i2t,
                    images,
                    instructions,
                    modified_captions,
                    loss_ce,
                    args,
                )
                (retrieval_loss / float(accum_steps)).backward()
                shared_a_saved_grads = None
                if geo_enabled and shared_a_retrieval_only and shared_a_param_names:
                    shared_a_saved_grads = _capture_named_grads(m, shared_a_param_names)
                if geo_enabled:
                    geo_indices, geo_sampling_stats = select_geo_subset(
                        src_captions,
                        modified_captions,
                        geo_forward_instructions,
                        reverse_instructions,
                        retrieval_aux.get("per_sample_retrieval_loss"),
                        args,
                    )
                    geo_src = _select_batch_items(src_captions, geo_indices)
                    geo_tgt = _select_batch_items(modified_captions, geo_indices)
                    geo_fwd = _select_batch_items(geo_forward_instructions, geo_indices)
                    geo_rev = _select_batch_items(reverse_instructions, geo_indices)
                    with _geo_rng_context(geo_rng_state, enabled=True):
                        if geo_tgt:
                            geo_loss, geo_stats, geo_aux = get_loss_geo_text_branch(
                                g,
                                geo_src,
                                geo_tgt,
                                geo_fwd,
                                geo_rev,
                                args,
                            )
                        else:
                            geo_loss = _zero_geo_loss(g)
                            geo_stats = {
                                "loss_fwd": 0.0,
                                "loss_rev": 0.0,
                                "loss_reverse_consistency": 0.0,
                                "loss_zero_regularizer": 0.0,
                                "loss_geom_total": 0.0,
                                "z_fwd_align": 0.0,
                                "z_rev_align": 0.0,
                                "z_fwd_rev_cos": 0.0,
                                "z_fwd_rev_zero_norm": 0.0,
                                "geo_valid_ratio": 0.0,
                                "geo_valid_count": 0,
                                "geo_missing_src_ratio": 0.0,
                                "geo_small_delta_ratio": 0.0,
                                "geo_delta_norm_mean": 0.0,
                            }
                            geo_aux = {"geo_logits_unit": None}
                        weighted_geo_loss = geo_loss * geo_weight
                        (weighted_geo_loss / float(accum_steps)).backward()
                        if shared_a_saved_grads is not None:
                            restored_count = _restore_named_grads(m, shared_a_saved_grads)
                            if loss_stats is None:
                                loss_stats = {}
                            loss_stats["shared_a_restored"] = float(restored_count)
                else:
                    geo_loss, geo_stats, geo_aux, weighted_geo_loss = None, None, None, None

            loss_stats = {
                "loss_retrieval": float(retrieval_loss.detach().item()),
            }
            if retrieval_stats:
                loss_stats.update(retrieval_stats)
            if geo_enabled:
                if geo_sampling_stats:
                    loss_stats.update(geo_sampling_stats)
                if weighted_geo_loss is not None:
                    loss_stats["loss_geo_weighted"] = float(weighted_geo_loss.detach().item())
                    loss_stats["loss_parallel_total"] = float(
                        retrieval_loss.detach().item() + weighted_geo_loss.detach().item()
                    )
                if geo_stats:
                    loss_stats.update(geo_stats)
            total_loss = retrieval_loss / float(accum_steps)
            
            # Add instruction dropout statistics to loss_stats
            if instruction_dropout_prob > 0.0 and model.training:
                if num_eligible > 0:
                    loss_stats["inst_dropout_rate"] = num_dropped / num_eligible
                else:
                    loss_stats["inst_dropout_rate"] = 0.0

            # logging bookkeeping
            batch_size = images.size(0)

        # ---- legacy tuple batch -> old training path (kept) ----
        else:
            images, texts = batch[0], batch[1]
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if args.precision == "amp":
                with autocast():
                    total_loss = get_loss_img2text(
                        m,
                        i2t,
                        images,
                        loss_ce,
                        loss_ce,
                        args,
                    )
                    total_loss = total_loss / float(accum_steps)
                retrieval_scaler.scale(total_loss).backward()
            else:
                total_loss = get_loss_img2text(
                    m,
                    i2t,
                    images,
                    loss_ce,
                    loss_ce,
                    args,
                )
                (total_loss / float(accum_steps)).backward()

            batch_size = images.size(0)

        micro_idx += 1

        # optimizer step on accumulation boundary
        if (micro_idx % accum_steps) == 0:
            step = num_updates_per_epoch * epoch + update_idx
            geo_step_ready = geo_optimizer is not None and g is not None
            projection_stats = {}
            probe_metrics = None
            if args.precision == "amp":
                retrieval_scaler.unscale_(optimizer)
                if geo_step_ready:
                    geo_scaler.unscale_(geo_optimizer)
            if (
                geo_step_ready
                and getattr(args, "geo_conflict_projection", False)
            ):
                projection_stats = project_geo_gradients(m, g)
                if loss_stats is None:
                    loss_stats = {}
                loss_stats.update(projection_stats)
            scheduler(step)
            if geo_scheduler is not None and geo_step_ready:
                geo_scheduler(step)

            if args.precision == "amp":
                retrieval_scaler.step(optimizer)
                retrieval_scaler.update()
                if geo_step_ready:
                    geo_scaler.step(geo_optimizer)
                    geo_scaler.update()
            else:
                optimizer.step()
                if geo_step_ready:
                    geo_optimizer.step()
            if retrieval_model_ema is not None:
                retrieval_model_ema.update(model)
            if img2text_ema is not None:
                img2text_ema.update(img2text)
            if geo_step_ready and geo_text_ema is not None:
                geo_text_ema.update(geo_text_model)
            optimizer.zero_grad(set_to_none=True)
            if geo_optimizer is not None:
                geo_optimizer.zero_grad(set_to_none=True)

            conflict_probe_every = int(getattr(args, "conflict_probe_every", 0))
            conflict_probe_start = int(getattr(args, "conflict_probe_start", 0))
            conflict_probe_end = int(getattr(args, "conflict_probe_end", 0))
            probe_enabled = bool(getattr(args, "conflict_probe", False))
            should_probe = (
                probe_enabled
                and conflict_probe_every > 0
                and (step + 1) >= max(conflict_probe_start, 1)
                and (conflict_probe_end <= 0 or (step + 1) <= conflict_probe_end)
                and (((step + 1) - max(conflict_probe_start, 1)) % conflict_probe_every == 0)
            )
            if should_probe and isinstance(batch, dict):
                probe_metrics = _run_unix_conflict_probe(
                    m,
                    i2t,
                    images,
                    instructions,
                    modified_captions,
                    src_captions,
                    reverse_instructions,
                    loss_ce,
                    args,
                    step + 1,
                )
                if probe_metrics is not None:
                    if is_master(args) and conflict_probe_state is not None:
                        record = {
                            "epoch": int(epoch),
                            "step": int(step + 1),
                            "s_base": probe_metrics["s_base"],
                            "s_cross_before": probe_metrics["s_cross_before"],
                            "s_cross_after": probe_metrics["s_cross_after"],
                            "gc_before": probe_metrics["gc_before"],
                            "gc_after": probe_metrics["gc_after"],
                            "ret_vectors": probe_metrics["ret_vectors"],
                            "edit_vectors": probe_metrics["edit_vectors"],
                            "edit_proj_vectors": probe_metrics["edit_proj_vectors"],
                            "base_a_vectors": probe_metrics["base_a_vectors"],
                            "base_b_vectors": probe_metrics["base_b_vectors"],
                        }
                        _append_conflict_probe_record(conflict_probe_state, record)
                    if loss_stats is None:
                        loss_stats = {}
                    if probe_metrics.get("gc_before_mean") is not None:
                        loss_stats["gc_before_mean"] = float(probe_metrics["gc_before_mean"])
                    if probe_metrics.get("gc_after_mean") is not None:
                        loss_stats["gc_after_mean"] = float(probe_metrics["gc_after_mean"])

            batch_time = time.time() - end
            end = time.time()

            log_interval = int(getattr(args, "log_interval", 20))
            log_every_s = float(getattr(args, "log_every_s", 60.0))
            now = time.time()
            should_log = (update_idx % max(log_interval, 1) == 0) or ((now - last_log_time) >= log_every_s)

            if is_master(args) and should_log:
                # samples processed so far in this epoch (microbatches)
                num_samples = micro_idx * batch_size * args.world_size
                samples_per_epoch = getattr(dataloader, "num_samples", None)
                if samples_per_epoch is None:
                    samples_per_epoch = num_microbatches * args.batch_size * args.world_size
                percent_complete = 100.0 * micro_idx / num_microbatches

                total_grad_norm = 0.0
                if args.debug:
                    params = list(m.parameters()) + list(i2t.parameters())
                    if g is not None:
                        params += list(g.parameters())
                    for p in params:
                        if p.grad is None or (not p.requires_grad):
                            continue
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1.0 / 2.0)

                logit_scale_exp = m.logit_scale.exp().item()
                log_msg = (
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                    f"Loss: {(total_loss.detach().item() * float(accum_steps)):.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f} (exp={logit_scale_exp:.3f})"
                )
                if geo_optimizer is not None:
                    log_msg += f"\tgeo_LR: {geo_optimizer.param_groups[0]['lr']:5f}"
                if loss_stats is not None:
                    if "loss_retrieval" in loss_stats:
                        log_msg += f"\tloss_retrieval: {loss_stats['loss_retrieval']:.4f}"
                    if "loss_q2c" in loss_stats:
                        log_msg += f"\tloss_q2c: {loss_stats['loss_q2c']:.4f}\tloss_c2q: {loss_stats['loss_c2q']:.4f}"
                    if "loss_geo_weighted" in loss_stats:
                        log_msg += f"\tloss_geo_weighted: {loss_stats['loss_geo_weighted']:.4f}"
                    if "loss_geom_total" in loss_stats:
                        log_msg += f"\tloss_geom_total: {loss_stats['loss_geom_total']:.4f}"
                    if "loss_reverse_consistency" in loss_stats:
                        log_msg += f"\tloss_reverse: {loss_stats['loss_reverse_consistency']:.4f}"
                    if "z_fwd_align" in loss_stats:
                        log_msg += f"\tz_fwd_align: {loss_stats['z_fwd_align']:.4f}"
                    if "z_rev_align" in loss_stats:
                        log_msg += f"\tz_rev_align: {loss_stats['z_rev_align']:.4f}"
                    if "z_fwd_rev_cos" in loss_stats:
                        log_msg += f"\tz_fwd_rev_cos: {loss_stats['z_fwd_rev_cos']:.4f}"
                    if "geo_conflict_ratio" in loss_stats:
                        log_msg += f"\tgeo_conflict: {loss_stats['geo_conflict_ratio']:.2%}"
                    if "gc_before_mean" in loss_stats:
                        log_msg += f"\tgc_before: {loss_stats['gc_before_mean']:.4f}"
                    if "gc_after_mean" in loss_stats:
                        log_msg += f"\tgc_after: {loss_stats['gc_after_mean']:.4f}"
                    if "geo_valid_ratio" in loss_stats:
                        log_msg += f"\tgeo_valid: {loss_stats['geo_valid_ratio']:.2%}"
                    if "geo_missing_src_ratio" in loss_stats:
                        log_msg += f"\tgeo_missing_src: {loss_stats['geo_missing_src_ratio']:.2%}"
                    if "geo_small_delta_ratio" in loss_stats:
                        log_msg += f"\tgeo_small_delta: {loss_stats['geo_small_delta_ratio']:.2%}"
                    if "geo_selected_count" in loss_stats:
                        log_msg += f"\tgeo_selected: {loss_stats['geo_selected_count']:.0f}"
                    if "geo_selected_hardness_mean" in loss_stats:
                        log_msg += f"\tgeo_hardness: {loss_stats['geo_selected_hardness_mean']:.4f}"
                    if "inst_dropout_rate" in loss_stats:
                        log_msg += f"\tinst_dropout: {loss_stats['inst_dropout_rate']:.2%}"
                if args.precision == "amp":
                    log_msg += f"\tamp_ret: {retrieval_scaler.get_scale():.1f}"
                    if geo_optimizer is not None and geo_scaler is not None:
                        log_msg += f"\tamp_geo: {geo_scaler.get_scale():.1f}"
                if args.debug and total_grad_norm > 0:
                    log_msg += f"\tgrad_norm: {total_grad_norm:.4f}"
                logging.info(log_msg)
                last_log_time = now

                timestep = step
                log_data = {
                    "loss": float(total_loss.detach().item() * float(accum_steps)),
                    "data_time": float(data_time),
                    "batch_time": float(batch_time),
                    "scale": float(m.logit_scale.data.item()),
                    "scale_exp": float(m.logit_scale.exp().item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                if loss_stats is not None:
                    for k, v in loss_stats.items():
                        if isinstance(v, (int, float)):
                            log_data[k] = float(v)
                if args.debug and total_grad_norm > 0:
                    log_data["grad_norm"] = float(total_grad_norm)
                if args.precision == "amp":
                    log_data["amp_scale_retrieval"] = float(retrieval_scaler.get_scale())
                    if geo_optimizer is not None and geo_scaler is not None:
                        log_data["amp_scale_geo"] = float(geo_scaler.get_scale())

                for name, val in log_data.items():
                    name = "train/" + name
                    if tb_writer is not None:
                        tb_writer.add_scalar(name, val, timestep)
                    if args.wandb:
                        wandb.log({name: val, "step": timestep})

            eval_every = int(getattr(args, "cirr_val_eval_every", 0))
            cirr_eval_enabled = (
                eval_every > 0
                and ("cirr_val_query_loader" in data)
                and ("cirr_val_target_loader" in data)
            )
            global_step = step + 1
            if cirr_eval_enabled and (global_step % eval_every == 0):
                if args.distributed and dist.is_initialized():
                    dist.barrier()

                if is_master(args):
                    eval_start = time.time()
                    with (use_ema_weights(ema_pairs) if ema_eval_enabled else nullcontext()):
                        cirr_metrics = evaluate_cirr(
                            model,
                            img2text,
                            args,
                            data["cirr_val_query_loader"],
                            data["cirr_val_target_loader"],
                        )
                    eval_time = time.time() - eval_start

                    if cirr_metrics:
                        flat_metrics = {}
                        for feat_name, feat_metrics in cirr_metrics.items():
                            if not isinstance(feat_metrics, dict):
                                continue
                            for metric_name, metric_val in feat_metrics.items():
                                try:
                                    flat_metrics[f"{feat_name}/{metric_name}"] = float(metric_val)
                                except Exception:
                                    continue

                        summary = ", ".join([f"{k}={v:.4f}" for k, v in sorted(flat_metrics.items())])
                        logging.info(
                            f"[CIRR-VAL] epoch={epoch} step={global_step} time={eval_time:.2f}s {summary}"
                        )

                        for metric_name, metric_val in flat_metrics.items():
                            tag = f"cirr_val/{metric_name}"
                            if tb_writer is not None:
                                tb_writer.add_scalar(tag, metric_val, global_step)
                            if args.wandb:
                                wandb.log({tag: metric_val, "step": global_step})

                        log_file = getattr(args, "cirr_val_eval_log_path", None)
                        if log_file:
                            record = {
                                "epoch": int(epoch),
                                "step": int(global_step),
                                "eval_time_sec": float(eval_time),
                                "metrics": cirr_metrics,
                                "used_ema": bool(ema_eval_enabled),
                            }
                            with open(log_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                model.train()
                img2text.train()
                if geo_text_model is not None:
                    geo_text_model.train()

                if args.distributed and dist.is_initialized():
                    dist.barrier()

            multidataset_eval_every = int(getattr(args, "multidataset_eval_every", 0))
            multidataset_eval_enabled = (
                multidataset_eval_every > 0
                and ("fashion_eval_loaders" in data)
                and ("genecis_eval_loaders" in data)
            )
            if multidataset_eval_enabled and (global_step % multidataset_eval_every == 0):
                if args.distributed and dist.is_initialized():
                    dist.barrier()

                if is_master(args):
                    eval_start = time.time()
                    with (use_ema_weights(ema_pairs) if ema_eval_enabled else nullcontext()):
                        multidataset_metrics = _run_periodic_multidataset_eval(model, img2text, args, data)
                    eval_time = time.time() - eval_start

                    summary_parts = []
                    for cloth in ["dress", "shirt", "toptee"]:
                        composed_metrics = (
                            multidataset_metrics.get("fashioniq", {})
                            .get(cloth, {})
                            .get("composed", {})
                        )
                        if "R@10" in composed_metrics:
                            summary_parts.append(
                                f"fashion/{cloth}/composed/R@10={float(composed_metrics['R@10']):.2f}"
                            )
                    for task in ["focus_attribute", "change_attribute", "focus_object", "change_object"]:
                        genecis_metrics = multidataset_metrics.get("genecis", {}).get(task, {})
                        if "R@1" in genecis_metrics:
                            summary_parts.append(
                                f"genecis/{task}/R@1={float(genecis_metrics['R@1']):.2f}"
                            )
                    summary = ", ".join(summary_parts) if summary_parts else "no_metrics"
                    logging.info(
                        f"[MULTIDATASET-EVAL] epoch={epoch} step={global_step} time={eval_time:.2f}s {summary}"
                    )

                    for cloth, feat_metrics in multidataset_metrics.get("fashioniq", {}).items():
                        for feat_name, metric_dict in feat_metrics.items():
                            for metric_name, metric_val in metric_dict.items():
                                tag = f"multieval/fashioniq/{cloth}/{feat_name}/{metric_name}"
                                if tb_writer is not None:
                                    tb_writer.add_scalar(tag, float(metric_val), global_step)
                                if args.wandb:
                                    wandb.log({tag: float(metric_val), "step": global_step})

                    for task, metric_dict in multidataset_metrics.get("genecis", {}).items():
                        for metric_name, metric_val in metric_dict.items():
                            tag = f"multieval/genecis/{task}/{metric_name}"
                            if tb_writer is not None:
                                tb_writer.add_scalar(tag, float(metric_val), global_step)
                            if args.wandb:
                                wandb.log({tag: float(metric_val), "step": global_step})

                    log_file = getattr(args, "multidataset_eval_log_path", None)
                    if log_file:
                        record = {
                            "epoch": int(epoch),
                            "step": int(global_step),
                            "eval_time_sec": float(eval_time),
                            "metrics": multidataset_metrics,
                            "used_ema": bool(ema_eval_enabled),
                        }
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                model.train()
                img2text.train()
                if geo_text_model is not None:
                    geo_text_model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if args.distributed and dist.is_initialized():
                    dist.barrier()

            def _save_step_checkpoint(save_step: int):
                def _build_checkpoint_payload():
                    return {
                        "epoch": epoch,
                        "step": save_step,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "state_dict_geo_text": geo_text_model.state_dict() if geo_text_model is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "optimizer_retrieval": optimizer.state_dict(),
                        "optimizer_geo": geo_optimizer.state_dict() if geo_optimizer is not None else None,
                        "ema_retrieval_model": retrieval_model_ema.state_dict() if retrieval_model_ema is not None else None,
                        "ema_img2text": img2text_ema.state_dict() if img2text_ema is not None else None,
                        "ema_geo_text": geo_text_ema.state_dict() if geo_text_ema is not None else None,
                    }

                save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}_step_{save_step}.pt")
                torch.save(_build_checkpoint_payload(), save_path)
                logging.info(f"Saved checkpoint at step {save_step}: {save_path}")

                if geo_text_model is not None:
                    geo_lora_state_dict = {}
                    for n, p in geo_text_model.named_parameters():
                        if n.endswith(".A") or n.endswith(".B"):
                            geo_lora_state_dict[n] = p.data.clone()
                    if geo_lora_state_dict:
                        geo_lora_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}_step_{save_step}_geo_lora.pt")
                        torch.save(geo_lora_state_dict, geo_lora_path)
                        logging.info(
                            f"Saved geo LoRA-only weights ({len(geo_lora_state_dict)} params) to {geo_lora_path}"
                        )
                if getattr(args, "ema_save_checkpoints", False) and ema_enabled:
                    ema_save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}_step_{save_step}_ema.pt")
                    with use_ema_weights(ema_pairs):
                        ema_payload = _build_checkpoint_payload()
                        ema_payload["optimizer"] = None
                        ema_payload["optimizer_retrieval"] = None
                        ema_payload["optimizer_geo"] = None
                        ema_payload["ema_checkpoint"] = True
                        torch.save(ema_payload, ema_save_path)
                        logging.info(f"Saved EMA checkpoint at step {save_step}: {ema_save_path}")
                        if geo_text_model is not None and geo_text_ema is not None:
                            geo_lora_ema_state_dict = {}
                            for n, p in geo_text_model.named_parameters():
                                if n.endswith(".A") or n.endswith(".B"):
                                    geo_lora_ema_state_dict[n] = p.data.clone()
                            if geo_lora_ema_state_dict:
                                geo_lora_ema_path = os.path.join(
                                    args.checkpoint_path,
                                    f"epoch_{epoch}_step_{save_step}_geo_lora_ema.pt",
                                )
                                torch.save(geo_lora_ema_state_dict, geo_lora_ema_path)
                                logging.info(
                                    f"Saved geo EMA LoRA-only weights ({len(geo_lora_ema_state_dict)} params) to {geo_lora_ema_path}"
                                )

            if is_master(args):
                save_step_start = int(getattr(args, "save_step_start", 0))
                save_step_end = int(getattr(args, "save_step_end", 0))
                save_step_interval = int(getattr(args, "save_step_interval", 0))
                if (
                    save_step_start > 0
                    and save_step_end >= save_step_start
                    and save_step_interval > 0
                    and global_step >= save_step_start
                    and global_step <= save_step_end
                    and ((global_step - save_step_start) % save_step_interval == 0)
                ):
                    _save_step_checkpoint(global_step)

            update_idx += 1

    if is_master(args):
        _finalize_conflict_probe(conflict_probe_state)
