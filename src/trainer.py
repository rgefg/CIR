# trainer.py
# Copyright 2022 Google LLC
# Licensed under the Apache License, Version 2.0

import os
import time
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import wandb
import json

from torch.cuda.amp import autocast
from third_party.open_clip.clip import tokenize
from eval_utils import evaluate_cirr
from utils import is_master


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


def _short_preview_text(text, limit=160):
    text = re.sub(r"\s+", " ", _to_text(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_reasoning_latent(z_src, z_tgt, z_fwd):
    """
    Encode the state transition A -> B into a compact latent that keeps both
    the target delta and the instruction-aligned direction.
    """
    delta_fwd = F.normalize(z_tgt - z_src, dim=-1)
    z_reason = F.normalize(delta_fwd + z_fwd, dim=-1)
    return z_reason, delta_fwd


def get_reason_llm_target_loss(reasoning_modules, z_reason, src_texts, instructions, target_texts, args):
    projector = reasoning_modules["projector"]
    llm = reasoning_modules["llm"]
    tokenizer_llm = reasoning_modules["tokenizer"]

    prompt_template = getattr(
        args,
        "reason_llm_template",
        "Source caption: {src}\nInstruction: {instruction}\nTarget caption:",
    )
    prompts = [
        prompt_template.format(src=_to_text(src), instruction=_to_text(inst))
        for src, inst in zip(src_texts, instructions)
    ]
    targets = [_to_text(tgt).strip() for tgt in target_texts]

    soft_prompts = projector(z_reason)
    embed_layer = llm.get_input_embeddings()
    embed_device = embed_layer.weight.device
    embed_dtype = embed_layer.weight.dtype
    soft_prompts = soft_prompts.to(device=embed_device, dtype=embed_dtype)

    prompt_ids = tokenizer_llm(
        prompts,
        add_special_tokens=True,
        truncation=True,
        max_length=int(getattr(args, "reason_llm_max_prompt_len", 128)),
        return_attention_mask=False,
    )["input_ids"]
    target_ids = tokenizer_llm(
        targets,
        add_special_tokens=False,
        truncation=True,
        max_length=int(getattr(args, "reason_llm_max_target_len", 48)),
        return_attention_mask=False,
    )["input_ids"]

    eos_id = tokenizer_llm.eos_token_id
    if eos_id is None:
        eos_id = tokenizer_llm.pad_token_id
    if eos_id is None:
        raise ValueError("Frozen LLM tokenizer needs either eos_token_id or pad_token_id.")

    combined_embeds = []
    combined_labels = []
    combined_masks = []
    effective_target_ids = []
    for idx, (prompt_seq, target_seq) in enumerate(zip(prompt_ids, target_ids)):
        if len(target_seq) == 0:
            target_seq = [eos_id]
        effective_target_ids.append(target_seq)
        prompt_tensor = torch.tensor(prompt_seq, dtype=torch.long, device=embed_device)
        target_tensor = torch.tensor(target_seq, dtype=torch.long, device=embed_device)
        prompt_embeds = embed_layer(prompt_tensor)
        target_embeds = embed_layer(target_tensor)
        soft = soft_prompts[idx]
        merged = torch.cat([soft, prompt_embeds, target_embeds], dim=0)
        labels = torch.full((merged.size(0),), -100, dtype=torch.long, device=embed_device)
        labels[soft.size(0) + prompt_tensor.size(0):] = target_tensor
        mask = torch.ones((merged.size(0),), dtype=torch.long, device=embed_device)
        combined_embeds.append(merged)
        combined_labels.append(labels)
        combined_masks.append(mask)

    max_len = max(x.size(0) for x in combined_embeds)
    hidden = combined_embeds[0].size(-1)
    batch_size = len(combined_embeds)
    batch_embeds = torch.zeros((batch_size, max_len, hidden), dtype=embed_dtype, device=embed_device)
    batch_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=embed_device)
    batch_masks = torch.zeros((batch_size, max_len), dtype=torch.long, device=embed_device)

    for idx, (embeds, labels, mask) in enumerate(zip(combined_embeds, combined_labels, combined_masks)):
        cur_len = embeds.size(0)
        batch_embeds[idx, :cur_len] = embeds
        batch_labels[idx, :cur_len] = labels
        batch_masks[idx, :cur_len] = mask

    outputs = llm(
        inputs_embeds=batch_embeds,
        attention_mask=batch_masks,
        labels=batch_labels,
        use_cache=False,
    )
    stats = {
        "loss_reason_llm": float(outputs.loss.detach().item()),
        "z_reason_norm": float(torch.norm(z_reason, dim=-1).mean().detach().item()),
        "soft_prompt_norm": float(torch.norm(soft_prompts, dim=-1).mean().detach().item()),
    }
    if int(getattr(args, "reason_llm_preview_every", 0)) > 0 and batch_size > 0:
        preview_idx = 0
        preview_target_ids = effective_target_ids[preview_idx]
        preview_start = soft_prompts.size(1) + len(prompt_ids[preview_idx])
        pred_start = max(preview_start - 1, 0)
        pred_end = pred_start + len(preview_target_ids)
        pred_slice = outputs.logits[preview_idx, pred_start:pred_end]
        pred_ids = pred_slice.argmax(dim=-1).detach().cpu().tolist() if pred_slice.numel() > 0 else []
        stats["reason_preview_src"] = _short_preview_text(src_texts[preview_idx])
        stats["reason_preview_instruction"] = _short_preview_text(instructions[preview_idx])
        stats["reason_preview_target"] = _short_preview_text(targets[preview_idx])
        stats["reason_preview_pred"] = _short_preview_text(
            tokenizer_llm.decode(pred_ids, skip_special_tokens=True)
        )
    return outputs.loss, stats


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
    prompts = []
    for inst in instructions:
        inst_str = str(inst).strip()
        if not inst_str:
            # Empty instruction: use generic prompt or just placeholder
            prompts.append(f"a photo of {placeholder}")
        else:
            prompts.append(f"a photo of {placeholder} and {inst_str}")
    
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

    logit_scale_raw = model.logit_scale.exp()
    # Clamp to max 100 (standard CLIP practice)
    logit_scale = torch.clamp(logit_scale_raw, max=100.0).mean()

    # gather across GPUs to enlarge negatives (supports heterogeneous batch sizes)
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        local_bs = torch.tensor([query_features.size(0)], device=device, dtype=torch.long)
        all_bs = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_bs, local_bs)
        max_bs = max(int(s.item()) for s in all_bs)

        embed_dim = query_features.size(1)
        q_padded = F.pad(query_features, (0, 0, 0, max_bs - query_features.size(0)))
        c_padded = F.pad(cap_features, (0, 0, 0, max_bs - cap_features.size(0)))

        gathered_q = [torch.zeros(max_bs, embed_dim, device=device, dtype=query_features.dtype) for _ in range(world_size)]
        gathered_c = [torch.zeros(max_bs, embed_dim, device=device, dtype=cap_features.dtype) for _ in range(world_size)]
        dist.all_gather(gathered_q, q_padded)
        dist.all_gather(gathered_c, c_padded)

        real_q = [gathered_q[i][:int(all_bs[i].item())] for i in range(world_size)]
        real_c = [gathered_c[i][:int(all_bs[i].item())] for i in range(world_size)]

        all_q = torch.cat([query_features] + [real_q[i] for i in range(world_size) if i != rank])
        all_c = torch.cat([cap_features] + [real_c[i] for i in range(world_size) if i != rank])
    else:
        all_q, all_c = query_features, cap_features

    logits_per_query = logit_scale * (all_q @ all_c.t())
    logits_per_caption = logits_per_query.t()

    targets = torch.arange(all_q.size(0), device=all_q.device, dtype=torch.long)

    loss_q2c = loss_fn(logits_per_query, targets)
    loss_c2q = loss_fn(logits_per_caption, targets)

    loss = (loss_q2c + loss_c2q) / 2

    stats = {
        "loss_q2c": float(loss_q2c.detach().item()),
        "loss_c2q": float(loss_c2q.detach().item()),
    }
    return loss, stats


def get_loss_joint_cc3m(
    model,
    img2text,
    ref_images,
    src_captions,
    modified_captions,
    forward_instructions,
    reverse_instructions,
    loss_fn,
    args,
    reasoning_modules=None,
):
    """
    Joint training: retrieval (Lcom) as primary loss + frozen-LLM reasoning as auxiliary.

    Retrieval path (same as get_loss_lcom_cc3m):
      query = compose(image, instruction) via placeholder injection
      target = encode_text(modified_caption)
      loss_retrieval = contrastive(query, target)

    Reasoning path:
      z_src = Text(C_src), z_tgt = Text(C_tgt), z_fwd = Text(T_fwd)
      z_reason = normalize(delta_fwd + z_fwd)
      z_reason -> projector -> frozen LLM -> CE loss on target caption
    """
    device = ref_images.device
    world_size = dist.get_world_size() if (args.distributed and dist.is_initialized()) else 1
    local_batch_size = int(ref_images.shape[0])
    actual_per_microbatch = int(
        getattr(args, "actual_samples_per_microbatch", local_batch_size * world_size)
    )
    # DDP averages gradients across ranks, so unequal local batch sizes need
    # an explicit rescale to recover a global sample-weighted mean loss.
    ddp_batch_scale = (world_size * local_batch_size) / float(max(actual_per_microbatch, 1))

    # ================================================================
    # Part 1: Retrieval loss (Lcom) — primary objective
    # ================================================================
    retrieval_loss, retrieval_stats = get_loss_lcom_cc3m(
        model, img2text, ref_images, forward_instructions, modified_captions, loss_fn, args,
    )

    retrieval_loss_scaled = retrieval_loss * ddp_batch_scale
    global_retrieval_loss_scaled = retrieval_loss_scaled.detach()
    if args.distributed and dist.is_initialized():
        global_retrieval_loss_scaled = global_retrieval_loss_scaled.clone()
        dist.all_reduce(global_retrieval_loss_scaled, op=dist.ReduceOp.SUM)
        global_retrieval_loss_scaled /= float(world_size)
    loss = retrieval_loss_scaled
    stats = {
        "loss_retrieval": float(retrieval_loss.detach().item()),
        "loss_retrieval_scaled": float(retrieval_loss_scaled.detach().item()),
        "loss_retrieval_global_scaled": float(global_retrieval_loss_scaled.detach().item()),
        "loss_q2c": retrieval_stats.get("loss_q2c", 0.0),
        "loss_c2q": retrieval_stats.get("loss_c2q", 0.0),
        "ddp_batch_scale": ddp_batch_scale,
        "local_batch_size": local_batch_size,
        "actual_samples_per_microbatch": actual_per_microbatch,
    }

    # ================================================================
    # Part 2: Frozen-LLM reasoning loss — auxiliary objective
    #   LLM lives only on rank 0.  Rank 0 computes LLM loss on its
    #   local batch only. Use the same batch-aware DDP correction so
    #   its gradient stays proportional to the number of local samples.
    # ================================================================
    reason_llm_divisor = float(getattr(args, "reason_llm_weight", 0.0))
    if reasoning_modules is not None and reason_llm_divisor > 0.0:
        csrc = [_to_text(x) for x in src_captions]
        ctgt = [_to_text(x) for x in modified_captions]
        fwd = [_to_text(x) for x in forward_instructions] if forward_instructions else [""] * len(ctgt)

        csrc_tokens = tokenize(csrc, truncate=True).to(device, non_blocking=True)
        ctgt_tokens = tokenize(ctgt, truncate=True).to(device, non_blocking=True)
        fwd_tokens = tokenize(fwd, truncate=True).to(device, non_blocking=True)

        z_src = F.normalize(model.encode_text(csrc_tokens), dim=-1)
        z_tgt = F.normalize(model.encode_text(ctgt_tokens), dim=-1)
        z_fwd = F.normalize(model.encode_text(fwd_tokens), dim=-1)
        z_reason, delta_fwd = build_reasoning_latent(z_src, z_tgt, z_fwd)

        reason_loss, reason_stats = get_reason_llm_target_loss(
            reasoning_modules, z_reason, csrc, fwd, ctgt, args,
        )
        reason_loss_scaled = reason_loss * ddp_batch_scale
        # LLM loss exists on rank 0 only, so multiply by world_size here to
        # compensate DDP's gradient averaging and realize a true global 1/n ratio.
        target_reason_contrib = (
            global_retrieval_loss_scaled * (float(world_size) / reason_llm_divisor)
        )
        reason_llm_coeff = target_reason_contrib / reason_loss_scaled.detach().clamp_min(1e-12)
        reason_contrib = reason_llm_coeff * reason_loss_scaled
        loss = loss + reason_contrib

        stats["loss_reason_llm"] = float(reason_loss.detach().item())
        stats["loss_reason_llm_scaled"] = float(reason_loss_scaled.detach().item())
        stats["reason_llm_divisor"] = reason_llm_divisor
        stats["reason_llm_coeff"] = float(reason_llm_coeff.detach().item())
        stats["reason_llm_target_contrib"] = float(target_reason_contrib.detach().item())
        stats["reason_llm_effective_contrib"] = float(reason_contrib.detach().item())
        stats["z_reason_align"] = float((z_reason * delta_fwd).sum(dim=-1).mean().detach().item())
        stats.update({k: v for k, v in reason_stats.items() if k not in stats})

    # ================================================================
    # (Deprecated) Geometric losses — ablation showed no additional gain
    #   after merge; kept commented for reference.
    # ----------------------------------------------------------------
    # rev = [_to_text(x) for x in reverse_instructions]
    # rev_tokens = tokenize(rev, truncate=True).to(device, non_blocking=True)
    # z_rev = F.normalize(model.encode_text(rev_tokens), dim=-1)
    # delta_rev = F.normalize(z_src - z_tgt, dim=-1)
    # loss_fwd = 1.0 - (z_fwd * delta_fwd).sum(dim=-1).mean()
    # loss_rev = 1.0 - (z_rev * delta_rev).sum(dim=-1).mean()
    # loss_zero = torch.norm(z_fwd + z_rev, dim=-1).mean()
    # geom_loss = loss_fwd + loss_rev + loss_zero
    # ================================================================

    return loss, stats


def train(
    model,
    img2text,
    data,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
    tb_writer=None,
    save_at_percentages=None,
    reasoning_modules=None,
):
    os.environ["WDS_EPOCH"] = str(epoch)

    # IMPORTANT: train mode so LoRA can learn (and dropout consistent)
    model.train()
    img2text.train()
    if reasoning_modules is not None:
        reasoning_modules["projector"].train()
        reasoning_modules["llm"].eval()

    # Reset saved percentages and clamp counter for this epoch
    if save_at_percentages is not None:
        train._saved_percentages = set()
    train.clamp_hit_count = 0

    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    loss_ce = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_ce = loss_ce.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    # handle streaming dataset (num_batches=None)
    num_batches_per_epoch = getattr(dataloader, "num_batches", None)
    if num_batches_per_epoch is None:
        num_batches_per_epoch = getattr(args, "wds_epoch_steps", 100000)

    accum_steps = max(int(getattr(args, "accum_steps", 1)), 1)
    # We interpret `num_batches_per_epoch` as the number of optimizer updates per epoch when accumulating.
    num_updates_per_epoch = num_batches_per_epoch
    num_microbatches = num_updates_per_epoch * accum_steps

    optimizer.zero_grad(set_to_none=True)

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
        m = model.module if hasattr(model, "module") else model
        i2t = img2text.module if hasattr(img2text, "module") else img2text
        loss_stats = None

        # ---- CC3M CIR dict batch -> Lcom-only ----
        if isinstance(batch, dict):
            images = batch["ref_img"]
            modified_captions = batch["modified_caption"]
            instructions = batch.get("instruction", None)
            reverse_instructions = batch.get("reverse_instruction", None)
            src_captions = batch.get("src_caption", None)

            # ============================================================
            # 🔒 Instruction Token Dropout: drop high-risk tokens only
            # ============================================================
            instruction_dropout_prob = getattr(args, "instruction_dropout_prob", 0.0)
            num_dropped = 0
            num_eligible = 0
            if (not getattr(args, "train_inference_lora", False)) and instruction_dropout_prob > 0.0 and model.training:
                risk_set = _load_or_build_risk_tokens(args)
                dropped_instructions = []
                for inst in instructions:
                    inst_str = str(inst)
                    tokens = inst_str.split()
                    kept = []
                    for tok in tokens:
                        if _is_high_risk_token(tok, risk_set):
                            num_eligible += 1
                            if random.random() < instruction_dropout_prob:
                                num_dropped += 1
                                continue
                        kept.append(tok)
                    dropped_instructions.append(" ".join(kept).strip())
                instructions = dropped_instructions
            # ============================================================

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if getattr(args, "train_inference_lora", False):
                if src_captions is None:
                    src_captions = [""] * len(modified_captions)
                if not hasattr(train, "_checked_joint_fields"):
                    def _is_empty(x):
                        return (x is None) or (str(x).strip() == "")
                    empty_src = sum(1 for x in src_captions if _is_empty(x))
                    bsz = max(1, len(modified_captions))
                    if is_master(args):
                        logging.info(
                            f"[joint] empty src_caption: {empty_src}/{bsz} ({100.0*empty_src/bsz:.1f}%)"
                        )
                    train._checked_joint_fields = True
                if args.precision == "amp":
                    with autocast():
                        total_loss, loss_stats = get_loss_joint_cc3m(
                            m, i2t, images,
                            src_captions, modified_captions,
                            instructions, reverse_instructions,
                            loss_ce, args,
                            reasoning_modules=reasoning_modules,
                        )
                        total_loss = total_loss / float(accum_steps)
                    scaler.scale(total_loss).backward()
                else:
                    total_loss, loss_stats = get_loss_joint_cc3m(
                        m, i2t, images,
                        src_captions, modified_captions,
                        instructions, reverse_instructions,
                        loss_ce, args,
                        reasoning_modules=reasoning_modules,
                    )
                    (total_loss / float(accum_steps)).backward()
            else:
                if args.precision == "amp":
                    with autocast():
                        total_loss, loss_stats = get_loss_lcom_cc3m(
                            m,
                            i2t,
                            images,
                            instructions,
                            modified_captions,
                            loss_ce,
                            args,
                        )
                        total_loss = total_loss / float(accum_steps)
                    scaler.scale(total_loss).backward()
                else:
                    total_loss, loss_stats = get_loss_lcom_cc3m(
                        m,
                        i2t,
                        images,
                        instructions,
                        modified_captions,
                        loss_ce,
                        args,
                    )
                    (total_loss / float(accum_steps)).backward()
            
            # Add instruction dropout statistics to loss_stats
            if loss_stats is None:
                loss_stats = {}
            if (not getattr(args, "train_inference_lora", False)) and instruction_dropout_prob > 0.0 and model.training:
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
                scaler.scale(total_loss).backward()
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
            scheduler(step)

            if args.precision == "amp":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # 🔒 Logit Scale clamp with counter
            logit_scale_clamp_min = getattr(args, "logit_scale_clamp_min", None)
            logit_scale_clamp_max = getattr(args, "logit_scale_clamp_max", None)

            if logit_scale_clamp_min is not None or logit_scale_clamp_max is not None:
                with torch.no_grad():
                    raw = m.logit_scale.data.clone()
                    if logit_scale_clamp_min is not None:
                        min_logit = torch.log(torch.tensor(logit_scale_clamp_min)).to(m.logit_scale.device)
                        m.logit_scale.data = torch.max(m.logit_scale.data, min_logit)
                    if logit_scale_clamp_max is not None:
                        max_logit = torch.log(torch.tensor(logit_scale_clamp_max)).to(m.logit_scale.device)
                        m.logit_scale.data = torch.min(m.logit_scale.data, max_logit)
                    if (m.logit_scale.data != raw).any():
                        if not hasattr(train, 'clamp_hit_count'):
                            train.clamp_hit_count = 0
                        train.clamp_hit_count += 1
                        if is_master(args) and update_idx % 100 == 0:
                            logging.info(f"🔒 CLAMP HIT #{train.clamp_hit_count}: raw={raw.item():.4f} -> clamped={m.logit_scale.data.item():.4f}")

            optimizer.zero_grad(set_to_none=True)

            batch_time = time.time() - end
            end = time.time()

            log_interval = int(getattr(args, "log_interval", 20))
            log_every_s = float(getattr(args, "log_every_s", 60.0))
            now = time.time()
            preview_every = int(getattr(args, "reason_llm_preview_every", 0))
            preview_due = (
                loss_stats is not None
                and preview_every > 0
                and ((step + 1) % preview_every == 0)
                and "reason_preview_pred" in loss_stats
            )
            should_log = (update_idx % max(log_interval, 1) == 0) or ((now - last_log_time) >= log_every_s)
            should_log = should_log or preview_due

            if is_master(args) and should_log:
                samples_per_micro = getattr(args, "actual_samples_per_microbatch",
                                            args.batch_size * args.world_size)
                num_samples = micro_idx * samples_per_micro
                samples_per_epoch = getattr(dataloader, "num_samples", None)
                if samples_per_epoch is None:
                    samples_per_epoch = num_microbatches * samples_per_micro
                percent_complete = 100.0 * micro_idx / num_microbatches

                total_grad_norm = 0.0
                if args.debug:
                    params = list(m.parameters()) + list(i2t.parameters())
                    if reasoning_modules is not None:
                        rproj = reasoning_modules["projector"]
                        params += list(rproj.parameters())
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
                if loss_stats is not None:
                    if "loss_q2c" in loss_stats and "loss_c2q" in loss_stats:
                        log_msg += f"\tloss_q2c: {loss_stats['loss_q2c']:.4f}\tloss_c2q: {loss_stats['loss_c2q']:.4f}"
                    if "loss_retrieval" in loss_stats:
                        log_msg += f"\tloss_retrieval: {loss_stats['loss_retrieval']:.4f}"
                    if "loss_retrieval_scaled" in loss_stats:
                        log_msg += f"\tloss_ret_scaled: {loss_stats['loss_retrieval_scaled']:.4f}"
                    if "loss_retrieval_global_scaled" in loss_stats:
                        log_msg += f"\tloss_ret_global: {loss_stats['loss_retrieval_global_scaled']:.4f}"
                    if "loss_reason_llm" in loss_stats:
                        log_msg += f"\tloss_reason_llm: {loss_stats['loss_reason_llm']:.4f}"
                    if "loss_reason_llm_scaled" in loss_stats:
                        log_msg += f"\tloss_reason_scaled: {loss_stats['loss_reason_llm_scaled']:.4f}"
                    if "reason_llm_divisor" in loss_stats:
                        log_msg += f"\treason_llm_div: {loss_stats['reason_llm_divisor']:.3f}"
                    if "reason_llm_coeff" in loss_stats:
                        log_msg += f"\treason_llm_coeff: {loss_stats['reason_llm_coeff']:.4f}"
                    if "reason_llm_effective_contrib" in loss_stats:
                        log_msg += f"\treason_llm_term: {loss_stats['reason_llm_effective_contrib']:.4f}"
                    if "z_reason_align" in loss_stats:
                        log_msg += f"\tz_reason_align: {loss_stats['z_reason_align']:.4f}"
                    if "inst_dropout_rate" in loss_stats:
                        log_msg += f"\tinst_dropout: {loss_stats['inst_dropout_rate']:.2%}"
                if args.debug and total_grad_norm > 0:
                    log_msg += f"\tgrad_norm: {total_grad_norm:.4f}"
                logging.info(log_msg)
                if preview_due:
                    logging.info(
                        "[reason-preview] src=%s | inst=%s | tgt=%s | pred=%s",
                        loss_stats.get("reason_preview_src", ""),
                        loss_stats.get("reason_preview_instruction", ""),
                        loss_stats.get("reason_preview_target", ""),
                        loss_stats.get("reason_preview_pred", ""),
                    )
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

                for name, val in log_data.items():
                    name = "train/" + name
                    if tb_writer is not None:
                        tb_writer.add_scalar(name, val, timestep)
                    if args.wandb:
                        wandb.log({name: val, "step": timestep})

            # Periodic CIRR validation evaluation (default enabled).
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
                            }
                            with open(log_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                model.train()
                img2text.train()
                if reasoning_modules is not None:
                    reasoning_modules["projector"].train()
                    reasoning_modules["llm"].eval()

                if args.distributed and dist.is_initialized():
                    dist.barrier()

            def _save_step_checkpoint(save_step: int):
                save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}_step_{save_step}.pt")
                reason_projector_state = None
                if reasoning_modules is not None:
                    reason_projector_state = reasoning_modules["projector"].state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "step": save_step,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "state_dict_reason_projector": reason_projector_state,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                logging.info(f"Saved checkpoint at step {save_step}: {save_path}")
                # Also save LoRA-only weights (all LoRA params, visual + text).
                lora_state_dict = {}
                for n, p in model.named_parameters():
                    if not (n.endswith(".A") or n.endswith(".B")):
                        continue
                    lora_state_dict[n] = p.data.clone()
                if lora_state_dict:
                    lora_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}_step_{save_step}_lora.pt")
                    torch.save(lora_state_dict, lora_path)
                    logging.info(f"Saved LoRA-only weights ({len(lora_state_dict)} params) to {lora_path}")

            # Save checkpoints by global-step interval within [start, end].
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

            # Save checkpoint at specified percentages for single epoch training
            if save_at_percentages is not None and is_master(args):
                percent_complete = (update_idx + 1) / num_updates_per_epoch
                for pct in save_at_percentages:
                    if not hasattr(train, '_saved_percentages'):
                        train._saved_percentages = set()
                    if pct not in train._saved_percentages and percent_complete >= pct:
                        save_step = update_idx + 1
                        _save_step_checkpoint(save_step)
                        train._saved_percentages.add(pct)

            # Log clamp summary at end of epoch
            if is_master(args) and hasattr(train, 'clamp_hit_count') and train.clamp_hit_count > 0:
                logging.info(f"🔒 [Epoch {epoch}] Logit scale clamp hit {train.clamp_hit_count} times in total")

            update_idx += 1
