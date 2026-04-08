"""
========================================================================
从 src/trainer.py 提取的与 method 相关的核心代码
========================================================================
"""

# ==========================================================
# 1. Retrieval Branch Loss: get_loss_lcom_cc3m()
# ==========================================================

def _build_retrieval_prompt(instruction, placeholder, args):
    """构建 composed query 的 text prompt"""
    inst_str = _to_text(instruction).strip()
    if not inst_str:
        return f"a photo of {placeholder}"
    return f"a photo of {placeholder} that {inst_str}"


def get_loss_lcom_cc3m(model, img2text, ref_images, instructions, modified_captions, loss_fn, args):
    """
    检索分支的核心训练损失。

    Query: h(I_r, t) = encode_text_img_vis(prompt_with_*, img2text(encode_image(I_r)))
    Target: h_m = encode_text(modified_caption)
    Loss: symmetric cross-entropy (query↔caption)
    """
    device = ref_images.device

    # 1. 视觉编码 (LoRA 在 visual tower 也有，梯度会回传)
    image_features = model.encode_image(ref_images)          # [B, embed_dim]

    # 2. Textual inversion: visual feature → pseudo-word token
    token_features = img2text(image_features)               # [B, transformer_width]

    # 3. 构建 prompt 并编码 composed query
    placeholder = getattr(args, "prompt_placeholder", "*")
    placeholder_token_id = int(tokenize([placeholder])[0][1].item())
    prompts = [_build_retrieval_prompt(inst, placeholder, args) for inst in instructions]
    prompt_tokens = tokenize(prompts, truncate=True).to(device, non_blocking=True)

    query_features = model.encode_text_img_vis(prompt_tokens, token_features, split_ind=placeholder_token_id)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)

    # 4. Target caption 编码
    cap_tokens = tokenize([_to_str(x) for x in modified_captions], truncate=True).to(device)
    cap_features = model.encode_text(cap_tokens)
    cap_features = cap_features / cap_features.norm(dim=-1, keepdim=True)

    # 5. Temperature-scaled logit
    logit_scale = torch.clamp(model.logit_scale.exp(), max=100.0).mean()

    # 6. Distributed all_gather to enlarge negative pool
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered_q = [torch.zeros_like(query_features) for _ in range(world_size)]
        gathered_c = [torch.zeros_like(cap_features) for _ in range(world_size)]
        dist.all_gather(gathered_q, query_features)
        dist.all_gather(gathered_c, cap_features)
        all_q = torch.cat([query_features] + gathered_q[:rank] + gathered_q[rank + 1:])
        all_c = torch.cat([cap_features] + gathered_c[:rank] + gathered_c[rank + 1:])
    else:
        all_q, all_c = query_features, cap_features

    # 7. Symmetric cross-entropy loss
    logits_per_query = logit_scale * (all_q @ all_c.t())
    logits_per_caption = logits_per_query.t()
    targets = torch.arange(all_q.size(0), device=all_q.device, dtype=torch.long)
    loss_q2c = loss_fn(logits_per_query, targets)
    loss_c2q = loss_fn(logits_per_caption, targets)
    loss = (loss_q2c + loss_c2q) / 2

    # per-sample retrieval loss (用于 geo 的 hard sample mining)
    local_batch_size = query_features.size(0)
    local_targets = torch.arange(local_batch_size, device=all_q.device, dtype=torch.long)
    local_q2c = F.cross_entropy(logits_per_query[:local_batch_size], local_targets, reduction="none")
    local_c2q = F.cross_entropy(logits_per_caption[:local_batch_size], local_targets, reduction="none")
    per_sample_retrieval_loss = 0.5 * (local_q2c + local_c2q)

    return loss, stats, {"per_sample_retrieval_loss": per_sample_retrieval_loss.detach(), ...}


# ==========================================================
# 2. Geometric Consistency Branch Loss: get_loss_geo_text_branch()
# ==========================================================

def get_loss_geo_text_branch(
    text_model,       # geo 分支的独立 text encoder (TextEncoderBranch with geo LoRA)
    src_captions,     # source captions
    modified_captions,# target captions
    forward_instructions,   # forward instructions
    reverse_instructions,   # reverse instructions
    args,
):
    """
    Geometric Consistency Loss.

    在 text embedding 空间中要求：
    - f_fwd 方向 ≈ (f_tgt - f_src) 方向
    - f_rev 方向 ≈ -(f_tgt - f_src) 方向
    - f_fwd + f_rev ≈ 0 (zero-sum)
    """
    device = next(text_model.parameters()).device

    # 编码 4 种文本
    z_src = normalize(text_model.encode_text(tokenize(src_captions)))
    z_tgt = normalize(text_model.encode_text(tokenize(modified_captions)))
    z_fwd = normalize(text_model.encode_text(tokenize(forward_instructions)))
    z_rev = normalize(text_model.encode_text(tokenize(reverse_instructions)))

    # caption delta 方向
    delta_raw = z_tgt - z_src
    delta_raw_norm = delta_raw.norm(dim=-1)
    delta_valid = delta_raw_norm > delta_min_norm   # 过滤 delta 太小的样本
    valid_mask = has_all_fields & delta_valid

    delta_fwd = delta_raw / delta_raw_norm.unsqueeze(-1).clamp_min(eps)   # 归一化方向
    delta_rev = -delta_fwd

    # Loss 1: Forward alignment (z_fwd 与 delta 方向对齐)
    fwd_align = (z_fwd * delta_fwd).sum(dim=-1)    # cosine similarity
    loss_fwd = (1.0 - fwd_align[valid_mask]).mean()

    # Loss 2: Reverse alignment (z_rev 与 -delta 方向对齐)
    rev_align = (z_rev * delta_rev).sum(dim=-1)
    loss_rev = (1.0 - rev_align[valid_mask]).mean()

    # Loss 3: Zero-sum regularizer (z_fwd + z_rev 应该接近零向量)
    loss_zero = torch.norm((z_fwd + z_rev)[valid_mask], dim=-1).mean()

    # 总 geo loss
    geom_loss = (
        loss_fwd
        + loss_rev
        + (zero_loss_weight * loss_zero)
    )
    return geom_loss, stats, aux


# ==========================================================
# 3. Hard Sample Mining: select_geo_subset()
# ==========================================================

def select_geo_subset(
    src_captions, modified_captions,
    forward_instructions, reverse_instructions,
    per_sample_retrieval_loss,    # 来自 retrieval branch 的 per-sample loss
    args,
):
    """
    从 batch 中选取 geo 分支要处理的子集。

    mode="hard", topk=8:
      1. 过滤出同时有 src_caption, modified_caption, fwd_instruction, rev_instruction 的样本
      2. 按 per_sample_retrieval_loss 降序排列
      3. 取 top-k 最难样本
    """
    # ... 过滤逻辑 ...
    candidate_mask = has_src & has_tgt & has_fwd & has_rev
    candidate_indices = candidate_mask.nonzero().flatten()

    if mode == "hard" and per_sample_retrieval_loss is not None:
        candidate_scores = per_sample_retrieval_loss[candidate_indices]
        _, order = torch.topk(candidate_scores, k=topk, largest=True, sorted=False)
        selected_indices = candidate_indices[order]

    return selected_indices.tolist(), stats


# ==========================================================
# 4. PCGrad-style Gradient Projection (简要提及)
# ==========================================================

def project_geo_gradients(retrieval_model, geo_text_model):
    """
    对 geo 分支的 LoRA 梯度做投影：
    如果 geo 梯度与 retrieval 梯度的点积 < 0（冲突），
    将 geo 梯度投影到与 retrieval 梯度正交的方向。

    减少双分支训练中的梯度干扰。
    """
    for name, param in geo_text_model.named_parameters():
        if param.grad is None:
            continue
        retr_grad = retrieval_grads.get(name)
        if retr_grad is None:
            continue
        geo_flat = param.grad.reshape(-1).float()
        retr_flat = retr_grad.reshape(-1).float()
        dot_val = torch.dot(geo_flat, retr_flat)
        if dot_val < 0:  # 梯度冲突
            retr_norm_sq = torch.dot(retr_flat, retr_flat)
            if retr_norm_sq > 1e-12:
                scale = (dot_val / retr_norm_sq).to(dtype=param.grad.dtype)
                param.grad.sub_(scale * retr_grad)


# ==========================================================
# 5. 训练主循环中的双分支并行训练逻辑 (关键片段)
# ==========================================================

def train(..., geo_text_model=None, geo_optimizer=None, ...):
    """训练主循环的关键逻辑"""

    for batch in dataloader:
        # ---- Retrieval Branch ----
        retrieval_loss, retrieval_stats, retrieval_aux = get_loss_lcom_cc3m(
            model, img2text, images, instructions, modified_captions, loss_ce, args
        )
        retrieval_scaler.scale(retrieval_loss / accum_steps).backward()

        # visual encoder / img2text 仅在 retrieval branch 中训练
        # 保存 shared-B 参数上 retrieval 分支的梯度
        if shared_b_retrieval_only:
            shared_b_saved_grads = _capture_named_grads(model, shared_b_param_names)

        # ---- Geo Branch (与 retrieval 同步训练，基于同一 batch 的 hard subset) ----
        if geo_enabled:
            # hard sample mining: 根据 retrieval per-sample loss 选 top-k
            geo_indices, _ = select_geo_subset(
                src_captions, modified_captions, instructions, reverse_instructions,
                retrieval_aux["per_sample_retrieval_loss"], args
            )
            geo_src = [src_captions[i] for i in geo_indices]
            geo_tgt = [modified_captions[i] for i in geo_indices]
            geo_fwd = [instructions[i] for i in geo_indices]
            geo_rev = [reverse_instructions[i] for i in geo_indices]

            with _geo_rng_context(geo_rng_state, enabled=True):
                geo_loss, geo_stats, _ = get_loss_geo_text_branch(
                    geo_text_model, geo_src, geo_tgt, geo_fwd, geo_rev, args
                )
                weighted_geo_loss = geo_loss * geo_weight
                geo_scaler.scale(weighted_geo_loss / accum_steps).backward()

                # 恢复 shared-B 的梯度为 retrieval-only
                if shared_b_saved_grads is not None:
                    _restore_named_grads(model, shared_b_saved_grads)

        # ---- Optimizer Step (gradient accumulation boundary) ----
        if (micro_idx % accum_steps) == 0:
            # 用于稳定联合训练的 PCGrad-style projection
            if geo_conflict_projection:
                project_geo_gradients(model, geo_text_model)

            optimizer.step()        # retrieval + img2text + shared-B
            geo_optimizer.step()    # geo A only (shared-B excluded from geo optimizer)
            optimizer.zero_grad()
            geo_optimizer.zero_grad()
