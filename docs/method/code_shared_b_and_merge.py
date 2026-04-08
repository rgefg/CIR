"""
========================================================================
从 src/main.py 和 data/merge_lora_ties.py 提取的
Shared-B 训练设计 + Hybrid Layerwise Merge 相关代码
========================================================================
"""

# ===========================================================
# Part A: 训练阶段 — Shared-B LoRA 绑定 (from src/main.py)
# ===========================================================

class TextEncoderBranch(nn.Module):
    """
    Geo 分支的独立 text encoder。
    从 CLIP text encoder 深拷贝基础权重（不含 LoRA），
    然后独立添加 LoRA adapter。
    """
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        base = clip_model.module if hasattr(clip_model, "module") else clip_model
        self.transformer = _clone_module_without_lora(base.transformer)
        self.token_embedding = _clone_module_without_lora(base.token_embedding)
        self.positional_embedding = nn.Parameter(base.positional_embedding.detach().clone())
        self.ln_final = _clone_module_without_lora(base.ln_final)
        self.text_projection = nn.Parameter(base.text_projection.detach().clone())
        self.end_id = int(base.end_id)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x


def _tie_lora_b_parameters(retr_module, geo_module, prefix, excluded_geo_names):
    """
    将 geo 分支的 LoRA B 参数指针绑定到 retrieval 分支的 B。
    训练时 geo 分支使用与 retrieval 相同的 B 矩阵，只有 A 是独立的。
    """
    if _is_lora_linear(retr_module) and _is_lora_linear(geo_module):
        assert retr_module.B.shape == geo_module.B.shape
        # 关键操作: geo 的 B 直接指向 retrieval 的 B
        setattr(geo_module, "B", retr_module.B)
        excluded_geo_names.add(f"{prefix}.B")


def tie_shared_b_between_text_branches(clip_model, text_branch, num_layers=6):
    """
    对浅层 blocks [0, num_layers-1] 的所有 LoRA 模块，
    将 geo 分支的 B 参数绑定到 retrieval 分支的 B。

    效果:
    - 浅层: B_r = B_g = B (共享), A_r 和 A_g 各自独立
    - 深层: A_r, B_r, A_g, B_g 全部独立
    """
    excluded_geo_names = set()

    for name, geo_module in geo_modules.items():
        block_idx = _text_resblock_index(name)
        if block_idx is not None and block_idx >= num_layers:
            continue  # 深层不共享
        retr_module = retr_modules.get(name)
        if retr_module is None:
            continue
        _tie_lora_b_parameters(retr_module, geo_module, prefix, excluded_geo_names)

    return excluded_geo_names  # 这些参数从 geo optimizer 中排除


# Geo optimizer 构建时排除 shared-B 参数
# geo_param_groups = build_param_groups(
#     geo_named_parameters,
#     args.geo_wd,
#     exclude_name_set=shared_b_geo_exclude_names,  # <-- shared-B 的参数不在 geo optimizer 中
# )


# ===========================================================
# Part B: 合并阶段 — Hybrid Layerwise Merge (from data/merge_lora_ties.py)
# ===========================================================

# 关键范围说明：
# - 只合并 text encoder 的 LoRA 参数
# - visual encoder / img2text / logit_scale / base text weights
#   都沿用 retrieval branch 训练出的参数

def _write_shared_b_sum(prefix, pair_a, pair_b, weight_a, weight_b, scale_a, scale_b, base_scale):
    """
    浅层 Shared-B 合并：直接对 A 做加权平均，B 保持不变。

    由于训练时 B_a = B_b = B (共享)，合并时:
      A_merged = w_r * (scale_r/base_scale) * A_r + w_g * (scale_g/base_scale) * A_g
      B_merged = B  (不变)

    这是精确合并，无信息损失。
    """
    aA, aB = pair_a["A"].float(), pair_a["B"].float()
    bA, bB = pair_b["A"].float(), pair_b["B"].float()

    b_ref = aB if base == "a" else bB  # B 不变
    coeff_a = weight_a * (scale_a / base_scale)
    coeff_b = weight_b * (scale_b / base_scale)
    merged_A = (coeff_a * aA) + (coeff_b * bA)

    return merged_A, b_ref


def _write_delta_ties(prefix, pair_a, pair_b, weights, density, scale_a, scale_b, base_scale):
    """
    深层 TIES 合并：在 delta-weight 空间中做 TIES merge，然后 SVD 回 LoRA 格式。

    步骤:
    1. 计算 effective delta: ΔW_r = scale_r * (B_r @ A_r)
    2. 计算 effective delta: ΔW_g = scale_g * (B_g @ A_g)
    3. TIES merge: ΔW_m = TIES([ΔW_r, ΔW_g], weights, density)
       - Trim: 保留 top (density)% 的元素
       - Elect Sign: 多数投票决定每个位置的符号
       - Disjoint Merge: 每个位置只保留与选定符号一致的值并平均
    4. SVD 分解: ΔW_m = U S V^T → A_m, B_m
    """
    delta_a_eff = scale_a * (aB @ aA)   # [out_dim, in_dim]
    delta_b_eff = scale_b * (bB @ bA)

    delta_m_eff = ties(
        [delta_a_eff, delta_b_eff],
        weights=w,
        density=density,
        majority_sign_method="total",
    )
    delta_m = delta_m_eff / base_scale

    # SVD 分解回 LoRA 格式
    u, s, vh = torch.linalg.svd(delta_m, full_matrices=False)
    s_root = torch.sqrt(s[:rank])
    B_merged = u[:, :rank] * s_root.unsqueeze(0)   # [out, r]
    A_merged = s_root.unsqueeze(1) * vh[:rank, :]   # [r, in]

    return A_merged, B_merged


# Hybrid Layerwise Merge 主循环
def hybrid_layerwise_merge(valid_prefixes, pair_a, pair_b, shared_b_num_layers=6):
    """
    对每个 LoRA 模块，根据层深度选择合并策略:
    - 浅层 (block < N): Shared-B A-sum (精确合并)
    - 深层 (block >= N): TIES merge (鲁棒合并)

    这和 gradient conflict probe 的发现一致：
    浅层冲突较弱，适合结构化精确合并；
    深层冲突更强，适合在 delta space 做保守 merge。
    """
    for prefix in valid_prefixes:
        layer_idx = _text_resblock_index(prefix)
        if layer_idx is not None and layer_idx < shared_b_num_layers:
            # 浅层: B 共享，只加 A
            _write_shared_b_sum(prefix, pair_a[prefix], pair_b[prefix])
        else:
            # 深层: TIES merge in delta space
            _write_delta_ties(prefix, pair_a[prefix], pair_b[prefix])
