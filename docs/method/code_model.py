"""
========================================================================
从 model/model.py 提取的与 method 相关的核心代码
========================================================================
"""

# ---------- img2text 映射网络 ----------

class IM2TEXT(nn.Module):
    """原始 Pic2Word 风格：Linear-Dropout-ReLU × n_layer + Linear"""
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class Phi(nn.Module):
    """SEARLE-compatible textual inversion network (Linear-GELU-Dropout × 2 + Linear)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


# ---------- CLIP 的 composed query 编码 ----------

class CLIP(nn.Module):
    # ... (标准 CLIP 定义省略) ...

    def encode_text_img_vis(self, text, img_tokens, split_ind=4):
        """
        核心 composed encoding：
        将 pseudo-token s* 插入到 text prompt 中 placeholder ('*') 的位置，
        然后通过 text transformer 编码。

        text: tokenized prompt, e.g. a dataset-aligned template
              such as "a photo of * and {instruction}" (CIRR)
              or "a photo of * that {instruction}" (CIRCO/FashionIQ)
        img_tokens: [B, d_model]，来自 img2text(encode_image(I_r))
        split_ind: placeholder token id (对应 '*')
        """
        x = self.token_embedding(text).type(self.dtype)
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]
        new_x = []
        for i, sample in enumerate(x):
            ind_insert = text[i] == split_ind
            sample = sample.view(1, x.size(1), -1)
            img = img_tokens[i].view(1, 1, -1)
            inds = ind_insert.nonzero()
            ind0 = inds[0]
            sample = torch.cat([sample[:, :ind0], img, sample[:, ind0+1:]], dim=1)
            new_x.append(sample)
        x = torch.cat(new_x, dim=0)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x


# ---------- LoRA 定义 ----------

class LoRALinear(nn.Module):
    """
    Wrap a frozen nn.Linear with LoRA adapters.
    y = base(x) + (alpha/r) * (x @ A^T) @ B^T
    """
    def __init__(self, base: nn.Linear, r: int = 64, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.empty(r, base.in_features))      # [r, in]
        self.B = nn.Parameter(torch.empty(base.out_features, r))      # [out, r]
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        out = F.linear(x, self.base.weight, self.base.bias)
        lora = (self.dropout(x) @ self.A.t()) @ self.B.t()
        return out + self.scaling * lora


class LoRAMultiheadAttention(nn.MultiheadAttention):
    """
    LoRA on Q/K/V projections (via LoRAProjection) and output projection (via LoRALinear).
    """
    def __init__(self, base: nn.MultiheadAttention, r=64, alpha=16, dropout=0.0):
        super().__init__(...)
        super().load_state_dict(base.state_dict())
        # 冻结原始权重
        self.in_proj_weight.requires_grad = False
        # 添加 LoRA
        self.q_proj_lora = LoRAProjection(self.embed_dim, self.embed_dim, r=r, alpha=alpha)
        self.k_proj_lora = LoRAProjection(self.embed_dim, self.embed_dim, r=r, alpha=alpha)
        self.v_proj_lora = LoRAProjection(self.embed_dim, self.embed_dim, r=r, alpha=alpha)
        self.out_proj = LoRALinear(self.out_proj, r=r, alpha=alpha, dropout=dropout)

    @property
    def effective_in_proj_weight(self):
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        return torch.cat([
            q_weight + self.q_proj_lora.lora_weight,
            k_weight + self.k_proj_lora.lora_weight,
            v_weight + self.v_proj_lora.lora_weight,
        ], dim=0)
