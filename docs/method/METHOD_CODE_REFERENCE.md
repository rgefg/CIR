# Method Code Reference

This file is a compact code bundle for writer/reference models. It collects the method-related implementation that matches the current repository state.

Scope:
- model architecture
- LoRA / Shared-B design
- training data format
- retrieval loss
- geo loss with `blend + fwd + rev + zero`
- merge strategy with `Shared-B` and `SVD` denoising
- gradient-conflict analysis / projection probe

Repository root:
- `/data2/mingyu/composed_image_retrieval`


## 1. Current Method Configuration To Describe

This is the method configuration that is most aligned with the current codebase and recent experiments:

- backbone: `CLIP ViT-L/14`
- img2text: `Phi` initialized from `SEARLE_ViT-L14.pt`
- retrieval branch: image-conditioned text query via `encode_text_img_vis`
- geo branch: text-space edit geometry, with source anchor optionally changed from pure text to `blend(text_source, image_anchor)`
- geo loss: `fwd + rev + zero`
- LoRA: applied on linear layers in text and visual transformer blocks
- branch relation: shallow text blocks can use `Shared-B`
- merge: text-only merge, typically `TIES 0.5:0.5`, or hybrid `Shared-B shallow + deep TIES`, or shallow `SVD-denoise`
- gradient analysis: offline gradient conflict probe and projection comparison


## 2. Training Data Example

Source:
- `data/cc3m_cir_dataset_cleaned_v1mid_v2_with_reverse.jsonl`

Raw JSONL example:

```json
{
  "id": "000003390",
  "brainstorming": "The original caption features a man engaging in business activities during an event. The proposed change is to focus on the visual aspect of the man's attire, suggesting that he is wearing a business suit, which is a significant piece of attire in a business context.",
  "instruction": "Change the man's attire to a business suit.",
  "modified_caption": "A man in a business suit conducts business during an event.",
  "reverse_instruction": "Change the man's attire to non-business suit attire"
}
```

The WebDataset loader combines this JSONL supervision with the image shard sample and attaches:

```python
def _cc3m_cir_wds_attach(sample: dict, id2meta: dict) -> dict:
    """Attach instruction/modified_caption/reverse_instruction to decoded sample."""
    k = sample["__key__"]
    ins, cap, rev = id2meta[k]
    src_caption = sample.get("src_caption", "")
    if isinstance(src_caption, bytes):
        src_caption = src_caption.decode("utf-8", errors="ignore")
    elif src_caption is None:
        src_caption = ""
    else:
        src_caption = str(src_caption)
    return {
        "id": k,
        "ref_img": sample["image"],  # PIL after decode
        "src_caption": src_caption,
        "instruction": ins,
        "modified_caption": cap,
        "reverse_instruction": rev,
    }
```

Relevant loader code:

```python
def _load_cc3m_cir_jsonl(jsonl_path: str, reverse_jsonl_path: str = None):
    """id -> (instruction, modified_caption, reverse_instruction)"""
    ...
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            _id = str(obj.get("id"))
            ins = _normalize_instruction(obj.get("instruction", ""))
            cap = _normalize_modified_caption(obj.get("modified_caption", ""))
            rev = _normalize_reverse_instruction(obj.get("reverse_instruction", ""))
            mp[_id] = (ins, cap, rev)
    ...
```


## 3. Retrieval Prompt Construction

Retrieval prompts are built from the placeholder token and the instruction:

```python
def _build_retrieval_prompt(instruction, placeholder, args):
    inst_str = _to_text(instruction).strip()
    if not inst_str:
        return f"a photo of {placeholder}"
    connector = getattr(args, "retrieval_prompt_connector", "that")
    if connector == "and":
        return f"a photo of {placeholder} and {inst_str}"
    return f"a photo of {placeholder} that {inst_str}"
```

For `CIRR`, the current repo convention is `and`.


## 4. Img2Text / Textual Inversion Module

Source:
- `model/model.py`

The repository currently supports both the original `IM2TEXT` and the SEARLE-style `Phi`. The current experiments use `Phi`.

```python
class IM2TEXT(nn.Module):
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
```


## 5. Image-Conditioned Text Query Encoding

Source:
- `model/model.py`

The retrieval branch constructs a query by inserting the image token into the placeholder position in the tokenized text prompt:

```python
def encode_text_img_vis(self, text, img_tokens, split_ind=4):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    collect_ind = text == self.end_id
    collect_ind = collect_ind.nonzero()[:, 1]
    new_x = []
    for i, sample in enumerate(x):
        ind_insert = text[i] == split_ind
        sample = sample.view(1, x.size(1), -1)
        if isinstance(img_tokens, tuple):
            indexes = ind_insert.nonzero()
            for j, index in enumerate(indexes):
                img = img_tokens[j].view(1, 1, -1)
                sample = torch.cat([sample[:, :index], img, sample[:, index+1:]], dim=1)
        else:
            img = img_tokens[i].view(1, 1, -1)
            inds = ind_insert.nonzero()
            if len(inds) == 0:
                raise ValueError(f"Placeholder token id {split_ind} not found in text[{i}]")
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
```


## 6. LoRA Injection: Linear Layers

Source:
- `src/main.py`

LoRA is injected recursively into linear layers and multi-head attention:

```python
def apply_lora_to_linear_layers(module: nn.Module, r: int, alpha: int, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAMultiheadAttention) or _is_lora_linear(child):
            continue
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear_layers(child, r=r, alpha=alpha, dropout=dropout)
```

Trainable LoRA parameters are the `.A` and `.B` matrices:

```python
for n, p in model.named_parameters():
    if n.endswith(".A") or n.endswith(".B"):
        p.requires_grad = True
```


## 7. Shared-B Design

Source:
- `src/main.py`
- `src/params.py`

Shallow text transformer blocks can tie the `B` factor between retrieval and geo branches:

```python
def tie_shared_b_between_text_branches(clip_model: nn.Module, text_branch: nn.Module, num_layers: int = 6) -> set:
    base = clip_model.module if hasattr(clip_model, "module") else clip_model
    branch = text_branch.module if hasattr(text_branch, "module") else text_branch

    retr_modules = dict(base.transformer.named_modules())
    geo_modules = dict(branch.transformer.named_modules())
    excluded_geo_names = set()
    tied_count = 0

    for name, geo_module in geo_modules.items():
        block_idx = _text_resblock_index(name)
        if block_idx is not None and block_idx >= int(num_layers):
            continue
        retr_module = retr_modules.get(name)
        if retr_module is None:
            continue
        before_count = len(excluded_geo_names)
        prefix = f"transformer.{name}" if name else "transformer"
        _tie_lora_b_parameters(retr_module, geo_module, prefix, excluded_geo_names)
        if len(excluded_geo_names) > before_count:
            tied_count += len(excluded_geo_names) - before_count

    if tied_count == 0:
        raise RuntimeError("Shared-B LoRA was enabled, but no text LoRA B tensors were tied.")
    return excluded_geo_names
```

Activation in training:

```python
if getattr(args, "shared_b_lora", False):
    shared_b_geo_exclude_names = tie_shared_b_between_text_branches(
        model,
        geo_text_model,
        num_layers=int(getattr(args, "shared_b_num_layers", 6)),
    )
    args.shared_b_param_names = sorted(shared_b_geo_exclude_names)
    ...
```

Relevant CLI:

```python
parser.add_argument("--shared-b-lora", dest="shared_b_lora", action="store_true", default=False,
                    help="Tie text LoRA B tensors between retrieval and geo branches, keeping only A task-specific.")
parser.add_argument("--shared-b-num-layers", dest="shared_b_num_layers", type=int, default=6,
                    help="Number of shallow text transformer blocks that use Shared-B LoRA. Deeper blocks remain task-specific.")
parser.add_argument("--shared-b-retrieval-only-update", dest="shared_b_retrieval_only_update", action="store_true", default=False,
                    help="When Shared-B LoRA is enabled, let geo branch use the shared B in forward but restore B gradients to retrieval-only values before stepping.")
```


## 8. Retrieval Loss

Source:
- `src/trainer.py`

The retrieval branch builds an image-grounded composed query and matches it against the modified caption embedding with a symmetric CLIP-style contrastive objective:

```python
def get_loss_lcom_cc3m(model, img2text, ref_images, instructions, modified_captions, loss_fn, args):
    """
    Lcom-only (query->modified_caption) using '*' placeholder injection.

    query: h(i,t) = encode_text_img_vis(prompt_with_*, img2text(encode_image(i)), split_ind=4)
    pos/neg: h_m = encode_text(modified_caption)
    """
    device = ref_images.device

    image_features = model.encode_image(ref_images)          # [B, embed_dim]
    token_features = img2text(image_features)                # [B, transformer_width]

    placeholder = getattr(args, "prompt_placeholder", "*")
    placeholder_token_id = int(tokenize([placeholder])[0][1].item())
    prompts = [_build_retrieval_prompt(inst, placeholder, args) for inst in instructions]
    prompt_tokens = tokenize(prompts, truncate=True).to(device, non_blocking=True)

    query_features = model.encode_text_img_vis(prompt_tokens, token_features, split_ind=placeholder_token_id)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    ...
```


## 9. Geo Loss: `blend + fwd + rev + zero`

Source:
- `src/trainer.py`

Current CLI knobs:

```python
parser.add_argument("--geo-reverse-weight", type=float, default=0.25,
                    help="Weight for the soft reverse-instruction consistency term in the geo branch.")
parser.add_argument("--geo-use-reverse-alignment", action="store_true", default=True,
                    help="Directly align reverse instructions to the reverse text delta in the geo branch.")
parser.add_argument("--geo-zero-loss-weight", type=float, default=0.0,
                    help="Optional weight for the strong zero-style regularizer ||z_fwd + z_rev|| in the geo branch.")
parser.add_argument("--geo-src-prompt-style", type=str, default="plain", choices=["plain", "photo"], ...)
parser.add_argument("--geo-src-anchor-mode", type=str, default="text", choices=["text", "image", "blend"], ...)
parser.add_argument("--geo-src-image-weight", type=float, default=0.25, ...)
parser.add_argument("--geo-src-anchor-detach", action="store_true", default=False, ...)
```

Core implementation:

```python
def get_loss_geo_text_branch(
    text_model,
    src_captions,
    modified_captions,
    forward_instructions,
    reverse_instructions,
    args,
    src_image_tokens=None,
):
    device = next(text_model.parameters()).device
    src_prompt_style = str(getattr(args, "geo_src_prompt_style", "plain")).lower()
    src_anchor_mode = str(getattr(args, "geo_src_anchor_mode", "text")).lower()
    src_image_weight = float(getattr(args, "geo_src_image_weight", 0.25))
    reverse_weight = float(getattr(args, "geo_reverse_weight", 0.25))
    reverse_margin = float(getattr(args, "geo_reverse_margin", 0.0))
    zero_loss_weight = float(getattr(args, "geo_zero_loss_weight", 0.0))
    use_reverse_alignment = bool(getattr(args, "geo_use_reverse_alignment", True))

    csrc_raw = [_to_text(x) for x in src_captions]
    if src_prompt_style == "photo":
        csrc = [f"a photo of {x}" if x.strip() else "" for x in csrc_raw]
    else:
        csrc = csrc_raw
    ctgt = [_to_text(x) for x in modified_captions]
    fwd = [_to_text(x) for x in forward_instructions] if forward_instructions else [""] * len(ctgt)
    rev = [_to_text(x) for x in reverse_instructions] if reverse_instructions else [""] * len(ctgt)
    ...
    z_src_text = _safe_l2_normalize(text_model.encode_text(csrc_tokens), dim=-1, eps=embed_norm_eps)
    z_tgt = _safe_l2_normalize(text_model.encode_text(ctgt_tokens), dim=-1, eps=embed_norm_eps)
    z_fwd = _safe_l2_normalize(text_model.encode_text(fwd_tokens), dim=-1, eps=embed_norm_eps)
    z_rev = _safe_l2_normalize(text_model.encode_text(rev_tokens), dim=-1, eps=embed_norm_eps)

    z_src_image = None
    src_anchor_cos = None
    if src_anchor_mode in {"image", "blend"}:
        if src_image_tokens is None:
            raise ValueError("geo_src_anchor_mode requires src_image_tokens, but none were provided")
        src_image_tokens = src_image_tokens.to(device=device, dtype=text_model.dtype)
        placeholder = getattr(args, "prompt_placeholder", "*")
        placeholder_token_id = int(tokenize([placeholder])[0][1].item())
        src_anchor_prompts = [f"a photo of {placeholder}" for _ in csrc]
        src_anchor_tokens = tokenize(src_anchor_prompts, truncate=True).to(device, non_blocking=True)
        z_src_image = _safe_l2_normalize(
            text_model.encode_text_img_vis(src_anchor_tokens, src_image_tokens, split_ind=placeholder_token_id),
            dim=-1,
            eps=embed_norm_eps,
        )
        src_anchor_cos = (z_src_text * z_src_image).sum(dim=-1)

    if src_anchor_mode == "image":
        z_src = z_src_image
    elif src_anchor_mode == "blend":
        z_src = _safe_l2_normalize(
            ((1.0 - src_image_weight) * z_src_text) + (src_image_weight * z_src_image),
            dim=-1,
            eps=embed_norm_eps,
        )
    else:
        z_src = z_src_text

    delta_raw = z_tgt - z_src
    delta_raw_norm = delta_raw.norm(dim=-1)
    delta_fwd = delta_raw / delta_raw_norm.unsqueeze(-1).clamp_min(delta_norm_eps)
    delta_rev = -delta_fwd

    fwd_align = (z_fwd * delta_fwd).sum(dim=-1)
    rev_align = (z_rev * delta_rev).sum(dim=-1)
    fwd_rev_cos = (z_fwd * z_rev).sum(dim=-1)
    ...
    loss_fwd = (1.0 - valid_fwd_align).mean()
    if use_reverse_alignment:
        loss_rev = (1.0 - valid_rev_align).mean()
    else:
        loss_rev = loss_fwd.detach() * 0.0
    loss_reverse = F.relu(valid_fwd_rev_cos + reverse_margin).mean()
    loss_zero = valid_zero_residual.mean()
    geom_loss = loss_fwd + loss_rev + (reverse_weight * loss_reverse) + (zero_loss_weight * loss_zero)
```

Interpretation of the four terms:
- `loss_fwd`: align forward instruction to `z_tgt - z_src`
- `loss_rev`: align reverse instruction to `z_src - z_tgt`
- `loss_reverse`: soft reverse-consistency term on `cos(z_fwd, z_rev)`
- `loss_zero`: strong anti-parallel regularizer `||z_fwd + z_rev||`

For the current requested method writeup, the preferred presentation is:
- source anchor uses `blend(text source, image anchor)`
- geo loss uses `fwd + rev + zero`


## 10. How The Image Anchor For Geo Is Built

Source:
- `src/trainer.py`

The `image` and `blend` geo-source modes require a source image token computed from the same visual encoder and `f_phi` path used by the retrieval branch:

```python
text_model.encode_text_img_vis(src_anchor_tokens, src_image_tokens, split_ind=placeholder_token_id)
```

Conceptually:
- `ref_img -> f_v -> f_phi`
- inject into `a photo of *`
- encode with the same text tower interface

This is the implementation basis for drawing a geo branch that is not completely detached from the visual pathway.


## 11. Merge Strategy: TIES, Shared-B, and SVD Denoise

Source:
- `data/merge_lora_ties.py`

The merge script supports several modes:

```python
parser.add_argument(
   "--merge-mode",
   type=str,
   default="ties",
   choices=["ties", "shared_b_sum_a", "shared_b_svd_a", "hybrid_layerwise", "hybrid_layerwise_svd_a", "shared_a_sum_b"],
   help="Use TIES in delta space, or keep base B fixed and sum only A for Shared-B LoRA.",
)
parser.add_argument("--shared-b-num-layers", dest="shared_b_num_layers", type=int, default=6, ...)
parser.add_argument("--svd-topk-rank", type=int, default=32, ...)
parser.add_argument("--svd-rescale", action="store_true", default=False, ...)
```

Plain `TIES` is done in delta space:

```python
def _write_delta_ties(prefix: str, pa: Dict[str, torch.Tensor], pb: Dict[str, torch.Tensor]):
   aA = pa["A"].float()
   aB = pa["B"].float()
   bA = pb["A"].float()
   bB = pb["B"].float()
   delta_a_eff = scale_a * (aB @ aA)
   delta_b_eff = scale_b * (bB @ bA)
   delta_m_eff = ties(
      [delta_a_eff, delta_b_eff],
      weights=w,
      density=args.density,
      majority_sign_method=args.majority_sign_method,
   )
   delta_m = delta_m_eff / base_scale
   rank_target = pair_b[prefix]["A"].shape[0] if args.base == "b" else pair_a[prefix]["A"].shape[0]
   out_dtype = pair_b[prefix]["A"].dtype if args.base == "b" else pair_a[prefix]["A"].dtype
   mA, mB = _svd_factorize(delta_m, rank=rank_target, out_dtype=out_dtype)
   ...
```

SVD denoising on the shallow merged `A` is:

```python
def _svd_topk_matrix(mat: torch.Tensor, rank_keep: int, rescale: bool = False) -> torch.Tensor:
   if mat.ndim != 2:
      raise ValueError(f"_svd_topk_matrix expects 2D tensor, got shape={tuple(mat.shape)}")
   max_rank = min(mat.shape[0], mat.shape[1])
   use_r = max(1, min(int(rank_keep), max_rank))
   if use_r >= max_rank:
      return mat
   u, s, vh = torch.linalg.svd(mat, full_matrices=False)
   truncated = (u[:, :use_r] * s[:use_r].unsqueeze(0)) @ vh[:use_r, :]
   if not rescale:
      return truncated
   keep_ratio = float(use_r) / float(max_rank)
   if keep_ratio <= 0.0:
      return truncated
   return truncated / keep_ratio
```

Shared-B SVD merge:

```python
def _write_shared_b_svd_a(prefix: str, pa: Dict[str, torch.Tensor], pb: Dict[str, torch.Tensor]):
   ...
   b_ref = aB if args.base == "a" else bB
   coeff_a = weight_a * (scale_a / base_scale)
   coeff_b = weight_b * (scale_b / base_scale)
   merged_A = (coeff_a * aA) + (coeff_b * bA)
   merged_A = _svd_topk_matrix(
      merged_A,
      rank_keep=args.svd_topk_rank,
      rescale=args.svd_rescale,
   )
   ...
   merged_AB[kA] = merged_A.to(dtype=out_dtype)
   merged_AB[kB] = b_ref.to(dtype=out_dtype)
```

Hybrid layerwise merge:

```python
elif effective_merge_mode == "hybrid_layerwise_svd_a":
   layer_idx = _text_resblock_index(p)
   if layer_idx is not None and layer_idx < int(args.shared_b_num_layers):
      shallow_shared_prefixes += 1
      _write_shared_b_svd_a(p, pair_a[p], pair_b[p])
   else:
      deep_ties_prefixes += 1
      _write_delta_ties(p, pair_a[p], pair_b[p])
```

This is the exact basis for the “shallow Shared-B/SVD-denoise, deep TIES” merge story.


## 12. Gradient Conflict Analysis / Probe

Source:
- `src/probe_offline.py`

The offline probe compares retrieval and geo gradients layerwise and after projection.

Joint loss for a subset:

```python
def compute_joint_loss(model, img2text, batch, indices, loss_fn, args):
    images = subset_tensor(batch["ref_img"], indices)
    instructions = select_items(batch["instruction"], indices)
    modified = select_items(batch["modified_caption"], indices)
    src = select_items(batch["src_caption"], indices)
    reverse = select_items(batch["reverse_instruction"], indices)
    retrieval_loss = get_loss_lcom_textprobe_cc3m(model, img2text, images, instructions, modified, loss_fn, args)
    geo_loss, _, _ = get_loss_geo_text_branch(model, src, modified, instructions, reverse, args)
    return retrieval_loss + (float(getattr(args, "geo_weight", 1.0)) * geo_loss)
```

Gradient collection for retrieval and edit losses:

```python
zero_grads(model, img2text)
ret_loss = get_loss_lcom_textprobe_cc3m(
    model,
    img2text,
    batch["ref_img"],
    batch["instruction"],
    batch["modified_caption"],
    loss_fn,
    args,
)
ret_loss.backward()
ret_vectors = _extract_text_cproj_effective_grads(model)

zero_grads(model, img2text)
edit_loss, _, _ = get_loss_geo_text_branch(
    model,
    batch["src_caption"],
    batch["modified_caption"],
    batch["instruction"],
    batch["reverse_instruction"],
    args,
)
edit_loss.backward()
edit_vectors = _extract_text_cproj_effective_grads(model)
```

Layerwise conflict summary:

```python
def finalize_gc(block_sums, count):
    ...
    s_base[idx] = _cosine_or_none(base_a_avg[idx], base_b_avg[idx])
    s_cross[idx] = _cosine_or_none(ret_avg[idx], edit_avg[idx])
    s_cross_after[idx] = _cosine_or_none(ret_avg[idx], edit_proj_avg[idx])
    s_cross_bidir[idx] = _cosine_or_none(ret_biproj_avg[idx], edit_proj_avg[idx])
    if s_base[idx] is not None and s_cross[idx] is not None:
        gc[idx] = float(s_base[idx] - s_cross[idx])
    if s_base[idx] is not None and s_cross_after[idx] is not None:
        gc_after[idx] = float(s_base[idx] - s_cross_after[idx])
    if s_base[idx] is not None and s_cross_bidir[idx] is not None:
        gc_bidir[idx] = float(s_base[idx] - s_cross_bidir[idx])
```

Parameter-space projection used for the interference test:

```python
def build_projected_param_grads_stats(ret_params, edit_params):
    projected = {}
    n_conflict = 0
    n_total = 0
    for name, edit_grad in edit_params.items():
        ret_grad = ret_params.get(name)
        if ret_grad is None:
            continue
        edit_flat = edit_grad.reshape(-1).float()
        ret_flat = ret_grad.reshape(-1).float()
        dot_val = torch.dot(edit_flat, ret_flat)
        n_total += 1
        proj = edit_flat.clone()
        if float(dot_val.item()) < 0.0:
            n_conflict += 1
            retr_norm_sq = torch.dot(ret_flat, ret_flat)
            if float(retr_norm_sq.item()) > 1e-12:
                proj = proj - (dot_val / retr_norm_sq) * ret_flat
        projected[name] = proj.reshape_as(edit_grad)
    ratio = n_conflict / n_total if n_total > 0 else 0.0
```

This is the implementation basis for:
- “gradient conflict” plots
- “projection stabilizes joint optimization” analysis
- deciding where hybrid merge is more justified


## 13. Training Script Entry

Source:
- `train_with_dropout.sh`

The main launcher wires together:
- training JSON
- prompt connector
- geo loss weights
- shared-B flags
- merge/eval outputs

Representative lines:

```bash
TRAIN_JSON="${TRAIN_JSON:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2_with_reverse.jsonl}"
...
echo "Geo strict loss: reverse_weight=${GEO_REVERSE_WEIGHT}, reverse_margin=${GEO_REVERSE_MARGIN}, zero_loss_weight=${GEO_ZERO_LOSS_WEIGHT}"
...
--cc3m-cir-jsonl "${TRAIN_JSON}" \
...
--output-jsonl "${CIRR_MERGED_JSONL}" \
```


## 14. Suggested Minimal Reading Order For A Writer LLM

If another model only has limited context, give it these sections first:

1. Section 2: Training data example
2. Section 4: `Phi`
3. Section 5: `encode_text_img_vis`
4. Section 7: `Shared-B`
5. Section 8: retrieval loss
6. Section 9: geo loss with `blend + fwd + rev + zero`
7. Section 11: merge strategy
8. Section 12: gradient conflict probe


## 15. Canonical File Pointers

- model:
  - `/data2/mingyu/composed_image_retrieval/model/model.py`
- training entry:
  - `/data2/mingyu/composed_image_retrieval/src/main.py`
- loss / training loop:
  - `/data2/mingyu/composed_image_retrieval/src/trainer.py`
- args:
  - `/data2/mingyu/composed_image_retrieval/src/params.py`
- data:
  - `/data2/mingyu/composed_image_retrieval/src/data.py`
- merge:
  - `/data2/mingyu/composed_image_retrieval/data/merge_lora_ties.py`
- gradient probe:
  - `/data2/mingyu/composed_image_retrieval/src/probe_offline.py`
- launcher:
  - `/data2/mingyu/composed_image_retrieval/train_with_dropout.sh`
