# Copyright 2022 Google LLC
# Licensed under the Apache License, Version 2.0

import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json
import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from third_party.open_clip.scheduler import cosine_lr
from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, ReasoningProjector
from trainer import train
from data import get_data, CIRR
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32

import math
import torch.nn as nn
import copy
import types
import torch.nn.functional as F


# -------------------
# LoRA (minimal, local)
# -------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 64, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.empty(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def lora_weight(self):
        delta = self.B @ self.A
        return (self.scaling * delta).to(dtype=self.base.weight.dtype)

    @property
    def effective_weight(self):
        return self.base.weight + self.lora_weight

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.base.weight, self.base.bias)
        x_d = self.dropout(x)
        A = self.A.to(dtype=x_d.dtype)
        B = self.B.to(dtype=x_d.dtype)
        lora = (x_d @ A.t()) @ B.t()
        return out + (self.scaling * lora).to(dtype=out.dtype)

    @property
    def weight(self):
        return self.effective_weight

    @property
    def bias(self):
        return self.base.bias


class LoRAProjection(nn.Module):
    def __init__(self, out_features: int, in_features: int, r: int = 64, alpha: int = 16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.empty(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def lora_weight(self):
        delta = self.B @ self.A
        return self.scaling * delta


class LoRAMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, base: nn.MultiheadAttention, r: int = 64, alpha: int = 16, dropout: float = 0.0):
        super().__init__(
            embed_dim=base.embed_dim,
            num_heads=base.num_heads,
            dropout=base.dropout,
            bias=base.in_proj_bias is not None,
            add_bias_kv=base.bias_k is not None,
            add_zero_attn=base.add_zero_attn,
            kdim=base.kdim,
            vdim=base.vdim,
            batch_first=base.batch_first,
        )
        super().load_state_dict(base.state_dict())

        if not self._qkv_same_embed_dim:
            raise NotImplementedError("LoRAMultiheadAttention only supports _qkv_same_embed_dim=True.")

        self.in_proj_weight.requires_grad = False
        if self.in_proj_bias is not None:
            self.in_proj_bias.requires_grad = False
        if self.bias_k is not None:
            self.bias_k.requires_grad = False
        if self.bias_v is not None:
            self.bias_v.requires_grad = False

        self.q_proj_lora = LoRAProjection(self.embed_dim, self.embed_dim, r=r, alpha=alpha)
        self.k_proj_lora = LoRAProjection(self.embed_dim, self.embed_dim, r=r, alpha=alpha)
        self.v_proj_lora = LoRAProjection(self.embed_dim, self.embed_dim, r=r, alpha=alpha)
        self.out_proj = LoRALinear(self.out_proj, r=r, alpha=alpha, dropout=dropout)

    @property
    def effective_in_proj_weight(self):
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        return torch.cat(
            [
                q_weight + self.q_proj_lora.lora_weight.to(dtype=q_weight.dtype, device=q_weight.device),
                k_weight + self.k_proj_lora.lora_weight.to(dtype=k_weight.dtype, device=k_weight.device),
                v_weight + self.v_proj_lora.lora_weight.to(dtype=v_weight.dtype, device=v_weight.device),
            ],
            dim=0,
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.effective_in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        return attn_output, attn_output_weights


def apply_lora_to_linear_layers(module: nn.Module, r: int, alpha: int, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
            continue
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear_layers(child, r=r, alpha=alpha, dropout=dropout)


def freeze_clip_except_lora_and_logit_scale(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False
    # LoRA params are named *.A, *.B
    for n, p in model.named_parameters():
        if n.endswith(".A") or n.endswith(".B"):
            p.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def freeze_clip_text_lora_only(model: nn.Module):
    """Freeze all CLIP params, train only text-side LoRA A/B params."""
    for _, p in model.named_parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if (n.endswith(".A") or n.endswith(".B")) and (not n.startswith("visual.")):
            p.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = False


def get_lora_state_dict(model: nn.Module, text_only: bool = False) -> dict:
    """Extract LoRA parameters (A and B matrices) from model."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if not (name.endswith(".A") or name.endswith(".B")):
            continue
        normalized_name = name[len("module."):] if name.startswith("module.") else name
        if text_only and normalized_name.startswith("visual."):
            continue
        lora_state_dict[name] = param.data.clone()
    return lora_state_dict


def _reason_llm_enabled(args) -> bool:
    return bool(getattr(args, "reason_llm_model", None)) and float(getattr(args, "reason_llm_weight", 0.0)) > 0.0


def _get_llm_hidden_size(model) -> int:
    for attr in ["hidden_size", "n_embd", "d_model"]:
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from frozen LLM config.")


def _get_reason_llm_dtype(args):
    dtype_name = str(getattr(args, "reason_llm_dtype", "fp16")).lower()
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float16


def _resolve_reason_llm_source(args) -> str:
    model_spec = str(getattr(args, "reason_llm_model", "") or "").strip()
    if not model_spec:
        raise ValueError("reason_llm_model is empty while frozen-LLM branch is enabled.")

    lower_spec = model_spec.lower()
    # Allow short aliases for convenience.
    alias_map = {
        "qwen3.5-2b": "Qwen/Qwen3.5-2B",
        "qwen-3.5-2b": "Qwen/Qwen3.5-2B",
    }
    hf_model_id = alias_map.get(lower_spec, model_spec)

    local_dir_arg = getattr(args, "reason_llm_local_dir", None)
    if local_dir_arg:
        local_dir = Path(local_dir_arg).expanduser()
    elif hf_model_id == "Qwen/Qwen3.5-2B":
        local_dir = Path(get_project_root()) / "checkpoint" / "hf_models" / "Qwen3.5-2B"
    else:
        local_dir = None

    if local_dir is not None and local_dir.exists():
        return str(local_dir)

    if local_dir is not None and bool(getattr(args, "reason_llm_auto_download", False)):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "reason_llm_auto_download=True requires huggingface_hub. "
                "Please install it via pip install huggingface_hub."
            ) from exc
        local_dir.mkdir(parents=True, exist_ok=True)
        if is_master(args):
            logging.info(f"⬇️ Downloading frozen LLM '{hf_model_id}' to local dir: {local_dir}")
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=getattr(args, "reason_llm_hf_token", None),
            revision=getattr(args, "reason_llm_revision", None),
        )
        if is_master(args):
            logging.info(f"✅ Frozen LLM download completed: {local_dir}")
        return str(local_dir)

    # Fallback to direct HF repo ID / user-provided source.
    return hf_model_id


def build_reasoning_modules(args, model, img2text, device):
    if not _reason_llm_enabled(args):
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "reason_llm_model was set, but transformers is not installed in this environment."
        ) from exc

    if is_master(args):
        logging.info("=" * 80)
        logging.info("🧠 Building frozen-LLM reasoning branch")
        logging.info(f"   reason_llm_model: {args.reason_llm_model}")
        logging.info(f"   reason_llm_divisor: {args.reason_llm_weight}")
        logging.info(f"   soft_prompt_len: {args.reason_llm_soft_prompt_len}")
        logging.info("=" * 80)

    llm_source = _resolve_reason_llm_source(args)

    tokenizer = AutoTokenizer.from_pretrained(
        llm_source,
        trust_remote_code=bool(getattr(args, "reason_llm_trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm_load_kwargs = {
        "trust_remote_code": bool(getattr(args, "reason_llm_trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    llm_dtype = _get_reason_llm_dtype(args)
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_source,
            dtype=llm_dtype,
            **llm_load_kwargs,
        )
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_source,
            torch_dtype=llm_dtype,
            **llm_load_kwargs,
        )
    llm.config.use_cache = False
    llm.requires_grad_(False)
    llm.eval()
    llm.to(device)

    projector = ReasoningProjector(
        embed_dim=model.embed_dim,
        middle_dim=args.middle_dim,
        clip_token_dim=model.token_embedding.weight.shape[1],
        llm_hidden_size=_get_llm_hidden_size(llm),
        prompt_length=int(getattr(args, "reason_llm_soft_prompt_len", 4)),
        n_layer=args.n_layer,
        dropout=getattr(args, "droprate", 0.1),
    )
    if getattr(args, "reason_projector_init", "pic2word") == "pic2word":
        msg = projector.init_from_img2text(img2text)
        if is_master(args):
            logging.info(
                "Initialized reasoning projector stem from pic2word weights "
                f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})."
            )
    projector.to(device)

    if is_master(args):
        proj_for_log = projector.module if hasattr(projector, "module") else projector
        proj_params = sum(p.numel() for p in proj_for_log.parameters())
        logging.info(f"   reasoning projector parameters: {proj_params:,}")

    return {
        "projector": projector,
        "llm": llm,
        "tokenizer": tokenizer,
    }


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"{name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # =========================================================================
    # 1. Build CLIP Model
    # =========================================================================
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(args.model, jit=False)
    elif args.pic2word_pretrained:
        # Load from pic2word pretrained model
        if is_master(args):
            logging.info("=" * 80)
            logging.info(f"🔄 Loading pic2word pretrained model from: {args.pic2word_pretrained}")
            logging.info("=" * 80)
        
        checkpoint = torch.load(args.pic2word_pretrained, map_location='cpu', weights_only=False)
        
        # Debug: Print checkpoint keys
        if is_master(args):
            logging.info(f"📦 Checkpoint top-level keys: {list(checkpoint.keys())}")
            for k in checkpoint.keys():
                if isinstance(checkpoint[k], dict):
                    logging.info(f"   - {k}: {len(checkpoint[k])} parameters")
        
        # [Robustness Fix]: Handle different checkpoint formats
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            model = CLIP(**model_config)
            if is_master(args):
                logging.info("✅ Using model_config from checkpoint to build CLIP model")
        else:
            # Fallback to default config if not present (assuming ViT-L/14)
            if is_master(args): 
                logging.warning("⚠️ model_config not found in checkpoint, assuming ViT-L/14 default config.")
            model, _, _ = load("ViT-L/14", jit=False) # Load structure only

        # Load CLIP weights
        sd = checkpoint['state_dict']
        if is_master(args):
            logging.info(f"📥 CLIP state_dict contains {len(sd)} parameters")
            # Show sample keys
            sample_keys = list(sd.keys())[:5]
            logging.info(f"   Sample keys: {sample_keys}...")
        
        if next(iter(sd.items()))[0].startswith('module.'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
            if is_master(args):
                logging.info("🔧 Removed 'module.' prefix from state_dict keys")
        
        msg = model.load_state_dict(sd, strict=False)
        if is_master(args):
            logging.info("=" * 80)
            logging.info(f"📊 CLIP Weight Loading Summary:")
            logging.info(f"   ✅ Missing keys: {len(msg.missing_keys)}")
            logging.info(f"   ⚠️  Unexpected keys: {len(msg.unexpected_keys)}")
            if len(msg.missing_keys) > 0:
                logging.warning(f"   Missing keys (first 10): {msg.missing_keys[:10]}")
            if len(msg.unexpected_keys) > 0:
                logging.warning(f"   Unexpected keys (first 10): {msg.unexpected_keys[:10]}")
            logging.info("=" * 80)
        
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)
    else:
        # Initial from scratch (rarely used)
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        assert os.path.exists(model_config_file)
        with open(model_config_file, "r") as f:
            model_info = json.load(f)
        if args.use_prefix:
            model_info["vocab_size"] += 1
            model_info["use_prefix"] = True
        model = CLIP(**model_info)
        convert_weights(model)
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    # =========================================================================
    # 2. Build img2text Mapping Network
    # =========================================================================
    img2text = IM2TEXT(
        embed_dim=model.embed_dim,
        middle_dim=args.middle_dim,
        output_dim=model.token_embedding.weight.shape[1],
        n_layer=args.n_layer,
        dropout=getattr(args, "droprate", 0.1),
    )

    # =========================================================================
    # 3. Load img2text weights (CRITICAL FIX)
    # =========================================================================
    if args.pic2word_pretrained:
        if is_master(args):
            logging.info("=" * 80)
            logging.info("🔄 Loading img2text weights from pic2word checkpoint...")
            logging.info("=" * 80)
        
        checkpoint = torch.load(args.pic2word_pretrained, map_location='cpu', weights_only=False)
        
        # Try finding img2text weights
        sd_i2t = None
        if 'state_dict_img2text' in checkpoint:
            sd_i2t = checkpoint['state_dict_img2text']
            if is_master(args):
                logging.info(f"✅ Found 'state_dict_img2text' key in checkpoint")
                logging.info(f"   Contains {len(sd_i2t)} parameters")
                # Show all keys in img2text
                logging.info(f"   All img2text parameter names:")
                for k in sorted(sd_i2t.keys()):
                    shape = sd_i2t[k].shape if hasattr(sd_i2t[k], 'shape') else 'N/A'
                    logging.info(f"      - {k}: shape={shape}")
        else:
            # Maybe mixed in main state_dict? (Unlikely for standard code but possible)
            if is_master(args): 
                logging.warning("⚠️ 'state_dict_img2text' key not found in checkpoint!")
                logging.info("   Available keys in checkpoint:")
                for k in checkpoint.keys():
                    logging.info(f"      - {k}")
        
        if sd_i2t is not None:
            # Handle DDP prefix
            if next(iter(sd_i2t.items()))[0].startswith('module.'):
                sd_i2t = {k[len('module.'):]: v for k, v in sd_i2t.items()}
                if is_master(args):
                    logging.info("🔧 Removed 'module.' prefix from img2text state_dict keys")
            
            # Check expected img2text parameters
            if is_master(args):
                expected_keys = set(img2text.state_dict().keys())
                checkpoint_keys = set(sd_i2t.keys())
                logging.info(f"📋 Expected img2text keys: {len(expected_keys)}")
                logging.info(f"📋 Checkpoint img2text keys: {len(checkpoint_keys)}")
                missing = expected_keys - checkpoint_keys
                extra = checkpoint_keys - expected_keys
                if missing:
                    logging.warning(f"   ⚠️  Missing keys: {missing}")
                if extra:
                    logging.warning(f"   ⚠️  Extra keys: {extra}")
            
            msg = img2text.load_state_dict(sd_i2t, strict=False)
            if is_master(args): 
                logging.info("=" * 80)
                logging.info(f"📊 img2text Weight Loading Summary:")
                logging.info(f"   ✅ Missing keys: {len(msg.missing_keys)}")
                logging.info(f"   ⚠️  Unexpected keys: {len(msg.unexpected_keys)}")
                if len(msg.missing_keys) > 0:
                    logging.error(f"   ❌ Missing keys: {msg.missing_keys}")
                if len(msg.unexpected_keys) > 0:
                    logging.warning(f"   ⚠️  Unexpected keys: {msg.unexpected_keys}")
                
                # Verify weights are actually loaded (not random)
                if len(msg.missing_keys) == 0:
                    logging.info("✅ SUCCESS: All img2text weights loaded successfully!")
                else:
                    logging.error("❌ ERROR: Some img2text weights are missing! Model may be partially random initialized!")
                logging.info("=" * 80)
        else:
            # FATAL ERROR CHECK
            err_msg = "❌ ERROR: --pic2word-pretrained was provided but 'state_dict_img2text' was NOT found in the checkpoint. Model would be random initialized!"
            if is_master(args): 
                logging.error("=" * 80)
                logging.error(err_msg)
                logging.error("=" * 80)
            raise RuntimeError(err_msg)
            
    else:
        if is_master(args):
            logging.warning("=" * 80)
            logging.warning("⚠️ WARNING: No pic2word pretrained path provided. img2text is INITIALIZED FROM SCRATCH (Random)!")
            logging.warning("=" * 80)

    # =========================================================================
    # 4. Precision & LoRA Setup
    # =========================================================================
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    # ---- Enable LoRA on CLIP (paper-consistent) ----
    if not getattr(args, "no_lora", False):
        apply_lora_to_linear_layers(model, r=getattr(args, "lora_r", 64), alpha=getattr(args, "lora_alpha", 16), dropout=getattr(args, "lora_dropout", 0.0))
    
    # Joint training (train_inference_lora) now uses retrieval + LLM loss together,
    # so both visual and text LoRA must be trainable (same as retrieval-only path).
    freeze_clip_except_lora_and_logit_scale(model)
    
    # ============================================================
    # 🔒 Logit Scale 修复选项
    # ============================================================
    # 如果logit_scale被错误初始化或加载，可以强制重置
    if getattr(args, "reset_logit_scale", False):
        import numpy as np
        with torch.no_grad():
            model.logit_scale.data.fill_(np.log(1 / 0.07))  # 标准CLIP初始化
        if is_master(args):
            logging.info("🔧 Reset logit_scale to standard CLIP value: log(1/0.07) ≈ 2.659")

    # 可选：锁定logit_scale（用于调试）
    if getattr(args, "freeze_logit_scale", False):
        model.logit_scale.requires_grad = False
        if is_master(args):
            logging.info("🔒 Frozen logit_scale (requires_grad=False) for debugging")

    # 🔒 两阶段 logit_scale 训练
    # 前 freeze_percent 步冻结，后解冻但 clamp 到安全区间
    logit_scale_clamp_min = getattr(args, "logit_scale_clamp_min", None)
    logit_scale_clamp_max = getattr(args, "logit_scale_clamp_max", None)
    logit_scale_freeze_percent = getattr(args, "logit_scale_freeze_percent", 0.3)

    # 在 trainer.py 中会使用这些参数来控制 logit_scale 的训练和 clamp
    if logit_scale_clamp_min is not None or logit_scale_clamp_max is not None:
        if is_master(args):
            logging.info(f"📏 Logit Scale clamp: [{logit_scale_clamp_min}, {logit_scale_clamp_max}]")

    if is_master(args):
        logging.info(f"⏳ Logit Scale first {logit_scale_freeze_percent*100:.0f}% steps frozen, then trainable with clamping")
    # ============================================================
    
    # img2text is always trainable (joint training needs it for retrieval loss).
    for p in img2text.parameters():
        p.requires_grad = True
    
    # Log model parameter statistics
    if is_master(args):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        img2text_params = sum(p.numel() for p in img2text.parameters())
        logging.info("=" * 80)
        logging.info("📊 Model Parameter Statistics:")
        logging.info(f"   CLIP total parameters: {total_params:,}")
        logging.info(f"   CLIP trainable parameters: {trainable_params:,} ({100.*trainable_params/total_params:.2f}%)")
        logging.info(f"   img2text parameters: {img2text_params:,}")
        
        # Verify img2text weights are non-zero (if loaded from checkpoint)
        if args.pic2word_pretrained:
            logging.info("=" * 80)
            logging.info("🔍 Verifying img2text weights (checking if non-zero)...")
            all_zero = True
            for name, param in img2text.named_parameters():
                if param.abs().sum().item() > 1e-6:  # Check if any weight is non-zero
                    all_zero = False
                    logging.info(f"   ✅ {name}: contains non-zero weights (shape: {param.shape})")
                    break
            if all_zero:
                logging.error("   ❌ WARNING: All img2text weights appear to be zero! This may indicate loading failure!")
            else:
                logging.info("   ✅ img2text weights verified: non-zero weights detected")
            logging.info("=" * 80)

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            img2text_has_trainable = any(p.requires_grad for p in img2text.parameters())
            if img2text_has_trainable:
                img2text = torch.nn.parallel.DistributedDataParallel(
                    img2text, device_ids=[args.gpu], find_unused_parameters=False
                )
            elif is_master(args):
                logging.info("img2text has no trainable parameters; skip wrapping it with DDP.")
        
        # ... (DP logic omitted for brevity, Distributed is preferred)

    reasoning_modules = None
    if _reason_llm_enabled(args):
        if not getattr(args, "train_inference_lora", False) and is_master(args):
            logging.warning(
                "Frozen-LLM reasoning supervision is currently designed for dict-batch training. "
                "It is recommended to pair it with --train-inference-lora."
            )
        if is_master(args):
            if torch.cuda.is_available():
                reasoning_device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
            else:
                reasoning_device = torch.device("cpu")
            reasoning_modules = build_reasoning_modules(
                args,
                model.module if hasattr(model, "module") else model,
                img2text,
                reasoning_device,
            )
            logging.info(
                f"🧠 LLM loaded on rank 0 only (GPU {args.gpu}). "
                f"Other {args.world_size - 1} ranks run retrieval loss only."
            )
        else:
            logging.info(f"Rank {args.rank}: skipping LLM load (rank 0 handles reasoning).")

    # Rank 0 carries the LLM so it may need a smaller retrieval batch.
    args.base_batch_size = args.batch_size
    rank0_bs = int(getattr(args, "rank0_batch_size", 0))
    if rank0_bs > 0 and is_master(args) and reasoning_modules is not None:
        args.batch_size = rank0_bs
        actual_per_microbatch = rank0_bs + args.base_batch_size * (args.world_size - 1)
        args.actual_samples_per_microbatch = actual_per_microbatch
        logging.info(
            f"🔧 Rank 0 batch_size overridden: {args.base_batch_size} → {rank0_bs} "
            f"(LLM occupies GPU memory). "
            f"Actual per micro-batch: {rank0_bs}+{args.base_batch_size}×{args.world_size - 1}={actual_per_microbatch}"
        )
    else:
        args.actual_samples_per_microbatch = args.batch_size * args.world_size

    # Data
    data = get_data(args, (preprocess_train, preprocess_val))
    args.cirr_val_eval_log_path = os.path.join(args.logs, args.name, "cirr_val_eval.log")
    cirr_eval_every = int(getattr(args, "cirr_val_eval_every", 0))
    if cirr_eval_every > 0:
        try:
            root_project = os.path.join(get_project_root(), "data")
            cirr_val_batch_size = int(args.batch_size)
            cirr_val_workers = int(args.workers)

            source_dataset = CIRR(transforms=preprocess_val, root=root_project)
            target_dataset = CIRR(transforms=preprocess_val, root=root_project, mode="imgs")
            data["cirr_val_query_loader"] = DataLoader(
                source_dataset,
                batch_size=cirr_val_batch_size,
                shuffle=False,
                num_workers=cirr_val_workers,
                pin_memory=True,
                drop_last=False,
            )
            data["cirr_val_target_loader"] = DataLoader(
                target_dataset,
                batch_size=cirr_val_batch_size,
                shuffle=False,
                num_workers=cirr_val_workers,
                pin_memory=True,
                drop_last=False,
            )
            if is_master(args):
                logging.info(
                    f"✅ Periodic CIRR-val eval enabled: every {cirr_eval_every} steps, "
                    f"query={len(source_dataset)}, gallery={len(target_dataset)}, "
                    f"batch_size={cirr_val_batch_size}, workers={cirr_val_workers}, "
                    f"log_file={args.cirr_val_eval_log_path}"
                )
        except Exception as e:
            if is_master(args):
                logging.warning(f"⚠️ Failed to initialize periodic CIRR-val eval; disabling it. Error: {e}")
            args.cirr_val_eval_every = 0

    # ---- Optimizer ----
    def exclude(n):
        return ("bn" in n) or ("ln" in n) or ("bias" in n) or ("logit_scale" in n)

    def include(n):
        return not exclude(n)

    # collect trainable params
    named_parameters = []
    img2text_for_opt = img2text.module if hasattr(img2text, "module") else img2text
    model_for_opt = model.module if hasattr(model, "module") else model
    named_parameters += list(img2text_for_opt.named_parameters())
    named_parameters += list(model_for_opt.named_parameters())
    if reasoning_modules is not None:
        named_parameters += list(reasoning_modules["projector"].named_parameters())

    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # ---- Scheduler ----
    nb = getattr(data["train"].dataloader, "num_batches", None)
    if nb is None:
        nb = getattr(args, "wds_epoch_steps", 100000)
    total_steps = nb * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # Resume Logic (Training Resume, not Pretrain Load)
    start_epoch = 0
    if args.resume == "auto":
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith("epoch")]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split("_")[1].split(".")[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f"epoch_{latest_epoch}.pt")
        else:
            args.resume = None

    if args.resume is not None:
        if os.path.isfile(args.resume):
            loc = None if args.gpu is None else f"cuda:{args.gpu}"
            checkpoint = torch.load(args.resume, map_location=loc, weights_only=False) if loc else torch.load(args.resume, weights_only=False)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            sd_img2text = checkpoint["state_dict_img2text"]
            sd_reason_projector = checkpoint.get("state_dict_reason_projector", None)
            
            # Handle DDP
            if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            if not args.distributed and next(iter(sd_img2text.items()))[0].startswith("module"):
                sd_img2text = {k[len("module.") :]: v for k, v in sd_img2text.items()}
            if (
                sd_reason_projector is not None
                and not args.distributed
                and next(iter(sd_reason_projector.items()))[0].startswith("module")
            ):
                sd_reason_projector = {k[len("module.") :]: v for k, v in sd_reason_projector.items()}

            model.load_state_dict(sd, strict=False)
            img2text.load_state_dict(sd_img2text, strict=True)
            if reasoning_modules is not None and sd_reason_projector is not None:
                reasoning_modules["projector"].load_state_dict(sd_reason_projector, strict=True)
                logging.info("=> loaded reasoning projector state from resume checkpoint")
            elif reasoning_modules is not None and is_master(args):
                logging.warning(
                    "Resume checkpoint does not contain state_dict_reason_projector; "
                    "reasoning projector will keep its warm-start initialization."
                )
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as exc:
                    logging.warning(f"Skipping optimizer state restore due to mismatch: {exc}")
            logging.info(f"=> loaded RESUME checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True
    cudnn.deterministic = False

    args.save_logs = (args.logs is not None and args.logs != "" and args.logs.lower() != "none") and (
        (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        logging.debug("Starting wandb.")
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        wandb.save(params_file)

    # Train Loop
    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f"Start epoch {epoch}")

        # Train and save at 25%, 50%, 75% for single epoch training
        if args.epochs == 1:
            train(
                model,
                img2text,
                data,
                epoch,
                optimizer,
                scaler,
                scheduler,
                args,
                writer,
                save_at_percentages=[0.25, 0.5, 0.75],
                reasoning_modules=reasoning_modules,
            )
        else:
            train(
                model,
                img2text,
                data,
                epoch,
                optimizer,
                scaler,
                scheduler,
                args,
                writer,
                reasoning_modules=reasoning_modules,
            )

        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0):
                # For single epoch, also save final checkpoint
                checkpoint_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt")
                reason_projector_state = None
                if reasoning_modules is not None:
                    reason_projector_state = reasoning_modules["projector"].state_dict()
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "state_dict_reason_projector": reason_projector_state,
                        "optimizer": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                # Also save LoRA-only weights (all LoRA params, visual + text)
                lora_state_dict = get_lora_state_dict(model, text_only=False)
                if lora_state_dict:
                    lora_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_lora.pt")
                    torch.save(lora_state_dict, lora_path)
                    logging.info(f"Saved LoRA-only weights ({len(lora_state_dict)} params) to {lora_path}")
    
    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish()


def main():
    args = parse_args()

    if args.name is None:
        args.name = (
            f"lr={args.lr}_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}"
        )
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print("Error. Experiment already exists. Use --name to specify a new experiment.")
        return -1

    assert args.precision in ["amp", "fp16", "fp32"]

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ""
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    torch.multiprocessing.set_start_method("spawn")

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node
        
        import copy
        import types
        args_clean = types.SimpleNamespace()
        for key, value in vars(args).items():
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
                if isinstance(value, (list, tuple)):
                    if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
                        setattr(args_clean, key, value)
                elif isinstance(value, dict):
                     # Simplified dict check
                     setattr(args_clean, key, value)
                else:
                    setattr(args_clean, key, value)
        
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args_clean))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()
