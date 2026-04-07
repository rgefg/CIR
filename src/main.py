# Copyright 2022 Google LLC
# Licensed under the Apache License, Version 2.0

import os
import time
import logging
import random
from time import gmtime, strftime
from pathlib import Path
import json
import wandb
import torch
import numpy as np
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from third_party.open_clip.scheduler import cosine_lr
from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, Phi, LoRAMultiheadAttention, LoRALinear as ModelLoRALinear
from trainer import train
from data import get_data, CIRR, FashionIQ, GeneCISDataset
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32, ModuleParamEMA, use_ema_weights

import math
import torch.nn as nn
import copy
import types
import re


def _safe_torch_load(path, map_location=None, weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=weights_only)


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

    def forward(self, x):
        out = self.base(x)
        x_d = self.dropout(x)
        A = self.A.to(dtype=x_d.dtype)
        B = self.B.to(dtype=x_d.dtype)
        lora = (x_d @ A.t()) @ B.t()
        return out + (self.scaling * lora).to(dtype=out.dtype)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _is_lora_linear(module: nn.Module) -> bool:
    return isinstance(module, (LoRALinear, ModelLoRALinear))


def _clone_multihead_without_lora(module: nn.MultiheadAttention) -> nn.MultiheadAttention:
    if not isinstance(module, (nn.MultiheadAttention, LoRAMultiheadAttention)):
        raise TypeError(f"Expected MultiheadAttention, got {type(module)}")

    cloned = nn.MultiheadAttention(
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        dropout=module.dropout,
        bias=module.in_proj_bias is not None,
        add_bias_kv=module.bias_k is not None,
        add_zero_attn=module.add_zero_attn,
        kdim=module.kdim,
        vdim=module.vdim,
        batch_first=module.batch_first,
    )

    src_out_proj = module.out_proj.base if _is_lora_linear(module.out_proj) else module.out_proj
    with torch.no_grad():
        cloned.in_proj_weight.copy_(module.in_proj_weight.detach())
        if cloned.in_proj_bias is not None and module.in_proj_bias is not None:
            cloned.in_proj_bias.copy_(module.in_proj_bias.detach())
        if cloned.bias_k is not None and module.bias_k is not None:
            cloned.bias_k.copy_(module.bias_k.detach())
        if cloned.bias_v is not None and module.bias_v is not None:
            cloned.bias_v.copy_(module.bias_v.detach())
        cloned.out_proj.weight.copy_(src_out_proj.weight.detach())
        if cloned.out_proj.bias is not None and src_out_proj.bias is not None:
            cloned.out_proj.bias.copy_(src_out_proj.bias.detach())
    return cloned


def _clone_module_without_lora(module: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(module)
    for name, child in list(cloned.named_children()):
        if _is_lora_linear(child):
            setattr(cloned, name, copy.deepcopy(child.base))
        elif isinstance(child, LoRAMultiheadAttention):
            setattr(cloned, name, _clone_multihead_without_lora(child))
        else:
            setattr(cloned, name, _clone_module_without_lora(child))
    return cloned


def _copy_multihead_base_weights(dst_module: nn.Module, src_module: nn.Module):
    dst_base = _clone_multihead_without_lora(dst_module) if isinstance(dst_module, LoRAMultiheadAttention) else dst_module
    src_base = _clone_multihead_without_lora(src_module) if isinstance(src_module, LoRAMultiheadAttention) else src_module
    if not isinstance(dst_base, nn.MultiheadAttention) or not isinstance(src_base, nn.MultiheadAttention):
        raise TypeError(f"Expected MultiheadAttention pair, got {type(dst_base)} and {type(src_base)}")

    src_out_proj = src_base.out_proj
    dst_out_proj = dst_module.out_proj.base if isinstance(dst_module, LoRAMultiheadAttention) and _is_lora_linear(dst_module.out_proj) else dst_base.out_proj
    with torch.no_grad():
        dst_base.in_proj_weight.copy_(src_base.in_proj_weight.detach())
        if dst_base.in_proj_bias is not None and src_base.in_proj_bias is not None:
            dst_base.in_proj_bias.copy_(src_base.in_proj_bias.detach())
        if dst_base.bias_k is not None and src_base.bias_k is not None:
            dst_base.bias_k.copy_(src_base.bias_k.detach())
        if dst_base.bias_v is not None and src_base.bias_v is not None:
            dst_base.bias_v.copy_(src_base.bias_v.detach())
        dst_out_proj.weight.copy_(src_out_proj.weight.detach())
        if dst_out_proj.bias is not None and src_out_proj.bias is not None:
            dst_out_proj.bias.copy_(src_out_proj.bias.detach())


def _copy_module_base_weights(dst_module: nn.Module, src_module: nn.Module):
    if _is_lora_linear(dst_module):
        dst_module = dst_module.base
    if _is_lora_linear(src_module):
        src_module = src_module.base
    if isinstance(dst_module, (nn.MultiheadAttention, LoRAMultiheadAttention)) or isinstance(src_module, (nn.MultiheadAttention, LoRAMultiheadAttention)):
        _copy_multihead_base_weights(dst_module, src_module)
        return

    dst_children = dict(dst_module.named_children())
    src_children = dict(src_module.named_children())

    if not dst_children and not src_children:
        dst_module.load_state_dict(src_module.state_dict(), strict=True)
        return

    if set(dst_children.keys()) != set(src_children.keys()):
        raise RuntimeError(
            f"Text branch structure mismatch while copying base weights: "
            f"dst={sorted(dst_children.keys())} src={sorted(src_children.keys())}"
        )

    for name in dst_children.keys():
        _copy_module_base_weights(dst_children[name], src_children[name])


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


def _tie_parameter(module: nn.Module, name: str, shared_param: nn.Parameter):
    setattr(module, name, shared_param)


def _tie_lora_b_parameters(retr_module: nn.Module, geo_module: nn.Module, prefix: str, excluded_geo_names: set):
    if retr_module is None or geo_module is None:
        return

    if isinstance(retr_module, LoRAMultiheadAttention) and isinstance(geo_module, LoRAMultiheadAttention):
        for proj_name in ["q_proj_lora", "k_proj_lora", "v_proj_lora"]:
            retr_proj = getattr(retr_module, proj_name, None)
            geo_proj = getattr(geo_module, proj_name, None)
            if retr_proj is None or geo_proj is None or not hasattr(retr_proj, "B") or not hasattr(geo_proj, "B"):
                continue
            if retr_proj.B.shape != geo_proj.B.shape:
                raise RuntimeError(
                    f"Shared-B shape mismatch at {prefix}.{proj_name}: "
                    f"{tuple(retr_proj.B.shape)} vs {tuple(geo_proj.B.shape)}"
                )
            _tie_parameter(geo_proj, "B", retr_proj.B)
            excluded_geo_names.add(f"{prefix}.{proj_name}.B")

        retr_out = getattr(retr_module, "out_proj", None)
        geo_out = getattr(geo_module, "out_proj", None)
        if _is_lora_linear(retr_out) and _is_lora_linear(geo_out):
            if retr_out.B.shape != geo_out.B.shape:
                raise RuntimeError(
                    f"Shared-B shape mismatch at {prefix}.out_proj: "
                    f"{tuple(retr_out.B.shape)} vs {tuple(geo_out.B.shape)}"
                )
            _tie_parameter(geo_out, "B", retr_out.B)
            excluded_geo_names.add(f"{prefix}.out_proj.B")
        return

    if _is_lora_linear(retr_module) and _is_lora_linear(geo_module):
        if retr_module.B.shape != geo_module.B.shape:
            raise RuntimeError(
                f"Shared-B shape mismatch at {prefix}: "
                f"{tuple(retr_module.B.shape)} vs {tuple(geo_module.B.shape)}"
            )
        _tie_parameter(geo_module, "B", retr_module.B)
        excluded_geo_names.add(f"{prefix}.B")


_RESBLOCK_RE = re.compile(r"(^|\.)resblocks\.(\d+)(\.|$)")


def _text_resblock_index(module_name: str):
    match = _RESBLOCK_RE.search(module_name)
    if match is None:
        return None
    return int(match.group(2))


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


def freeze_clip_except_lora_and_logit_scale(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False
    # LoRA params are named *.A, *.B
    for n, p in model.named_parameters():
        if n.endswith(".A") or n.endswith(".B"):
            p.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def freeze_module_except_lora(module: nn.Module):
    for _, p in module.named_parameters():
        p.requires_grad = False
    for n, p in module.named_parameters():
        if n.endswith(".A") or n.endswith(".B"):
            p.requires_grad = True


def get_lora_state_dict(module: nn.Module, text_only: bool = False) -> dict:
    state = {}
    for name, param in module.named_parameters():
        if not (name.endswith(".A") or name.endswith(".B")):
            continue
        normalized_name = name[len("module."):] if name.startswith("module.") else name
        if text_only and normalized_name.startswith("visual."):
            continue
        state[name] = param.data.clone()
    return state


class TextEncoderBranch(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        base = clip_model.module if hasattr(clip_model, "module") else clip_model
        self.transformer = _clone_module_without_lora(base.transformer)
        self.token_embedding = _clone_module_without_lora(base.token_embedding)
        self.positional_embedding = nn.Parameter(base.positional_embedding.detach().clone())
        self.ln_final = _clone_module_without_lora(base.ln_final)
        self.text_projection = nn.Parameter(base.text_projection.detach().clone())
        self.end_id = int(base.end_id)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        collect_ind = text == self.end_id
        collect_ind = collect_ind.nonzero()[:, 1]
        x = x[torch.arange(x.size(0), device=x.device), collect_ind] @ self.text_projection
        return x


def copy_text_branch_weights_from_clip(text_branch: nn.Module, clip_model: nn.Module):
    branch = text_branch.module if hasattr(text_branch, "module") else text_branch
    base = clip_model.module if hasattr(clip_model, "module") else clip_model

    _copy_module_base_weights(branch.transformer, base.transformer)
    _copy_module_base_weights(branch.token_embedding, base.token_embedding)
    _copy_module_base_weights(branch.ln_final, base.ln_final)
    with torch.no_grad():
        branch.positional_embedding.copy_(base.positional_embedding.detach())
        branch.text_projection.copy_(base.text_projection.detach())
    branch.end_id = int(base.end_id)


def seed_everything(base_seed: int, rank_offset: int = 0):
    seed = int(base_seed) + int(rank_offset)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


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

    rank_offset = args.rank if args.rank is not None else 0
    effective_seed = seed_everything(args.seed, rank_offset=rank_offset)
    if args.deterministic_train:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    logging.info(
        f"Random seed initialized: base_seed={args.seed} rank_offset={rank_offset} effective_seed={effective_seed} "
        f"deterministic_train={args.deterministic_train}"
    )

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
        
        checkpoint = _safe_torch_load(args.pic2word_pretrained, map_location='cpu', weights_only=False)
        
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
    img2text_arch = getattr(args, "img2text_arch", "im2text")
    if img2text_arch == "phi":
        phi_hidden = getattr(args, "middle_dim", 2048)
        img2text = Phi(
            input_dim=model.embed_dim,
            hidden_dim=phi_hidden,
            output_dim=model.token_embedding.weight.shape[1],
            dropout=getattr(args, "droprate", 0.5),
        )
        if is_master(args):
            logging.info(f"Using Phi (SEARLE-style) img2text: {model.embed_dim} -> {phi_hidden} -> {model.token_embedding.weight.shape[1]}")
    else:
        img2text = IM2TEXT(
            embed_dim=model.embed_dim,
            middle_dim=args.middle_dim,
            output_dim=model.token_embedding.weight.shape[1],
            n_layer=args.n_layer,
            dropout=getattr(args, "droprate", 0.1),
        )

    # =========================================================================
    # 2b. Load SEARLE / external img2text pretrained weights
    # =========================================================================
    img2text_pretrained = getattr(args, "img2text_pretrained", None)
    if img2text_pretrained:
        if is_master(args):
            logging.info(f"Loading img2text pretrained weights from: {img2text_pretrained}")
        ckpt = _safe_torch_load(img2text_pretrained, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and "Phi" in ckpt:
            sd = ckpt["Phi"]
        elif isinstance(ckpt, dict) and "state_dict_img2text" in ckpt:
            sd = ckpt["state_dict_img2text"]
        else:
            sd = ckpt
        if next(iter(sd.keys()), "").startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        msg = img2text.load_state_dict(sd, strict=False)
        if is_master(args):
            logging.info(f"img2text pretrained load: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")

    # =========================================================================
    # 3. Load img2text weights (CRITICAL FIX)
    # =========================================================================
    if args.pic2word_pretrained:
        if is_master(args):
            logging.info("=" * 80)
            logging.info("🔄 Loading img2text weights from pic2word checkpoint...")
            logging.info("=" * 80)
        
        checkpoint = _safe_torch_load(args.pic2word_pretrained, map_location='cpu', weights_only=False)
        
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
            if img2text_pretrained:
                logging.warning(
                    "ℹ️ No pic2word checkpoint provided. Using external img2text pretrained weights "
                    f"from: {img2text_pretrained}"
                )
            else:
                logging.warning("⚠️ WARNING: No pic2word or img2text pretrained path provided. img2text is INITIALIZED FROM SCRATCH (Random)!")
            logging.warning("=" * 80)

    # =========================================================================
    # 4. Precision & LoRA Setup
    # =========================================================================
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    # ---- Enable LoRA on CLIP (paper-consistent) ----
    if not getattr(args, "no_lora", False):
        apply_lora_to_linear_layers(model, r=getattr(args, "lora_r", 64), alpha=getattr(args, "lora_alpha", 16), dropout=getattr(args, "lora_dropout", 0.0))
    
    freeze_clip_except_lora_and_logit_scale(model)
    geo_text_model = None
    shared_b_geo_exclude_names = set()
    args.shared_b_param_names = []
    if float(getattr(args, "geo_weight", 0.0)) > 0.0:
        geo_text_model = TextEncoderBranch(model)
        if not getattr(args, "no_lora", False):
            apply_lora_to_linear_layers(
                geo_text_model,
                r=getattr(args, "geo_lora_r", 64),
                alpha=getattr(args, "geo_lora_alpha", 16),
                dropout=getattr(args, "geo_lora_dropout", 0.0),
            )
        freeze_module_except_lora(geo_text_model)
        if getattr(args, "shared_b_lora", False):
            shared_b_geo_exclude_names = tie_shared_b_between_text_branches(
                model,
                geo_text_model,
                num_layers=int(getattr(args, "shared_b_num_layers", 6)),
            )
            args.shared_b_param_names = sorted(shared_b_geo_exclude_names)
            if is_master(args):
                logging.info(
                    f"✅ Enabled Shared-B LoRA on text encoder: tied {len(shared_b_geo_exclude_names)} geo B tensors "
                    f"across shallow blocks [0, {int(getattr(args, 'shared_b_num_layers', 6)) - 1}]; "
                    "geo optimizer will update only task-specific A."
                )
                if getattr(args, "shared_b_retrieval_only_update", False):
                    logging.info(
                        "✅ Shared-B retrieval-only update is enabled: geo branch uses retrieval B in forward, "
                        "but shared B gradients are restored to retrieval-only values before optimizer step."
                    )
    
    # ============================================================
    # 🔒 Logit Scale 修复选项
    # ============================================================
    # 如果logit_scale被错误初始化或加载，可以强制重置
    if getattr(args, "reset_logit_scale", False):
        with torch.no_grad():
            model.logit_scale.data.fill_(np.log(1 / 0.07))  # 标准CLIP初始化
        if is_master(args):
            logging.info("🔧 Reset logit_scale to standard CLIP value: log(1/0.07) ≈ 2.659")
    
    # 可选：锁定logit_scale（用于调试）
    if getattr(args, "freeze_logit_scale", False):
        model.logit_scale.requires_grad = False
        if is_master(args):
            logging.info("🔒 Frozen logit_scale (requires_grad=False) for debugging")
    # ============================================================
    
    # img2text always trainable
    for p in img2text.parameters():
        p.requires_grad = True
    
    # Log model parameter statistics
    if is_master(args):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        img2text_params = sum(p.numel() for p in img2text.parameters())
        geo_trainable_params = (
            sum(p.numel() for p in geo_text_model.parameters() if p.requires_grad)
            if geo_text_model is not None
            else 0
        )
        logging.info("=" * 80)
        logging.info("📊 Model Parameter Statistics:")
        logging.info(f"   CLIP total parameters: {total_params:,}")
        logging.info(f"   CLIP trainable parameters: {trainable_params:,} ({100.*trainable_params/total_params:.2f}%)")
        logging.info(f"   img2text parameters: {img2text_params:,}")
        if geo_text_model is not None:
            logging.info(f"   geo text branch trainable parameters: {geo_trainable_params:,}")
        
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
        if geo_text_model is not None:
            geo_text_model.cuda(args.gpu)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            if geo_text_model is not None:
                convert_weights(geo_text_model)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, device_ids=[args.gpu], find_unused_parameters=False)
            if geo_text_model is not None:
                geo_text_model = torch.nn.parallel.DistributedDataParallel(
                    geo_text_model, device_ids=[args.gpu], find_unused_parameters=False
                )
        
        # ... (DP logic omitted for brevity, Distributed is preferred)

    # Data
    data = get_data(args, (preprocess_train, preprocess_val))
    args.actual_samples_per_microbatch = args.batch_size * args.world_size
    args.cirr_val_eval_log_path = os.path.join(args.logs, args.name, "cirr_val_eval.log")
    args.multidataset_eval_log_path = os.path.join(args.logs, args.name, "multidataset_eval.log")
    cirr_eval_every = int(getattr(args, "cirr_val_eval_every", 0))
    if cirr_eval_every > 0:
        try:
            root_project = os.path.join(get_project_root(), "data")
            source_dataset = CIRR(transforms=preprocess_val, root=root_project)
            target_dataset = CIRR(transforms=preprocess_val, root=root_project, mode="imgs")
            data["cirr_val_query_loader"] = DataLoader(
                source_dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.workers),
                pin_memory=True,
                drop_last=False,
            )
            data["cirr_val_target_loader"] = DataLoader(
                target_dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.workers),
                pin_memory=True,
                drop_last=False,
            )
            if is_master(args):
                logging.info(
                    f"✅ Periodic CIRR-val eval enabled: every {cirr_eval_every} steps, "
                    f"query={len(source_dataset)}, gallery={len(target_dataset)}, "
                    f"log_file={args.cirr_val_eval_log_path}"
                )
        except Exception as exc:
            if is_master(args):
                logging.warning(f"⚠️ Failed to initialize periodic CIRR-val eval; disabling it. Error: {exc}")
            args.cirr_val_eval_every = 0

    multidataset_eval_every = int(getattr(args, "multidataset_eval_every", 0))
    if multidataset_eval_every > 0:
        try:
            root_project = os.path.join(get_project_root(), "data")
            eval_batch_size = int(getattr(args, "multidataset_eval_batch_size", args.batch_size))
            eval_workers = int(getattr(args, "multidataset_eval_workers", args.workers))

            fashion_eval_loaders = {}
            for cloth in ["dress", "shirt", "toptee"]:
                source_dataset = FashionIQ(
                    cloth=cloth,
                    transforms=preprocess_val,
                    root=root_project,
                    is_return_target_path=True,
                )
                target_dataset = FashionIQ(
                    cloth=cloth,
                    transforms=preprocess_val,
                    root=root_project,
                    mode="imgs",
                )
                fashion_eval_loaders[cloth] = {
                    "query": DataLoader(
                        source_dataset,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        num_workers=eval_workers,
                        pin_memory=True,
                        drop_last=False,
                    ),
                    "target": DataLoader(
                        target_dataset,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        num_workers=eval_workers,
                        pin_memory=True,
                        drop_last=False,
                    ),
                }

            genecis_eval_loaders = {}
            genecis_vg_root = os.path.join(root_project, "genecis", "VG_100K")
            genecis_coco_root = os.path.join(root_project, "coco", "val2017")
            genecis_json_root = "/data2/mingyu/genecis/genecis"
            for task in ["focus_attribute", "change_attribute", "focus_object", "change_object"]:
                img_root = genecis_coco_root if "object" in task else genecis_vg_root
                dataset = GeneCISDataset(
                    data_root=img_root,
                    json_root=genecis_json_root,
                    task=task,
                    transforms=preprocess_val,
                    tokenizer=None,
                )
                genecis_eval_loaders[task] = DataLoader(
                    dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=eval_workers,
                    pin_memory=True,
                    drop_last=False,
                )

            data["fashion_eval_loaders"] = fashion_eval_loaders
            data["genecis_eval_loaders"] = genecis_eval_loaders
            if is_master(args):
                logging.info(
                    "✅ Periodic FashionIQ/GeneCIS eval enabled: "
                    f"every {multidataset_eval_every} steps, "
                    f"fashion_categories={list(fashion_eval_loaders.keys())}, "
                    f"genecis_tasks={list(genecis_eval_loaders.keys())}, "
                    f"batch_size={eval_batch_size}, workers={eval_workers}, "
                    f"log_file={args.multidataset_eval_log_path}"
                )
        except Exception as exc:
            if is_master(args):
                logging.warning(
                    "⚠️ Failed to initialize periodic FashionIQ/GeneCIS eval; disabling it. "
                    f"Error: {exc}"
                )
            args.multidataset_eval_every = 0

    # ---- Optimizer ----
    def exclude(n):
        return ("bn" in n) or ("ln" in n) or ("bias" in n) or ("logit_scale" in n)

    def include(n):
        return not exclude(n)

    def build_param_groups(named_parameters, weight_decay, exclude_name_set=None):
        exclude_name_set = set(exclude_name_set or [])
        gain_or_bias_params = []
        rest_params = []
        seen = set()
        for n, p in named_parameters:
            norm_name = n[len("module."):] if n.startswith("module.") else n
            if norm_name in exclude_name_set or (not p.requires_grad):
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            if exclude(norm_name):
                gain_or_bias_params.append(p)
            else:
                rest_params.append(p)
        groups = []
        if gain_or_bias_params:
            groups.append({"params": gain_or_bias_params, "weight_decay": 0.0})
        if rest_params:
            groups.append({"params": rest_params, "weight_decay": weight_decay})
        return groups

    retrieval_named_parameters = []
    retrieval_named_parameters += list((img2text.module if (args.distributed or args.dp) else img2text).named_parameters())
    retrieval_named_parameters += list((model.module if (args.distributed or args.dp) else model).named_parameters())
    geo_named_parameters = []
    if geo_text_model is not None:
        geo_named_parameters += list((geo_text_model.module if (args.distributed or args.dp) else geo_text_model).named_parameters())

    optimizer = optim.AdamW(
        build_param_groups(retrieval_named_parameters, args.wd),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    geo_optimizer = None
    geo_param_groups = build_param_groups(
        geo_named_parameters,
        args.geo_wd,
        exclude_name_set=shared_b_geo_exclude_names,
    )
    if geo_param_groups:
        geo_optimizer = optim.AdamW(
            geo_param_groups,
            lr=args.geo_lr,
            betas=(args.geo_beta1, args.geo_beta2),
            eps=args.geo_eps,
        )

    # ---- Scheduler ----
    nb = getattr(data["train"].dataloader, "num_batches", None)
    if nb is None:
        nb = getattr(args, "wds_epoch_steps", 100000)
    total_steps = nb * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    geo_scheduler = (
        cosine_lr(geo_optimizer, args.geo_lr, args.geo_warmup, total_steps)
        if geo_optimizer is not None
        else None
    )

    # Keep AMP state fully separated so geo overflows/backoff cannot perturb
    # retrieval branch scaling behavior.
    retrieval_scaler = (
        GradScaler(
            init_scale=float(args.amp_init_scale),
            growth_factor=float(args.amp_growth_factor),
            backoff_factor=float(args.amp_backoff_factor),
            growth_interval=int(args.amp_growth_interval),
        )
        if args.precision == "amp"
        else None
    )
    geo_scaler = (
        GradScaler(
            init_scale=float(args.geo_amp_init_scale),
            growth_factor=float(args.geo_amp_growth_factor),
            backoff_factor=float(args.geo_amp_backoff_factor),
            growth_interval=int(args.geo_amp_growth_interval),
        )
        if (args.precision == "amp" and geo_optimizer is not None)
        else None
    )

    retrieval_model_ema = None
    img2text_ema = None
    geo_text_ema = None
    if float(getattr(args, "retrieval_ema_decay", 0.0)) > 0.0:
        retrieval_model_ema = ModuleParamEMA(model, args.retrieval_ema_decay)
        img2text_ema = ModuleParamEMA(img2text, args.retrieval_ema_decay)
        logging.info(f"Enabled retrieval EMA with decay={args.retrieval_ema_decay}.")
    if geo_text_model is not None and float(getattr(args, "geo_ema_decay", 0.0)) > 0.0:
        geo_text_ema = ModuleParamEMA(geo_text_model, args.geo_ema_decay)
        logging.info(f"Enabled geo EMA with decay={args.geo_ema_decay}.")

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
            checkpoint = (
                _safe_torch_load(args.resume, map_location=loc, weights_only=False)
                if loc
                else _safe_torch_load(args.resume, weights_only=False)
            )
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            sd_img2text = checkpoint["state_dict_img2text"]
            
            # Handle DDP
            if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            if not args.distributed and next(iter(sd_img2text.items()))[0].startswith("module"):
                sd_img2text = {k[len("module.") :]: v for k, v in sd_img2text.items()}

            model.load_state_dict(sd, strict=False)
            img2text.load_state_dict(sd_img2text, strict=True)
            if geo_text_model is not None and checkpoint.get("state_dict_geo_text") is not None:
                sd_geo = checkpoint["state_dict_geo_text"]
                if not args.distributed and next(iter(sd_geo.items()))[0].startswith("module"):
                    sd_geo = {k[len("module.") :]: v for k, v in sd_geo.items()}
                geo_text_model.load_state_dict(sd_geo, strict=False)
            elif geo_text_model is not None:
                copy_text_branch_weights_from_clip(geo_text_model, model)
                logging.info("Initialized geo text branch from retrieval text tower (checkpoint had no geo branch state).")

            retrieval_optimizer_state = checkpoint.get("optimizer_retrieval")
            if retrieval_optimizer_state is None:
                retrieval_optimizer_state = checkpoint.get("optimizer")
            if optimizer is not None and retrieval_optimizer_state is not None:
                try:
                    optimizer.load_state_dict(retrieval_optimizer_state)
                except ValueError as exc:
                    logging.warning(f"Skipping retrieval optimizer state due to param-group mismatch: {exc}")

            geo_optimizer_state = checkpoint.get("optimizer_geo")
            if geo_optimizer is not None and geo_optimizer_state is not None:
                try:
                    geo_optimizer.load_state_dict(geo_optimizer_state)
                except ValueError as exc:
                    logging.warning(f"Skipping geo optimizer state due to param-group mismatch: {exc}")
            elif geo_optimizer is not None and checkpoint.get("optimizer") is not None and checkpoint.get("optimizer_retrieval") is None:
                logging.warning(
                    "Skipping legacy combined optimizer state for geo branch because retrieval/geo optimizers are now split."
                )
            if retrieval_model_ema is not None:
                retrieval_model_ema.load_state_dict(checkpoint.get("ema_retrieval_model"), model)
            if img2text_ema is not None:
                img2text_ema.load_state_dict(checkpoint.get("ema_img2text"), img2text)
            if geo_text_ema is not None:
                geo_text_ema.load_state_dict(checkpoint.get("ema_geo_text"), geo_text_model)
            logging.info(f"=> loaded RESUME checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

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

        train(
            model,
            img2text,
            data,
            epoch,
            optimizer,
            retrieval_scaler,
            scheduler,
            args,
            writer,
            geo_text_model=geo_text_model,
            geo_optimizer=geo_optimizer,
            geo_scheduler=geo_scheduler,
            geo_scaler=geo_scaler,
            retrieval_model_ema=retrieval_model_ema,
            img2text_ema=img2text_ema,
            geo_text_ema=geo_text_ema,
        )
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            ema_pairs = [
                (model, retrieval_model_ema),
                (img2text, img2text_ema),
                (geo_text_model, geo_text_ema),
            ]

            def _build_checkpoint_payload(epoch_to_save):
                return {
                    "epoch": epoch_to_save,
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

            if (epoch + 1) == args.epochs or (args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0):
                raw_payload = _build_checkpoint_payload(epoch + 1)
                torch.save(raw_payload, os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"))
                if geo_text_model is not None:
                    geo_lora_state_dict = get_lora_state_dict(geo_text_model, text_only=False)
                    if geo_lora_state_dict:
                        geo_lora_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_geo_lora.pt")
                        torch.save(geo_lora_state_dict, geo_lora_path)
                        logging.info(
                            f"Saved geo LoRA-only weights ({len(geo_lora_state_dict)} params) to {geo_lora_path}"
                        )
                if args.ema_save_checkpoints and any(ema is not None for _, ema in ema_pairs):
                    ema_ckpt_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_ema.pt")
                    with use_ema_weights(ema_pairs):
                        ema_payload = _build_checkpoint_payload(epoch + 1)
                        ema_payload["optimizer"] = None
                        ema_payload["optimizer_retrieval"] = None
                        ema_payload["optimizer_geo"] = None
                        ema_payload["ema_checkpoint"] = True
                        torch.save(ema_payload, ema_ckpt_path)
                        if geo_text_model is not None and geo_text_ema is not None:
                            geo_lora_ema = get_lora_state_dict(geo_text_model, text_only=False)
                            if geo_lora_ema:
                                geo_lora_ema_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_geo_lora_ema.pt")
                                torch.save(geo_lora_ema, geo_lora_ema_path)
                                logging.info(
                                    f"Saved geo EMA LoRA-only weights ({len(geo_lora_ema)} params) to {geo_lora_ema_path}"
                                )
                    logging.info(f"Saved EMA checkpoint to {ema_ckpt_path}")
    
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
