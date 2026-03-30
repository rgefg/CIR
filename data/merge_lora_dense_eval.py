#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft.utils.merge_utils import ties


def _load_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _extract_lora_dict(ckpt) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
        lora = {k: v for k, v in sd.items() if k.endswith(".A") or k.endswith(".B")}
        return lora, "full"
    if isinstance(ckpt, dict):
        lora = {k: v for k, v in ckpt.items() if k.endswith(".A") or k.endswith(".B")}
        return lora, "lora_only"
    raise ValueError("Unsupported checkpoint format.")


def _strip_module_prefix(name: str) -> str:
    if name.startswith("module."):
        return name[len("module.") :]
    return name


def _is_text_prefix(prefix: str) -> bool:
    return not _strip_module_prefix(prefix).startswith("visual.")


def _norm_prefix(prefix: str) -> str:
    return _strip_module_prefix(prefix)


def _build_pair_map(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in lora_sd.items():
        if k.endswith(".A"):
            p = k[: -len(".A")]
            pn = _norm_prefix(p)
            d = out.setdefault(pn, {})
            d["A"] = v
            d["A_key"] = p + ".A"
        elif k.endswith(".B"):
            p = k[: -len(".B")]
            pn = _norm_prefix(p)
            d = out.setdefault(pn, {})
            d["B"] = v
            d["B_key"] = p + ".B"
    return out


def _valid_pair(pair: Dict[str, torch.Tensor]) -> bool:
    return ("A" in pair) and ("B" in pair)


def _target_weight_spec(prefix: str) -> Tuple[str, int | None]:
    core = _strip_module_prefix(prefix)
    if core.endswith(".attn.q_proj_lora"):
        base = prefix[: -len(".q_proj_lora")]
        return base + ".in_proj_weight", 0
    if core.endswith(".attn.k_proj_lora"):
        base = prefix[: -len(".k_proj_lora")]
        return base + ".in_proj_weight", 1
    if core.endswith(".attn.v_proj_lora"):
        base = prefix[: -len(".v_proj_lora")]
        return base + ".in_proj_weight", 2
    return prefix + ".base.weight", None


def _apply_dense_delta(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    delta_eff: torch.Tensor,
    pair_base: Dict[str, torch.Tensor],
) -> None:
    base_prefix = pair_base["A_key"][: -len(".A")]
    weight_key, qkv_index = _target_weight_spec(base_prefix)
    if weight_key not in state_dict:
        raise KeyError(f"Target base weight key not found for prefix {prefix}: {weight_key}")
    weight = state_dict[weight_key].float().clone()
    if qkv_index is None:
        if tuple(weight.shape) != tuple(delta_eff.shape):
            raise RuntimeError(
                f"Shape mismatch for {prefix}: target {tuple(weight.shape)} vs delta {tuple(delta_eff.shape)}"
            )
        weight.add_(delta_eff)
    else:
        out_dim = delta_eff.shape[0]
        start = qkv_index * out_dim
        end = start + out_dim
        if end > weight.shape[0] or weight.shape[1] != delta_eff.shape[1]:
            raise RuntimeError(
                f"QKV shape mismatch for {prefix}: target {tuple(weight.shape)} vs delta {tuple(delta_eff.shape)}"
            )
        weight[start:end, :].add_(delta_eff)
    state_dict[weight_key] = weight.to(dtype=state_dict[weight_key].dtype)

    a_key = pair_base["A_key"]
    b_key = pair_base["B_key"]
    if a_key not in state_dict or b_key not in state_dict:
        raise KeyError(f"Missing LoRA tensors for zeroing on prefix {prefix}: {a_key}, {b_key}")
    state_dict[a_key] = torch.zeros_like(state_dict[a_key])
    state_dict[b_key] = torch.zeros_like(state_dict[b_key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-only dense LoRA merge: merge in delta space and write merged delta directly into base weights."
    )
    parser.add_argument("--ckpt-a", type=Path, required=True, help="Checkpoint A path.")
    parser.add_argument("--ckpt-b", type=Path, required=True, help="Checkpoint B path. Must be a full checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output merged checkpoint path.")
    parser.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5], metavar=("WA", "WB"))
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--majority-sign-method", type=str, default="total", choices=["total", "frequency"])
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="delta_ties",
        choices=["delta_ties", "shared_a_ties_b"],
        help="Merge dense deltas directly, or under Shared-A assume A is tied and apply TIES only on effective B.",
    )
    parser.add_argument("--text-only", action="store_true", default=False)
    parser.add_argument("--alpha-a", type=float, default=None)
    parser.add_argument("--rank-a", type=int, default=None)
    parser.add_argument("--alpha-b", type=float, default=None)
    parser.add_argument("--rank-b", type=int, default=None)
    args = parser.parse_args()

    print(f"loading ckpt_a: {args.ckpt_a}", flush=True)
    ckpt_a = _load_checkpoint(args.ckpt_a)
    print(f"loading ckpt_b: {args.ckpt_b}", flush=True)
    ckpt_b = _load_checkpoint(args.ckpt_b)
    if not (isinstance(ckpt_b, dict) and "state_dict" in ckpt_b and isinstance(ckpt_b["state_dict"], dict)):
        raise ValueError("ckpt_b must be a full checkpoint with state_dict for dense eval merge.")

    lora_a, type_a = _extract_lora_dict(ckpt_a)
    lora_b, type_b = _extract_lora_dict(ckpt_b)
    pair_a = _build_pair_map(lora_a)
    pair_b = _build_pair_map(lora_b)
    prefixes = sorted(set(pair_a.keys()) & set(pair_b.keys()))
    if args.text_only:
        prefixes = [p for p in prefixes if _is_text_prefix(p)]

    valid_prefixes: List[str] = []
    dropped = 0
    for p in prefixes:
        pa = pair_a[p]
        pb = pair_b[p]
        if (not _valid_pair(pa)) or (not _valid_pair(pb)):
            dropped += 1
            continue
        aA, aB = pa["A"], pa["B"]
        bA, bB = pb["A"], pb["B"]
        if (aA.ndim != 2) or (aB.ndim != 2) or (bA.ndim != 2) or (bB.ndim != 2):
            dropped += 1
            continue
        if aB.shape[1] != aA.shape[0] or bB.shape[1] != bA.shape[0]:
            dropped += 1
            continue
        if (aB.shape[0] != bB.shape[0]) or (aA.shape[1] != bA.shape[1]):
            dropped += 1
            continue
        valid_prefixes.append(p)

    if not valid_prefixes:
        raise RuntimeError("No valid LoRA pairs found.")

    scale_a = float(args.alpha_a) / float(args.rank_a) if (args.alpha_a is not None and args.rank_a) else 1.0
    scale_b = float(args.alpha_b) / float(args.rank_b) if (args.alpha_b is not None and args.rank_b) else 1.0
    w = torch.tensor(args.weights, dtype=torch.float32)

    out_ckpt = {
        "state_dict": dict(ckpt_b["state_dict"]),
        "state_dict_img2text": ckpt_b.get("state_dict_img2text"),
    }
    for meta_key in ["epoch", "name", "step", "global_step"]:
        if meta_key in ckpt_b:
            out_ckpt[meta_key] = ckpt_b[meta_key]

    out_sd = out_ckpt["state_dict"]
    replaced = 0
    for p in valid_prefixes:
        pa = pair_a[p]
        pb = pair_b[p]
        if args.merge_mode == "shared_a_ties_b":
            a_ref = pb["A"].float()
            if pa["A"].shape != pb["A"].shape:
                raise RuntimeError(
                    f"Shared-A B-TIES requires matching A shapes for {p}: {tuple(pa['A'].shape)} vs {tuple(pb['A'].shape)}"
                )
            if not torch.allclose(pa["A"].float(), pb["A"].float(), rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"Shared-A B-TIES requires equal A tensors for {p}, but they differ.")
            b_a_eff = scale_a * pa["B"].float()
            b_b_eff = scale_b * pb["B"].float()
            b_m_eff = ties(
                [b_a_eff, b_b_eff],
                weights=w,
                density=args.density,
                majority_sign_method=args.majority_sign_method,
            )
            delta_m_eff = b_m_eff @ a_ref
        else:
            delta_a_eff = scale_a * (pa["B"].float() @ pa["A"].float())
            delta_b_eff = scale_b * (pb["B"].float() @ pb["A"].float())
            delta_m_eff = ties(
                [delta_a_eff, delta_b_eff],
                weights=w,
                density=args.density,
                majority_sign_method=args.majority_sign_method,
            )
        _apply_dense_delta(out_sd, p, delta_m_eff, pb)
        replaced += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"saving dense merged checkpoint: {args.output}", flush=True)
    torch.save(out_ckpt, str(args.output))
    print(f"ckpt_a: {args.ckpt_a} ({type_a})")
    print(f"ckpt_b: {args.ckpt_b} ({type_b})")
    print(f"valid_prefixes: {len(valid_prefixes)}")
    print(f"dropped_prefixes: {dropped}")
    print(f"replaced_dense_prefixes: {replaced}")
    print(f"merge_mode: {args.merge_mode}")
    print(f"weights: {args.weights}, density: {args.density}, sign: {args.majority_sign_method}")
    print(f"scope: {'text_only' if args.text_only else 'default'}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
