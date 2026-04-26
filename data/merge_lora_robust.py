#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _load_checkpoint(path: Path):
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
    core = _strip_module_prefix(prefix)
    return not core.startswith("visual.")


def _norm_prefix(prefix: str) -> str:
    return _strip_module_prefix(prefix)


def _build_pair_map(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in lora_sd.items():
        if key.endswith(".A"):
            prefix = key[: -len(".A")]
            norm_prefix = _norm_prefix(prefix)
            pair = out.setdefault(norm_prefix, {})
            pair["A"] = value
            pair["A_key"] = prefix + ".A"
        elif key.endswith(".B"):
            prefix = key[: -len(".B")]
            norm_prefix = _norm_prefix(prefix)
            pair = out.setdefault(norm_prefix, {})
            pair["B"] = value
            pair["B_key"] = prefix + ".B"
    return out


def _valid_pair(pair: Dict[str, torch.Tensor]) -> bool:
    return ("A" in pair) and ("B" in pair)


def _svd_factorize(
    delta: torch.Tensor,
    rank: int,
    out_dtype: torch.dtype,
    compute_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = delta.to(device=compute_device, dtype=torch.float32)
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    use_r = min(rank, s.shape[0])
    s_root = torch.sqrt(torch.clamp(s[:use_r], min=0.0))
    b_small = u[:, :use_r] * s_root.unsqueeze(0)
    a_small = s_root.unsqueeze(1) * vh[:use_r, :]

    if use_r < rank:
        b_full = torch.zeros(delta.shape[0], rank, dtype=b_small.dtype, device=b_small.device)
        a_full = torch.zeros(rank, delta.shape[1], dtype=a_small.dtype, device=a_small.device)
        b_full[:, :use_r] = b_small
        a_full[:use_r, :] = a_small
    else:
        b_full, a_full = b_small, a_small
    return a_full.to(device="cpu", dtype=out_dtype), b_full.to(device="cpu", dtype=out_dtype)


def _prune_by_magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    density = float(max(0.0, min(1.0, density)))
    if density >= 1.0:
        return tensor
    if density <= 0.0:
        return torch.zeros_like(tensor)
    flat = tensor.abs().reshape(-1)
    keep = max(1, int(round(flat.numel() * density)))
    threshold = torch.topk(flat, keep, largest=True, sorted=False).values.min()
    return torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))


def _adaptive_delta_scale(a_matrix: torch.Tensor, eps: float) -> float:
    a_stat = float(a_matrix.abs().mean().item())
    return 1.0 / max(a_stat, eps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RobustMerge-style LoRA merge: magnitude pruning + adaptive complementary scaling + normalization."
    )
    parser.add_argument("--ckpt-a", type=Path, required=True, help="Checkpoint A path.")
    parser.add_argument("--ckpt-b", type=Path, required=True, help="Checkpoint B path.")
    parser.add_argument("--output", type=Path, required=True, help="Output merged checkpoint path.")
    parser.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5], metavar=("WA", "WB"))
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--text-only", action="store_true", default=False, help="Merge only text encoder LoRA pairs.")
    parser.add_argument("--all-lora", action="store_true", default=False, help="Merge both text and visual LoRA pairs.")
    parser.add_argument("--base", type=str, default="b", choices=["a", "b"], help="Output structure base checkpoint.")
    parser.add_argument("--alpha-a", type=float, default=None, help="LoRA alpha for checkpoint A (optional).")
    parser.add_argument("--rank-a", type=int, default=None, help="LoRA rank for checkpoint A (optional).")
    parser.add_argument("--alpha-b", type=float, default=None, help="LoRA alpha for checkpoint B (optional).")
    parser.add_argument("--rank-b", type=int, default=None, help="LoRA rank for checkpoint B (optional).")
    parser.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon for adaptive scaling.")
    parser.add_argument(
        "--svd-device",
        type=str,
        default="cpu",
        help="Device used for dense delta merge/SVD, e.g. cpu or cuda:4. No CUDA remapping is applied.",
    )
    args = parser.parse_args()
    compute_device = torch.device(args.svd_device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.svd_device}, but CUDA is not available.")

    ckpt_a = _load_checkpoint(args.ckpt_a)
    ckpt_b = _load_checkpoint(args.ckpt_b)
    lora_a, type_a = _extract_lora_dict(ckpt_a)
    lora_b, type_b = _extract_lora_dict(ckpt_b)

    pair_a = _build_pair_map(lora_a)
    pair_b = _build_pair_map(lora_b)
    prefixes = sorted(set(pair_a.keys()) & set(pair_b.keys()))
    only_a = sorted(set(pair_a.keys()) - set(pair_b.keys()))
    only_b = sorted(set(pair_b.keys()) - set(pair_a.keys()))

    if not args.all_lora:
        prefixes = [prefix for prefix in prefixes if _is_text_prefix(prefix)] if args.text_only else prefixes

    valid_prefixes: List[str] = []
    dropped_missing_ab = 0
    dropped_bad_shape = 0
    for prefix in prefixes:
        pa = pair_a[prefix]
        pb = pair_b[prefix]
        if (not _valid_pair(pa)) or (not _valid_pair(pb)):
            dropped_missing_ab += 1
            continue
        aA, aB = pa["A"], pa["B"]
        bA, bB = pb["A"], pb["B"]
        if (aA.ndim != 2) or (aB.ndim != 2) or (bA.ndim != 2) or (bB.ndim != 2):
            dropped_bad_shape += 1
            continue
        if aB.shape[1] != aA.shape[0] or bB.shape[1] != bA.shape[0]:
            dropped_bad_shape += 1
            continue
        if (aB.shape[0] != bB.shape[0]) or (aA.shape[1] != bA.shape[1]):
            dropped_bad_shape += 1
            continue
        valid_prefixes.append(prefix)

    if not valid_prefixes:
        raise RuntimeError("No valid LoRA module pairs found to merge.")

    scale_a = float(args.alpha_a) / float(args.rank_a) if (args.alpha_a is not None and args.rank_a) else 1.0
    scale_b = float(args.alpha_b) / float(args.rank_b) if (args.alpha_b is not None and args.rank_b) else 1.0
    base_scale = scale_b if args.base == "b" else scale_a
    if abs(base_scale) < 1e-12:
        raise ValueError("Base LoRA scale is zero. Check alpha/rank arguments.")

    weight_a, weight_b = [float(x) for x in args.weights]
    merged_ab: Dict[str, torch.Tensor] = {}
    scale_history_a: List[float] = []
    scale_history_b: List[float] = []

    for prefix in valid_prefixes:
        aA = pair_a[prefix]["A"].to(device=compute_device, dtype=torch.float32)
        aB = pair_a[prefix]["B"].to(device=compute_device, dtype=torch.float32)
        bA = pair_b[prefix]["A"].to(device=compute_device, dtype=torch.float32)
        bB = pair_b[prefix]["B"].to(device=compute_device, dtype=torch.float32)

        delta_a_eff = scale_a * (aB @ aA)
        delta_b_eff = scale_b * (bB @ bA)

        delta_a_eff = _prune_by_magnitude(delta_a_eff, args.density)
        delta_b_eff = _prune_by_magnitude(delta_b_eff, args.density)

        raw_scale_a = _adaptive_delta_scale(aA, args.eps)
        raw_scale_b = _adaptive_delta_scale(bA, args.eps)
        raw_mean = (raw_scale_a + raw_scale_b) / 2.0
        norm_scale_a = raw_scale_a / max(raw_mean, args.eps)
        norm_scale_b = raw_scale_b / max(raw_mean, args.eps)
        scale_history_a.append(norm_scale_a)
        scale_history_b.append(norm_scale_b)

        delta_m_eff = (weight_a * norm_scale_a * delta_a_eff) + (weight_b * norm_scale_b * delta_b_eff)
        delta_m = delta_m_eff / base_scale

        rank_target = pair_b[prefix]["A"].shape[0] if args.base == "b" else pair_a[prefix]["A"].shape[0]
        out_dtype = pair_b[prefix]["A"].dtype if args.base == "b" else pair_a[prefix]["A"].dtype
        mA, mB = _svd_factorize(delta_m, rank=rank_target, out_dtype=out_dtype, compute_device=compute_device)

        if args.base == "b":
            key_a = pair_b[prefix].get("A_key", prefix + ".A")
            key_b = pair_b[prefix].get("B_key", prefix + ".B")
        else:
            key_a = pair_a[prefix].get("A_key", prefix + ".A")
            key_b = pair_a[prefix].get("B_key", prefix + ".B")
        merged_ab[key_a] = mA
        merged_ab[key_b] = mB

    base_ckpt = ckpt_b if args.base == "b" else ckpt_a
    if isinstance(base_ckpt, dict) and "state_dict" in base_ckpt and isinstance(base_ckpt["state_dict"], dict):
        out_ckpt = dict(base_ckpt)
        out_sd = dict(base_ckpt["state_dict"])
        replaced = 0
        for key, value in merged_ab.items():
            if key in out_sd:
                out_sd[key] = value.to(dtype=out_sd[key].dtype)
                replaced += 1
        out_ckpt["state_dict"] = out_sd
    else:
        out_ckpt = dict(base_ckpt)
        replaced = 0
        for key, value in merged_ab.items():
            out_ckpt[key] = value.to(dtype=out_ckpt.get(key, value).dtype if key in out_ckpt else value.dtype)
            replaced += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(args.output))

    print(f"ckpt_a: {args.ckpt_a} ({type_a})")
    print(f"ckpt_b: {args.ckpt_b} ({type_b})")
    print(f"candidate_prefixes: {len(prefixes)}")
    print(f"only_in_a_prefixes: {len(only_a)}")
    print(f"only_in_b_prefixes: {len(only_b)}")
    print(f"valid_prefixes: {len(valid_prefixes)}")
    print(f"dropped_missing_ab: {dropped_missing_ab}")
    print(f"dropped_bad_shape: {dropped_bad_shape}")
    print(f"replaced_tensors(A+B): {replaced}")
    print(f"weights: {args.weights}, density: {args.density}")
    print(f"scale_a(alpha/r): {scale_a}, scale_b(alpha/r): {scale_b}, base_scale: {base_scale}")
    print(f"norm_scale_a_mean: {sum(scale_history_a) / max(len(scale_history_a), 1):.6f}")
    print(f"norm_scale_b_mean: {sum(scale_history_b) / max(len(scale_history_b), 1):.6f}")
    print(f"svd_device: {args.svd_device}")
    print(f"scope: {'all_lora' if args.all_lora else ('text_only' if args.text_only else 'default')}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
