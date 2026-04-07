#!/usr/bin/env python3
import argparse
import re
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
   core = _strip_module_prefix(prefix)
   return not core.startswith("visual.")


def _norm_prefix(prefix: str) -> str:
   return _strip_module_prefix(prefix)


_RESBLOCK_RE = re.compile(r"(^|\.)resblocks\.(\d+)(\.|$)")


def _text_resblock_index(prefix: str):
   match = _RESBLOCK_RE.search(_norm_prefix(prefix))
   if match is None:
      return None
   return int(match.group(2))


def _build_pair_map(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
   # Map by normalized prefix to avoid missing matches due to 'module.' differences.
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


def _valid_copy_pair(pair: Dict[str, torch.Tensor]) -> bool:
   if not _valid_pair(pair):
      return False
   a = pair["A"]
   b = pair["B"]
   if (a.ndim != 2) or (b.ndim != 2):
      return False
   return b.shape[1] == a.shape[0]


def _svd_factorize(delta: torch.Tensor, rank: int, out_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
   # delta: [out_dim, in_dim]
   u, s, vh = torch.linalg.svd(delta, full_matrices=False)
   use_r = min(rank, s.shape[0])
   s_root = torch.sqrt(torch.clamp(s[:use_r], min=0.0))
   b_small = u[:, :use_r] * s_root.unsqueeze(0)  # [out, r]
   a_small = s_root.unsqueeze(1) * vh[:use_r, :]  # [r, in]

   if use_r < rank:
      b_full = torch.zeros(delta.shape[0], rank, dtype=b_small.dtype, device=b_small.device)
      a_full = torch.zeros(rank, delta.shape[1], dtype=a_small.dtype, device=a_small.device)
      b_full[:, :use_r] = b_small
      a_full[:use_r, :] = a_small
   else:
      b_full, a_full = b_small, a_small
   return a_full.to(dtype=out_dtype), b_full.to(dtype=out_dtype)


def main() -> None:
   parser = argparse.ArgumentParser(description="True LoRA merge via TIES on delta weights (B@A), then SVD back to A/B.")
   parser.add_argument("--ckpt-a", type=Path, required=True, help="Checkpoint A path.")
   parser.add_argument("--ckpt-b", type=Path, required=True, help="Checkpoint B path.")
   parser.add_argument("--output", type=Path, required=True, help="Output merged checkpoint path.")
   parser.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5], metavar=("WA", "WB"))
   parser.add_argument("--density", type=float, default=0.5)
   parser.add_argument("--majority-sign-method", type=str, default="total", choices=["total", "frequency"])
   parser.add_argument(
      "--merge-mode",
      type=str,
      default="ties",
      choices=["ties", "shared_b_sum_a", "hybrid_layerwise", "shared_a_sum_b"],
      help="Use TIES in delta space, or keep base B fixed and sum only A for Shared-B LoRA.",
   )
   parser.add_argument(
      "--shared-b-num-layers",
      dest="shared_b_num_layers",
      type=int,
      default=6,
      help="When merge-mode=hybrid_layerwise, use Shared-B A-sum for shallow blocks [0, N-1] and TIES for deeper blocks.",
   )
   parser.add_argument("--shared-a-num-layers", dest="shared_b_num_layers", type=int, help=argparse.SUPPRESS)
   parser.add_argument("--text-only", action="store_true", default=False, help="Merge only text encoder LoRA pairs.")
   parser.add_argument("--all-lora", action="store_true", default=False, help="Merge both text and visual LoRA pairs.")
   parser.add_argument("--base", type=str, default="b", choices=["a", "b"], help="Output structure base checkpoint.")
   parser.add_argument(
      "--include-b-only",
      action="store_true",
      default=False,
      help="Also carry over LoRA pairs that exist only in checkpoint B into the output checkpoint.",
   )
   parser.add_argument("--alpha-a", type=float, default=None, help="LoRA alpha for checkpoint A (optional).")
   parser.add_argument("--rank-a", type=int, default=None, help="LoRA rank for checkpoint A (optional).")
   parser.add_argument("--alpha-b", type=float, default=None, help="LoRA alpha for checkpoint B (optional).")
   parser.add_argument("--rank-b", type=int, default=None, help="LoRA rank for checkpoint B (optional).")
   parser.add_argument(
      "--slim-output",
      action="store_true",
      default=False,
      help="When base checkpoint is full, save only the tensors required for eval.",
   )
   args = parser.parse_args()

   print(f"loading ckpt_a: {args.ckpt_a}", flush=True)
   ckpt_a = _load_checkpoint(args.ckpt_a)
   print(f"loading ckpt_b: {args.ckpt_b}", flush=True)
   ckpt_b = _load_checkpoint(args.ckpt_b)
   lora_a, type_a = _extract_lora_dict(ckpt_a)
   lora_b, type_b = _extract_lora_dict(ckpt_b)

   pair_a = _build_pair_map(lora_a)
   pair_b = _build_pair_map(lora_b)
   only_a = sorted(set(pair_a.keys()) - set(pair_b.keys()))
   only_b = sorted(set(pair_b.keys()) - set(pair_a.keys()))
   prefixes = sorted(set(pair_a.keys()) & set(pair_b.keys()))

   if not args.all_lora:
      prefixes = [p for p in prefixes if _is_text_prefix(p)] if args.text_only else prefixes
      only_b = [p for p in only_b if _is_text_prefix(p)] if args.text_only else only_b

   valid_prefixes: List[str] = []
   dropped_missing_ab = 0
   dropped_bad_shape = 0
   for p in prefixes:
      pa = pair_a[p]
      pb = pair_b[p]
      if (not _valid_pair(pa)) or (not _valid_pair(pb)):
         dropped_missing_ab += 1
         continue
      aA, aB = pa["A"], pa["B"]
      bA, bB = pb["A"], pb["B"]
      # A:[r,in], B:[out,r]
      if (aA.ndim != 2) or (aB.ndim != 2) or (bA.ndim != 2) or (bB.ndim != 2):
         dropped_bad_shape += 1
         continue
      if aB.shape[1] != aA.shape[0] or bB.shape[1] != bA.shape[0]:
         dropped_bad_shape += 1
         continue
      if (aB.shape[0] != bB.shape[0]) or (aA.shape[1] != bA.shape[1]):
         dropped_bad_shape += 1
         continue
      valid_prefixes.append(p)

   if not valid_prefixes:
      raise RuntimeError("No valid LoRA module pairs found to merge.")

   scale_a = float(args.alpha_a) / float(args.rank_a) if (args.alpha_a is not None and args.rank_a) else 1.0
   scale_b = float(args.alpha_b) / float(args.rank_b) if (args.alpha_b is not None and args.rank_b) else 1.0
   base_scale = scale_b if args.base == "b" else scale_a
   if abs(base_scale) < 1e-12:
      raise ValueError("Base LoRA scale is zero. Check alpha/rank arguments.")

   w = torch.tensor(args.weights, dtype=torch.float32)
   merged_AB: Dict[str, torch.Tensor] = {}
   copied_b_only_prefixes = 0
   shared_b_mismatch_prefixes = 0
   shallow_shared_prefixes = 0
   deep_ties_prefixes = 0
   weight_a = float(args.weights[0])
   weight_b = float(args.weights[1])

   def _write_shared_b_sum(prefix: str, pa: Dict[str, torch.Tensor], pb: Dict[str, torch.Tensor]):
      nonlocal shared_b_mismatch_prefixes
      aA = pa["A"].float()
      aB = pa["B"].float()
      bA = pb["A"].float()
      bB = pb["B"].float()
      if aB.shape != bB.shape:
         raise RuntimeError(
            f"Shared-B merge requires matching B shapes for {prefix}: {tuple(aB.shape)} vs {tuple(bB.shape)}"
         )
      if not torch.allclose(aB, bB, rtol=1e-4, atol=1e-5):
         shared_b_mismatch_prefixes += 1
      b_ref = aB if args.base == "a" else bB
      coeff_a = weight_a * (scale_a / base_scale)
      coeff_b = weight_b * (scale_b / base_scale)
      merged_A = (coeff_a * aA) + (coeff_b * bA)
      out_dtype = pair_b[prefix]["A"].dtype if args.base == "b" else pair_a[prefix]["A"].dtype
      if args.base == "b":
         kA = pair_b[prefix].get("A_key", prefix + ".A")
         kB = pair_b[prefix].get("B_key", prefix + ".B")
      else:
         kA = pair_a[prefix].get("A_key", prefix + ".A")
         kB = pair_a[prefix].get("B_key", prefix + ".B")
      merged_AB[kA] = merged_A.to(dtype=out_dtype)
      merged_AB[kB] = b_ref.to(dtype=out_dtype)

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
      if args.base == "b":
         kA = pair_b[prefix].get("A_key", prefix + ".A")
         kB = pair_b[prefix].get("B_key", prefix + ".B")
      else:
         kA = pair_a[prefix].get("A_key", prefix + ".A")
         kB = pair_a[prefix].get("B_key", prefix + ".B")
      merged_AB[kA] = mA
      merged_AB[kB] = mB

   for p in valid_prefixes:
      effective_merge_mode = "shared_b_sum_a" if args.merge_mode == "shared_a_sum_b" else args.merge_mode
      if effective_merge_mode == "shared_b_sum_a":
         _write_shared_b_sum(p, pair_a[p], pair_b[p])
      elif effective_merge_mode == "hybrid_layerwise":
         layer_idx = _text_resblock_index(p)
         if layer_idx is not None and layer_idx < int(args.shared_b_num_layers):
            shallow_shared_prefixes += 1
            _write_shared_b_sum(p, pair_a[p], pair_b[p])
         else:
            deep_ties_prefixes += 1
            _write_delta_ties(p, pair_a[p], pair_b[p])
      else:
         _write_delta_ties(p, pair_a[p], pair_b[p])

   if args.include_b_only:
      for p in only_b:
         pb = pair_b[p]
         if not _valid_copy_pair(pb):
            continue
         bA = pb["A"].float()
         bB = pb["B"].float()
         out_dtype = pb["A"].dtype
         kA = pb.get("A_key", p + ".A")
         kB = pb.get("B_key", p + ".B")
         effective_merge_mode = "shared_b_sum_a" if args.merge_mode == "shared_a_sum_b" else args.merge_mode
         if effective_merge_mode == "shared_b_sum_a":
            coeff_b = weight_b * (scale_b / base_scale)
            merged_AB[kA] = (coeff_b * bA).to(dtype=out_dtype)
            merged_AB[kB] = pb["B"].to(dtype=out_dtype)
         elif effective_merge_mode == "hybrid_layerwise" and (_text_resblock_index(p) is not None and _text_resblock_index(p) < int(args.shared_b_num_layers)):
            coeff_b = weight_b * (scale_b / base_scale)
            merged_AB[kA] = (coeff_b * bA).to(dtype=out_dtype)
            merged_AB[kB] = pb["B"].to(dtype=out_dtype)
         else:
            delta_b_eff = scale_b * (bB @ bA)
            delta_m = (weight_b * delta_b_eff) / base_scale
            rank_target = pb["A"].shape[0]
            mA, mB = _svd_factorize(delta_m, rank=rank_target, out_dtype=out_dtype)
            merged_AB[kA] = mA
            merged_AB[kB] = mB
         copied_b_only_prefixes += 1

   base_ckpt = ckpt_b if args.base == "b" else ckpt_a
   if isinstance(base_ckpt, dict) and "state_dict" in base_ckpt and isinstance(base_ckpt["state_dict"], dict):
      if args.slim_output:
         out_ckpt = {
            "state_dict": dict(base_ckpt["state_dict"]),
            "state_dict_img2text": base_ckpt.get("state_dict_img2text"),
         }
         for meta_key in ["epoch", "name", "step", "global_step"]:
            if meta_key in base_ckpt:
               out_ckpt[meta_key] = base_ckpt[meta_key]
      else:
         out_ckpt = dict(base_ckpt)
      out_sd = dict(base_ckpt["state_dict"])
      replaced = 0
      for k, v in merged_AB.items():
         if k in out_sd:
            out_sd[k] = v.to(dtype=out_sd[k].dtype)
            replaced += 1
         elif args.include_b_only:
            out_sd[k] = v
            replaced += 1
      out_ckpt["state_dict"] = out_sd
   else:
      out_ckpt = dict(base_ckpt)
      replaced = 0
      for k, v in merged_AB.items():
         if k in out_ckpt:
            out_ckpt[k] = v.to(dtype=out_ckpt[k].dtype)
         else:
            out_ckpt[k] = v
         replaced += 1

   args.output.parent.mkdir(parents=True, exist_ok=True)
   print(f"saving merged checkpoint: {args.output}", flush=True)
   torch.save(out_ckpt, str(args.output))

   print(f"ckpt_a: {args.ckpt_a} ({type_a})")
   print(f"ckpt_b: {args.ckpt_b} ({type_b})")
   print(f"candidate_prefixes: {len(prefixes)}")
   print(f"only_in_a_prefixes: {len(only_a)}")
   print(f"only_in_b_prefixes: {len(only_b)}")
   print(f"valid_prefixes: {len(valid_prefixes)}")
   print(f"dropped_missing_ab: {dropped_missing_ab}")
   print(f"dropped_bad_shape: {dropped_bad_shape}")
   print(f"copied_b_only_prefixes: {copied_b_only_prefixes}")
   print(f"replaced_tensors(A+B): {replaced}")
   print(f"weights: {args.weights}, density: {args.density}, sign: {args.majority_sign_method}")
   print(f"merge_mode: {args.merge_mode}")
   print(f"shared_b_num_layers: {args.shared_b_num_layers}")
   print(f"shallow_shared_prefixes: {shallow_shared_prefixes}")
   print(f"deep_ties_prefixes: {deep_ties_prefixes}")
   print(f"shared_b_mismatch_prefixes: {shared_b_mismatch_prefixes}")
   print(f"scale_a(alpha/r): {scale_a}, scale_b(alpha/r): {scale_b}, base_scale: {base_scale}")
   print(f"scope: {'all_lora' if args.all_lora else ('text_only' if args.text_only else 'default')}")
   print(f"output: {args.output}")


if __name__ == "__main__":
   main()
