#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
RUN_DIR="$ROOT/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp"
CKPT_DIR="$RUN_DIR/checkpoints"
PY="/data2/mingyu/miniconda3/envs/torch/bin/python"
SEARLE="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt"
GPU="${1:-0}"

export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

run_eval() {
  local out_jsonl="$1"
  shift
  : > "$out_jsonl"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/data/eval_cirr_merged_steps.py" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-jsonl "$out_jsonl" \
    --eval-gpu 0 \
    --model "ViT-L/14" \
    --img2text-arch "phi" \
    --middle-dim 3072 \
    --img2text-pretrained "$SEARLE" \
    --batch-size 32 \
    --workers 2 \
    --base-kind raw \
    --geo-kind ema \
    --min-step 1400 \
    --max-step 1400 \
    "$@"
}

run_eval "$RUN_DIR/cirr_val_step1400_ties_l0.jsonl" \
  --merge-mode ties

run_eval "$RUN_DIR/cirr_val_step1400_hybrid_svd_k32_ties_l2.jsonl" \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 2 \
  --svd-topk-rank 32

run_eval "$RUN_DIR/cirr_val_step1400_hybrid_svd_k32_ties_l4.jsonl" \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 4 \
  --svd-topk-rank 32

run_eval "$RUN_DIR/cirr_val_step1400_hybrid_svd_k32_ties_l6.jsonl" \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 6 \
  --svd-topk-rank 32

run_eval "$RUN_DIR/cirr_val_step1400_hybrid_svd_k32_ties_l8.jsonl" \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 8 \
  --svd-topk-rank 32

run_eval "$RUN_DIR/cirr_val_step1400_hybrid_svd_k32_ties_l10.jsonl" \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 10 \
  --svd-topk-rank 32

run_eval "$RUN_DIR/cirr_val_step1400_svd_denoise_l12.jsonl" \
  --merge-mode shared_b_svd_a \
  --svd-topk-rank 32
