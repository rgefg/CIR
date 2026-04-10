#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
RUN_DIR="$ROOT/logs/DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_And_NoDrop_SharedB_CIRR_MergeCmp"
CKPT_DIR="$RUN_DIR/checkpoints"
PY="/data2/mingyu/miniconda3/envs/torch/bin/python"
SEARLE="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt"

export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

run_eval() {
  local gpu="$1"
  local out_jsonl="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/data/eval_cirr_merged_steps.py" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-jsonl "$out_jsonl" \
    --eval-gpu 0 \
    --model "ViT-B/32" \
    --img2text-arch "phi" \
    --middle-dim 2048 \
    --img2text-pretrained "$SEARLE" \
    --batch-size 48 \
    --workers 2 \
    --base-kind raw \
    --geo-kind ema \
    --min-step 1400 \
    --max-step 1400 \
    "$@"
}

run_eval 2 \
  "$RUN_DIR/cirr_val_step1400_svd_merge.jsonl" \
  --merge-mode shared_b_svd_a \
  --svd-topk-rank 64 &

run_eval 3 \
  "$RUN_DIR/cirr_val_step1400_svd_denoise_k32.jsonl" \
  --merge-mode shared_b_svd_a \
  --svd-topk-rank 32 &

run_eval 4 \
  "$RUN_DIR/cirr_val_step1400_hybrid_svd_k32_ties.jsonl" \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 6 \
  --svd-topk-rank 32 &

wait
