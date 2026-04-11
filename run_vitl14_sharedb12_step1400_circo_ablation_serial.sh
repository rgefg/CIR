#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
RUN_DIR="$ROOT/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp"
CKPT_DIR="$RUN_DIR/checkpoints"
PY="/data2/mingyu/miniconda3/envs/torch/bin/python"
SEARLE="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt"
GPU="${1:-4}"
OUT_JSONL="$RUN_DIR/circo_val_step1400_ablation.jsonl"
RUN_NAME="$(basename "$RUN_DIR")"

export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"
: > "$OUT_JSONL"

run_eval() {
  local layers="$1"
  local merge_mode="$2"
  local merged_ckpt="/tmp/${RUN_NAME}_circo_l${layers}.pt"
  local out_json="/tmp/${RUN_NAME}_circo_l${layers}.json"
  local merge_args=(
    --ckpt-a "$CKPT_DIR/epoch_0_step_1400.pt"
    --ckpt-b "$CKPT_DIR/epoch_0_step_1400_geo_lora_ema.pt"
    --output "$merged_ckpt"
    --weights 0.5 0.5
    --density 0.9
    --text-only
    --base a
    --alpha-a 16 --rank-a 64
    --alpha-b 16 --rank-b 64
    --merge-mode "$merge_mode"
  )

  if [[ "$layers" != "0" ]]; then
    merge_args+=(--shared-b-num-layers "$layers" --svd-topk-rank 32)
  fi

  "$PY" "$ROOT/data/merge_lora_ties.py" "${merge_args[@]}"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/data/eval_multidataset_suite.py" \
    --resume "$merged_ckpt" \
    --output-json "$out_json" \
    --gpu 0 \
    --model "ViT-L/14" \
    --img2text-arch "phi" \
    --img2text-pretrained "$SEARLE" \
    --middle-dim 3072 \
    --batch-size 32 \
    --workers 2 \
    --retrieval-prompt-connector "that" \
    --datasets "circo"

  "$PY" - <<PY
import json
from pathlib import Path

record = {
    "layers": ${layers:-0},
    "merge_mode": "${merge_mode}",
}
record.update(json.loads(Path("${out_json}").read_text(encoding="utf-8")))
with Path("${OUT_JSONL}").open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\\n")
PY

  rm -f "$merged_ckpt" "$out_json"
}

run_eval 0 ties
run_eval 2 hybrid_layerwise_svd_a
run_eval 4 hybrid_layerwise_svd_a
run_eval 6 hybrid_layerwise_svd_a
run_eval 8 hybrid_layerwise_svd_a
run_eval 10 hybrid_layerwise_svd_a
run_eval 12 shared_b_svd_a
