#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_And_Drop0p3_CIRCO1700_v2}"
RUN_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
EVAL_GPU="${EVAL_GPU:-0}"

run_step() {
  local step="$1"
  local base_ckpt="${CKPT_DIR}/epoch_0_step_${step}.pt"
  local geo_ckpt="${CKPT_DIR}/epoch_0_step_${step}_geo_lora_ema.pt"
  local merged_ckpt="/tmp/${RUN_NAME}_step${step}_circo_test_merged.pt"
  local eval_name="circo_test_${RUN_NAME}_step${step}_gpu${EVAL_GPU}"
  local eval_dir="${ROOT}/logs/${eval_name}"
  local final_json="${RUN_DIR}/circo_submission_step${step}.json"

  rm -rf "${eval_dir}"
  rm -f "${merged_ckpt}" "${final_json}"

  "${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${base_ckpt}" \
    --ckpt-b "${geo_ckpt}" \
    --output "${merged_ckpt}" \
    --text-only \
    --base a \
    --weights 0.5 0.5 \
    --density 0.9

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --model "ViT-B/32" \
    --openai-pretrained \
    --img2text-arch "phi" \
    --img2text-pretrained "/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
    --middle_dim 2048 \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --resume "${merged_ckpt}" \
    --eval-mode circo \
    --name "${eval_name}"

  cp "${eval_dir}/circo_submission.json" "${final_json}"
  rm -f "${merged_ckpt}"
}

for step in 400 800 1200; do
  run_step "${step}"
done
