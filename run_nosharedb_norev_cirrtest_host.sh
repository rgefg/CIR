#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PY="/data2/mingyu/miniconda3/envs/torch/bin/python"
LOG="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_NoSharedB_NoRev_CIRR"
CKPT="${LOG}/checkpoints"
OUT="${LOG}/cirr_test_step1400"
MERGED="/tmp/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_NoSharedB_NoRev_CIRR_step1400_merged.pt"

export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"
rm -f "${MERGED}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

"${PY}" "${ROOT}/data/merge_lora_ties.py" \
  --ckpt-a "${CKPT}/epoch_0_step_1400.pt" \
  --ckpt-b "${CKPT}/epoch_0_step_1400_geo_lora_ema.pt" \
  --output "${MERGED}" \
  --weights 0.5 0.5 \
  --density 0.9 \
  --text-only \
  --base a \
  --alpha-a 16 --rank-a 64 \
  --alpha-b 16 --rank-b 64

CUDA_VISIBLE_DEVICES=2 "${PY}" "${ROOT}/src/eval_retrieval.py" \
  --resume "${MERGED}" \
  --openai-pretrained \
  --model "ViT-L/14" \
  --eval-mode cirr_test \
  --gpu 0 \
  --batch-size 48 \
  --workers 2 \
  --img2text-arch phi \
  --middle_dim 3072 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
  --retrieval-prompt-connector and \
  --cirr-output-dir "${OUT}" \
  > "${LOG}/cirr_step1400_eval.log" 2>&1

cp "${OUT}/composed.json" "${LOG}/cirr_step1400_composed.json"
cp "${OUT}/subset_composed.json" "${LOG}/cirr_step1400_subset_composed.json"
