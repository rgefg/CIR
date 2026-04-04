#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB32_SEARLEPhi_And_Drop0p5_GeneCIS"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
OUT_JSONL="${LOG_DIR}/circoval_merged_gpu2.jsonl"
GPU_ID="${GPU_ID:-2}"

: > "${OUT_JSONL}"

for STEP in 600 800 1000 1200 1400 final; do
  BASE_CKPT=""
  GEO_CKPT=""
  TAG=""
  STEP_JSON="null"
  if [[ "${STEP}" == "final" ]]; then
    BASE_CKPT="${CKPT_DIR}/epoch_1.pt"
    GEO_CKPT="${CKPT_DIR}/epoch_1_geo_lora_ema.pt"
    TAG="epoch1_final_raw_plus_geoema_merged"
  else
    BASE_CKPT="${CKPT_DIR}/epoch_0_step_${STEP}.pt"
    GEO_CKPT="${CKPT_DIR}/epoch_0_step_${STEP}_geo_lora_ema.pt"
    TAG="epoch0_step${STEP}_raw_plus_geoema_merged"
    STEP_JSON="${STEP}"
  fi
  MERGED_CKPT="/tmp/${RUN_NAME}_${TAG}_circo.pt"
  OUT_JSON="/tmp/${RUN_NAME}_${TAG}_circo_val.json"

  "${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${BASE_CKPT}" \
    --ckpt-b "${GEO_CKPT}" \
    --output "${MERGED_CKPT}" \
    --weights 0.5 0.5 \
    --density 0.9 \
    --text-only \
    --base a \
    --alpha-a 16 --rank-a 64 \
    --alpha-b 16 --rank-b 64

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT}/data/eval_multidataset_suite.py" \
    --resume "${MERGED_CKPT}" \
    --output-json "${OUT_JSON}" \
    --gpu 0 \
    --model "ViT-B/32" \
    --img2text-arch "phi" \
    --img2text-pretrained "/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
    --middle-dim 2048 \
    --batch-size 32 \
    --workers 2 \
    --datasets "circo"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
out_path = Path("${OUT_JSON}")
jsonl_path = Path("${OUT_JSONL}")
record = {"tag": "${TAG}", "step": ${STEP_JSON}}
record.update(json.loads(out_path.read_text(encoding="utf-8")))
with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\\n")
PY

  rm -f "${MERGED_CKPT}" "${OUT_JSON}"
done
