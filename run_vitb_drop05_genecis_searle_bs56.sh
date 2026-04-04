#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB32_SEARLEPhi_And_Drop0p5_GeneCIS_BS56"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
GENECIS_JSONL="${LOG_DIR}/genecis_merged.jsonl"

MODEL_NAME="ViT-B/32" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
IMG2TEXT_ARCH="phi" \
IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
MIDDLE_DIM="2048" \
RETRIEVAL_PROMPT_CONNECTOR="and" \
TRAIN_CUDA_DEVICES="0,1" \
DIST_URL="tcp://127.0.0.1:6177" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="56" \
TRAIN_ACCUM_STEPS="8" \
LR="2e-5" \
GEO_LR="2e-5" \
INSTRUCTION_DROPOUT_PROB="0.5" \
TRAIN_EPOCH_STEPS="1700" \
SAVE_STEP_START="600" \
SAVE_STEP_END="1600" \
SAVE_STEP_INTERVAL="200" \
RUN_POSTHOC_MERGED_EVAL="0" \
RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
RUN_POSTHOC_CIRR_EVAL="0" \
MULTIDATASET_EVAL_EVERY="0" \
CIRR_VAL_EVAL_EVERY="0" \
bash "${ROOT}/train_with_dropout.sh"

: > "${GENECIS_JSONL}"

eval_split() {
  local GPU_ID="$1"
  shift
  for STEP in "$@"; do
    local BASE_CKPT=""
    local GEO_CKPT=""
    local TAG=""
    local STEP_JSON="null"
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
    local MERGED_CKPT="/tmp/${RUN_NAME}_${TAG}.pt"
    local OUT_JSON="/tmp/${RUN_NAME}_${TAG}_genecis.json"

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
      --genecis-batch-size 32 \
      --datasets "genecis"

    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

out_path = Path("${OUT_JSON}")
jsonl_path = Path("${GENECIS_JSONL}")
record = {"tag": "${TAG}", "step": ${STEP_JSON}}
record.update(json.loads(out_path.read_text(encoding="utf-8")))
with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\\n")
PY

    rm -f "${MERGED_CKPT}" "${OUT_JSON}"
  done
}

eval_split 0 600 1000 1400 final &
PID_A=$!
eval_split 1 800 1200 1600 &
PID_B=$!
wait "${PID_A}" "${PID_B}"
