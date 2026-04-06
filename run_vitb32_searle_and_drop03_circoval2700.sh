#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_And_Drop0p3_CIRCO2700}"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
OUT_JSONL="${LOG_DIR}/circoval_merged_gpu6.jsonl"
EVAL_GPU="${EVAL_GPU:-6}"

MODEL_NAME="ViT-B/32" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
IMG2TEXT_ARCH="phi" \
IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
MIDDLE_DIM="2048" \
RETRIEVAL_PROMPT_CONNECTOR="and" \
TRAIN_CUDA_DEVICES="6,7" \
DIST_URL="tcp://127.0.0.1:6196" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="256" \
TRAIN_ACCUM_STEPS="2" \
LR="2e-5" \
GEO_LR="2e-5" \
INSTRUCTION_DROPOUT_PROB="0.3" \
TRAIN_EPOCH_STEPS="2700" \
SAVE_STEP_START="1700" \
SAVE_STEP_END="2700" \
SAVE_STEP_INTERVAL="200" \
RUN_POSTHOC_MERGED_EVAL="0" \
RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
RUN_POSTHOC_CIRR_EVAL="0" \
MULTIDATASET_EVAL_EVERY="0" \
CIRR_VAL_EVAL_EVERY="0" \
bash "${ROOT}/train_with_dropout.sh"

: > "${OUT_JSONL}"

run_step() {
  local tag="$1"
  local base_ckpt="$2"
  local geo_ckpt="$3"
  local step_json="$4"
  local merged_ckpt="/tmp/${RUN_NAME}_${tag}_circo.pt"
  local out_json="/tmp/${RUN_NAME}_${tag}_circo_val.json"

  "${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${base_ckpt}" \
    --ckpt-b "${geo_ckpt}" \
    --output "${merged_ckpt}" \
    --weights 0.5 0.5 \
    --density 0.9 \
    --text-only \
    --base a \
    --alpha-a 16 --rank-a 64 \
    --alpha-b 16 --rank-b 64

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" "${ROOT}/data/eval_multidataset_suite.py" \
    --resume "${merged_ckpt}" \
    --output-json "${out_json}" \
    --gpu 0 \
    --model "ViT-B/32" \
    --retrieval-prompt-connector "and" \
    --img2text-arch "phi" \
    --img2text-pretrained "/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
    --middle-dim 2048 \
    --batch-size 32 \
    --workers 2 \
    --datasets "circo"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
out_path = Path("${out_json}")
jsonl_path = Path("${OUT_JSONL}")
step_value = None if "${step_json}" == "null" else int("${step_json}")
record = {"tag": "${tag}", "step": step_value}
record.update(json.loads(out_path.read_text(encoding="utf-8")))
with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\\n")
PY

  rm -f "${merged_ckpt}" "${out_json}"
}

for STEP in 1700 1900 2100 2300 2500 2700; do
  BASE_CKPT="${CKPT_DIR}/epoch_0_step_${STEP}.pt"
  GEO_CKPT="${CKPT_DIR}/epoch_0_step_${STEP}_geo_lora_ema.pt"
  if [[ -f "${BASE_CKPT}" && -f "${GEO_CKPT}" ]]; then
    run_step "epoch0_step${STEP}_raw_plus_geoema_merged" "${BASE_CKPT}" "${GEO_CKPT}" "${STEP}"
  fi
done

if [[ -f "${CKPT_DIR}/epoch_1.pt" && -f "${CKPT_DIR}/epoch_1_geo_lora_ema.pt" ]]; then
  run_step "epoch1_final_raw_plus_geoema_merged" "${CKPT_DIR}/epoch_1.pt" "${CKPT_DIR}/epoch_1_geo_lora_ema.pt" "null"
fi
