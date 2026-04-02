#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB32_OpenAI_And_NoDrop_CIRR"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
FINAL_MERGED="/tmp/${RUN_NAME}_step1400_merged.pt"
CIRR_VAL_JSONL="${LOG_DIR}/cirr_val_merged.jsonl"

rm -rf "${LOG_DIR}"

MODEL_NAME="ViT-B/32" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
RETRIEVAL_PROMPT_CONNECTOR="and" \
TRAIN_CUDA_DEVICES="6,7" \
DIST_URL="tcp://127.0.0.1:6172" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="168" \
TRAIN_ACCUM_STEPS="3" \
INSTRUCTION_DROPOUT_PROB="0.0" \
TRAIN_EPOCH_STEPS="1400" \
SAVE_STEP_START="600" \
SAVE_STEP_END="1400" \
SAVE_STEP_INTERVAL="200" \
RUN_POSTHOC_MERGED_EVAL="0" \
RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
RUN_POSTHOC_CIRR_EVAL="0" \
MULTIDATASET_EVAL_EVERY="0" \
CIRR_VAL_EVAL_EVERY="0" \
bash "${ROOT}/train_with_dropout.sh"

rm -f "${CIRR_VAL_JSONL}"
"${PYTHON_BIN}" "${ROOT}/data/watch_cirr_parallel_merge_eval.py" \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-jsonl "${CIRR_VAL_JSONL}" \
  --eval-gpu 6 \
  --batch-size 48 \
  --workers 2 \
  --base-kind raw \
  --geo-kind ema \
  --merge-mode ties \
  --min-step 600 \
  --retrieval-weight 0.5 \
  --geo-weight 0.5 \
  --density 0.9 \
  --lora-alpha 16 \
  --lora-rank 64 \
  --poll-interval 1 \
  --timeout 21600 \
  --stop-on-final \
  --once

"${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
  --ckpt-a "${CKPT_DIR}/epoch_0_step_1400.pt" \
  --ckpt-b "${CKPT_DIR}/epoch_0_step_1400_geo_lora_ema.pt" \
  --output "${FINAL_MERGED}" \
  --weights 0.5 0.5 \
  --density 0.9 \
  --text-only \
  --base a \
  --alpha-a 16 --rank-a 64 \
  --alpha-b 16 --rank-b 64

mkdir -p "${LOG_DIR}/cirr_test_step1400"
CUDA_VISIBLE_DEVICES=6 "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
  --resume "${FINAL_MERGED}" \
  --openai-pretrained \
  --model "ViT-B/32" \
  --eval-mode cirr_test \
  --gpu 0 \
  --batch-size 48 \
  --workers 2 \
  --cirr-output-dir "${LOG_DIR}/cirr_test_step1400"

cp "${LOG_DIR}/cirr_test_step1400/composed.json" "${LOG_DIR}/cirr_step1400_composed.json"
cp "${LOG_DIR}/cirr_test_step1400/subset_composed.json" "${LOG_DIR}/cirr_step1400_subset_composed.json"
rm -rf "${LOG_DIR}/cirr_test_step1400"
rm -f "${FINAL_MERGED}"
