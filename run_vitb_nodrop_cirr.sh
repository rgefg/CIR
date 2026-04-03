#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB32_OpenAI_And_NoDrop_LR5x_CIRR"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
CIRR_VAL_JSONL="${LOG_DIR}/cirr_val_merged.jsonl"

rm -rf "${LOG_DIR}"

MODEL_NAME="ViT-B/32" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
RETRIEVAL_PROMPT_CONNECTOR="and" \
TRAIN_CUDA_DEVICES="0,1" \
DIST_URL="tcp://127.0.0.1:6174" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="256" \
TRAIN_ACCUM_STEPS="2" \
LR="1e-4" \
GEO_LR="1e-4" \
INSTRUCTION_DROPOUT_PROB="0.0" \
TRAIN_EPOCH_STEPS="1400" \
SAVE_STEP_START="200" \
SAVE_STEP_END="1400" \
SAVE_STEP_INTERVAL="200" \
RUN_POSTHOC_MERGED_EVAL="0" \
RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
RUN_POSTHOC_CIRR_EVAL="0" \
MULTIDATASET_EVAL_EVERY="0" \
CIRR_VAL_EVAL_EVERY="0" \
bash "${ROOT}/train_with_dropout.sh"

rm -f "${CIRR_VAL_JSONL}"
"${PYTHON_BIN}" "${ROOT}/data/eval_cirr_merged_steps.py" \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-jsonl "${CIRR_VAL_JSONL}" \
  --eval-gpu 0 \
  --model "ViT-B/32" \
  --batch-size 48 \
  --workers 2 \
  --base-kind raw \
  --geo-kind ema \
  --min-step 200 \
  --retrieval-weight 0.5 \
  --geo-weight 0.5 \
  --density 0.9 \
  --lora-alpha 16 \
  --lora-rank 64
