#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:?TRAIN_CUDA_DEVICES is required}"
DIST_URL="${DIST_URL:?DIST_URL is required}"
RESULT_JSON="${RESULT_JSON:-${ROOT}/docs/experiments/geo_branch_cirr_20260418/results/${RUN_NAME}.json}"
RETRIEVAL_PROMPT_CONNECTOR="${RETRIEVAL_PROMPT_CONNECTOR:-and}"
GEO_USE_REVERSE_ALIGNMENT="${GEO_USE_REVERSE_ALIGNMENT:-1}"
GEO_ZERO_LOSS_WEIGHT="${GEO_ZERO_LOSS_WEIGHT:-1.0}"
GEO_SRC_ANCHOR_MODE="${GEO_SRC_ANCHOR_MODE:-text}"
GEO_SRC_IMAGE_WEIGHT="${GEO_SRC_IMAGE_WEIGHT:-0.25}"
GEO_SRC_ANCHOR_DETACH="${GEO_SRC_ANCHOR_DETACH:-0}"

LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
IFS=',' read -r -a GPU_LIST <<< "${TRAIN_CUDA_DEVICES}"
EVAL_GPU="${GPU_LIST[0]}"

rm -rf "${LOG_DIR}"
mkdir -p "$(dirname "${RESULT_JSON}")"

MODEL_NAME="ViT-L/14" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
IMG2TEXT_ARCH="phi" \
IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt" \
MIDDLE_DIM="3072" \
RETRIEVAL_PROMPT_CONNECTOR="${RETRIEVAL_PROMPT_CONNECTOR}" \
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES}" \
DIST_URL="${DIST_URL}" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="56" \
TRAIN_ACCUM_STEPS="8" \
LR="2e-5" \
GEO_LR="2e-5" \
INSTRUCTION_DROPOUT_PROB="0.0" \
TRAIN_EPOCH_STEPS="1000" \
SAVE_STEP_START="1000" \
SAVE_STEP_END="1000" \
SAVE_STEP_INTERVAL="1000" \
SHARED_B_LORA="1" \
SHARED_B_NUM_LAYERS="12" \
SHARED_B_RETRIEVAL_ONLY_UPDATE="1" \
GEO_REVERSE_WEIGHT="0.0" \
GEO_ZERO_LOSS_WEIGHT="${GEO_ZERO_LOSS_WEIGHT}" \
GEO_USE_REVERSE_ALIGNMENT="${GEO_USE_REVERSE_ALIGNMENT}" \
GEO_SRC_PROMPT_STYLE="plain" \
GEO_SRC_ANCHOR_MODE="${GEO_SRC_ANCHOR_MODE}" \
GEO_SRC_IMAGE_WEIGHT="${GEO_SRC_IMAGE_WEIGHT}" \
GEO_SRC_ANCHOR_DETACH="${GEO_SRC_ANCHOR_DETACH}" \
RUN_POSTHOC_MERGED_EVAL="0" \
RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
RUN_POSTHOC_CIRR_EVAL="0" \
MULTIDATASET_EVAL_EVERY="0" \
CIRR_VAL_EVAL_EVERY="0" \
bash "${ROOT}/train_with_dropout.sh"

"${PYTHON_BIN}" "${ROOT}/data/eval_single_merged_cirr_val.py" \
  --ckpt-a "${CKPT_DIR}/epoch_0_step_1000.pt" \
  --ckpt-b "${CKPT_DIR}/epoch_0_step_1000_geo_lora_ema.pt" \
  --output-json "${RESULT_JSON}" \
  --eval-gpu "${EVAL_GPU}" \
  --model "ViT-L/14" \
  --img2text-arch phi \
  --middle-dim 3072 \
  --img2text-pretrained "/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt" \
  --batch-size 48 \
  --workers 2 \
  --retrieval-prompt-connector "${RETRIEVAL_PROMPT_CONNECTOR}" \
  --merge-weight-a 0.5 \
  --merge-weight-b 0.5 \
  --density 0.9 \
  --merge-mode ties \
  --shared-b-num-layers 12
