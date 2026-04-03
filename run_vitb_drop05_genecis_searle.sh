#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB32_SEARLEPhi_And_Drop0p5_GeneCIS"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
GENECIS_JSONL="${LOG_DIR}/genecis_merged.jsonl"
GENECIS_LOG="${LOG_DIR}/genecis_merged_watcher.log"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"
: > "${GENECIS_JSONL}"

"${PYTHON_BIN}" "${ROOT}/data/watch_multidataset_eval.py" \
  --mode merged \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-jsonl "${GENECIS_JSONL}" \
  --eval-gpu 0 \
  --model "ViT-B/32" \
  --img2text-arch phi \
  --middle-dim 2048 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
  --batch-size 32 \
  --workers 2 \
  --genecis-batch-size 32 \
  --datasets "genecis" \
  --base-kind raw \
  --geo-kind ema \
  --merge-mode ties \
  --min-step 600 \
  --merge-weight-a 0.5 \
  --merge-weight-b 0.5 \
  --merge-density 0.9 \
  --merge-alpha-a 16 \
  --merge-rank-a 64 \
  --merge-alpha-b 16 \
  --merge-rank-b 64 \
  --cpu-threads 1 \
  --poll-interval 15 \
  --timeout 7200 \
  --stop-on-final > "${GENECIS_LOG}" 2>&1 &
WATCHER_PID=$!

cleanup() {
  if kill -0 "${WATCHER_PID}" 2>/dev/null; then
    wait "${WATCHER_PID}" || true
  fi
}
trap cleanup EXIT

MODEL_NAME="ViT-B/32" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
IMG2TEXT_ARCH="phi" \
IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
MIDDLE_DIM="2048" \
RETRIEVAL_PROMPT_CONNECTOR="and" \
TRAIN_CUDA_DEVICES="2,3" \
DIST_URL="tcp://127.0.0.1:6173" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="256" \
TRAIN_ACCUM_STEPS="2" \
LR="2e-5" \
GEO_LR="2e-5" \
INSTRUCTION_DROPOUT_PROB="0.5" \
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
