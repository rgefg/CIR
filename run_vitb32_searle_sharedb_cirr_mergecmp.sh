#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_And_NoDrop_SharedB_CIRR_MergeCmp"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
HYBRID_JSONL="${LOG_DIR}/cirr_val_hybrid_svd_a_k32.jsonl"
TIES_JSONL="${LOG_DIR}/cirr_val_ties.jsonl"

rm -rf "${LOG_DIR}"

MODEL_NAME="ViT-B/32" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
IMG2TEXT_ARCH="phi" \
IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt" \
MIDDLE_DIM="2048" \
RETRIEVAL_PROMPT_CONNECTOR="and" \
TRAIN_CUDA_DEVICES="0,1" \
DIST_URL="tcp://127.0.0.1:6217" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="256" \
TRAIN_ACCUM_STEPS="2" \
LR="2e-5" \
GEO_LR="2e-5" \
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
SHARED_B_LORA="1" \
SHARED_B_NUM_LAYERS="6" \
SHARED_B_RETRIEVAL_ONLY_UPDATE="1" \
bash "${ROOT}/train_with_dropout.sh"

rm -f "${HYBRID_JSONL}" "${TIES_JSONL}"

"${PYTHON_BIN}" "${ROOT}/data/eval_cirr_merged_steps.py" \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-jsonl "${HYBRID_JSONL}" \
  --eval-gpu 2 \
  --model "ViT-B/32" \
  --img2text-arch phi \
  --middle-dim 2048 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
  --batch-size 48 \
  --workers 2 \
  --base-kind raw \
  --geo-kind ema \
  --min-step 600 \
  --retrieval-weight 0.5 \
  --geo-weight 0.5 \
  --density 0.9 \
  --merge-mode hybrid_layerwise_svd_a \
  --shared-b-num-layers 6 \
  --svd-topk-rank 32 \
  --lora-alpha 16 \
  --lora-rank 64 &
PID_HYBRID=$!

"${PYTHON_BIN}" "${ROOT}/data/eval_cirr_merged_steps.py" \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-jsonl "${TIES_JSONL}" \
  --eval-gpu 3 \
  --model "ViT-B/32" \
  --img2text-arch phi \
  --middle-dim 2048 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
  --batch-size 48 \
  --workers 2 \
  --base-kind raw \
  --geo-kind ema \
  --min-step 600 \
  --retrieval-weight 0.5 \
  --geo-weight 0.5 \
  --density 0.9 \
  --merge-mode ties \
  --lora-alpha 16 \
  --lora-rank 64 &
PID_TIES=$!

wait "${PID_HYBRID}"
wait "${PID_TIES}"

