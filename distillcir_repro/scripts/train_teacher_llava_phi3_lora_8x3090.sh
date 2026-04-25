#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is set to '${CUDA_VISIBLE_DEVICES}'. Unset it to avoid physical GPU remapping." >&2
  exit 2
fi

PHYSICAL_GPUS="${PHYSICAL_GPUS:-}"
if [[ -n "$PHYSICAL_GPUS" ]]; then
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$PHYSICAL_GPUS"
  GPUS="${GPUS:-${#GPU_ID_ARRAY[@]}}"
  PHYSICAL_GPU_ARGS=(--physical-gpus "$PHYSICAL_GPUS")
else
  GPUS="${GPUS:-8}"
  PHYSICAL_GPU_ARGS=()
fi
TEACHER_MODEL="${TEACHER_MODEL:-/data2/mingyu/composed_image_retrieval/checkpoint/hf_models/xtuner_llava_phi3_mini_hf}"
CC3M_JSONL="${CC3M_JSONL:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl}"
WDS_SHARDS="${WDS_SHARDS:-/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_lora_lcom}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

torchrun --standalone --nproc_per_node="$GPUS" src/train_llava_teacher_contrastive.py \
  --teacher-model "$TEACHER_MODEL" \
  --cc3m-cir-jsonl "$CC3M_JSONL" \
  --wds-shards "$WDS_SHARDS" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${PER_GPU_BATCH:-1}" \
  --accum-steps "${ACCUM_STEPS:-96}" \
  --workers "${WORKERS:-2}" \
  --max-steps "${MAX_STEPS:-2807}" \
  --lr "${LR:-2e-5}" \
  --dtype fp16 \
  --use-lora \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --train-projector \
  "${PHYSICAL_GPU_ARGS[@]}" \
  "$@"
