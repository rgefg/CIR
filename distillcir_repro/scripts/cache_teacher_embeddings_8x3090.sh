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
TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"
CC3M_JSONL="${CC3M_JSONL:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl}"
CACHE_DIR="${CACHE_DIR:-/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_cc3m_cache}"
BATCH_SIZE="${BATCH_SIZE:-32}"
if [[ -z "${WDS_SHARDS:-}" ]]; then
  WDS_SHARDS='/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar'
fi

mkdir -p "$CACHE_DIR/shards"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

ADAPTER_ARGS=()
if [[ -n "$TEACHER_ADAPTER" ]]; then
  ADAPTER_ARGS=(--adapter-path "$TEACHER_ADAPTER")
fi

torchrun --standalone --nproc_per_node="$GPUS" src/cache_llava_teacher_embeddings.py \
  --jsonl "$CC3M_JSONL" \
  --model-path "$TEACHER_MODEL" \
  "${ADAPTER_ARGS[@]}" \
  --output-dir "$CACHE_DIR/shards" \
  --batch-size "$BATCH_SIZE" \
  --dtype fp16 \
  "${PHYSICAL_GPU_ARGS[@]}"

python src/merge_teacher_embedding_shards.py \
  --shard-dir "$CACHE_DIR/shards" \
  --output-dir "$CACHE_DIR"

python src/validate_distillcir_ready.py \
  --cc3m-cir-jsonl "$CC3M_JSONL" \
  --wds-shards "$WDS_SHARDS" \
  --pic2word-pretrained "${PIC2WORD_CKPT:-/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt}" \
  --teacher-cache "$CACHE_DIR"
