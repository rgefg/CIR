#!/usr/bin/env bash
set -euo pipefail

# DistillCIR student training launcher for physical GPUs 0-7.
# Do not set CUDA_VISIBLE_DEVICES here: the training script refuses remapped GPUs
# unless --allow-cuda-visible-devices is passed explicitly.

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

PHYSICAL_GPUS="${PHYSICAL_GPUS:-}"
if [[ -n "$PHYSICAL_GPUS" ]]; then
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$PHYSICAL_GPUS"
  GPUS="${GPUS:-${#GPU_ID_ARRAY[@]}}"
  PHYSICAL_GPU_ARGS=(--physical-gpus "$PHYSICAL_GPUS")
else
  GPUS="${GPUS:-8}"
  PHYSICAL_GPU_ARGS=()
fi
PER_GPU_BATCH="${PER_GPU_BATCH:-24}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-2}"
WDS_EPOCH_STEPS="${WDS_EPOCH_STEPS:-2807}"
LR="${LR:-2e-5}"
WARMUP="${WARMUP:-1000}"
ALPHA_REASON="${ALPHA_REASON:-1.0}"
BETA_FEATURE="${BETA_FEATURE:-1.0}"
AMP_INIT_SCALE="${AMP_INIT_SCALE:-1024}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

CC3M_JSONL="${CC3M_JSONL:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl}"
if [[ -z "${WDS_SHARDS:-}" ]]; then
  WDS_SHARDS='/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar'
fi
PIC2WORD_CKPT="${PIC2WORD_CKPT:-/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt}"
TEACHER_CACHE="${TEACHER_CACHE:-/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_cc3m_cache}"

CONNECTOR="${CONNECTOR:-and}"         # CIRR uses and. CIRCO/FashionIQ use that.
REASON_CONNECTOR="${REASON_CONNECTOR:-that}"
RUN_NAME="${RUN_NAME:-DistillCIR_Repro_ViTL14_${CONNECTOR}_8x3090}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is set to '${CUDA_VISIBLE_DEVICES}'. Unset it to avoid physical GPU remapping." >&2
  exit 2
fi

python src/validate_distillcir_ready.py \
  --cc3m-cir-jsonl "$CC3M_JSONL" \
  --wds-shards "$WDS_SHARDS" \
  --pic2word-pretrained "$PIC2WORD_CKPT" \
  --teacher-cache "$TEACHER_CACHE"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

torchrun --standalone --nproc_per_node="$GPUS" src/train_distillcir.py \
  --model ViT-L/14 \
  --pic2word-pretrained "$PIC2WORD_CKPT" \
  --img2text-arch im2text \
  --middle-dim 512 \
  --n-layer 2 \
  --droprate 0.0 \
  --cc3m-cir-jsonl "$CC3M_JSONL" \
  --wds-shards "$WDS_SHARDS" \
  --teacher-cache "$TEACHER_CACHE" \
  --retrieval-prompt-connector "$CONNECTOR" \
  --reason-prompt-connector "$REASON_CONNECTOR" \
  --reason-prompt-tokens 8 \
  --alpha-reason "$ALPHA_REASON" \
  --beta-feature "$BETA_FEATURE" \
  --batch-size "$PER_GPU_BATCH" \
  --accum-steps "$ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --wds-epoch-steps "$WDS_EPOCH_STEPS" \
  --workers "$WORKERS" \
  --lr "$LR" \
  --wd 0.2 \
  --warmup "$WARMUP" \
  --precision amp \
  --amp-init-scale "$AMP_INIT_SCALE" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --lora-r 64 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --logs ./logs \
  --name "$RUN_NAME" \
  --log-interval 20 \
  "${PHYSICAL_GPU_ARGS[@]}" \
  "$@"
