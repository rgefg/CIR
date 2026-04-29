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
MODEL="${MODEL:-ViT-L/14}"
OPENAI_PRETRAINED="${OPENAI_PRETRAINED:-0}"
IMG2TEXT_ARCH="${IMG2TEXT_ARCH:-im2text}"
IMG2TEXT_PRETRAINED="${IMG2TEXT_PRETRAINED:-}"
MIDDLE_DIM="${MIDDLE_DIM:-512}"
N_LAYER="${N_LAYER:-2}"
DROPRATE="${DROPRATE:-0.0}"
LR="${LR:-2e-5}"
WARMUP="${WARMUP:-1000}"
WD="${WD:-0.2}"
ALPHA_REASON="${ALPHA_REASON:-1.0}"
BETA_FEATURE="${BETA_FEATURE:-1.0}"
AMP_INIT_SCALE="${AMP_INIT_SCALE:-1024}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
LOGS_DIR="${LOGS_DIR:-/data2/mingyu/composed_image_retrieval/logs/distillcir_repro}"
WDS_SHUFFLE="${WDS_SHUFFLE:-20000}"
WDS_SHARDSHUFFLE="${WDS_SHARDSHUFFLE:-1000}"
RESET_LOGIT_SCALE="${RESET_LOGIT_SCALE:-0}"
MIN_LOGIT_SCALE="${MIN_LOGIT_SCALE:-0}"
MAX_LOGIT_SCALE="${MAX_LOGIT_SCALE:-100}"

CC3M_JSONL="${CC3M_JSONL:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl}"
if [[ -z "${WDS_SHARDS:-}" ]]; then
  WDS_SHARDS='/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar'
fi
PIC2WORD_CKPT="${PIC2WORD_CKPT:-/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt}"
TEACHER_CACHE="${TEACHER_CACHE:-/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_lora_lcom_4x3090_b28_cc3m_cache}"

CONNECTOR="${CONNECTOR:-and}"         # CIRR uses and. CIRCO/FashionIQ use that.
REASON_CONNECTOR="${REASON_CONNECTOR:-that}"
MODEL_TAG="${MODEL//\//}"
MODEL_TAG="${MODEL_TAG//-/_}"
RUN_NAME="${RUN_NAME:-DistillCIR_Repro_${MODEL_TAG}_${CONNECTOR}_8x3090}"

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

EXTRA_ARGS=()
if [[ "$RESET_LOGIT_SCALE" == "1" ]]; then
  EXTRA_ARGS+=(--reset-logit-scale)
fi
if [[ "$OPENAI_PRETRAINED" == "1" ]]; then
  EXTRA_ARGS+=(--openai-pretrained)
else
  EXTRA_ARGS+=(--pic2word-pretrained "$PIC2WORD_CKPT")
fi
if [[ -n "$IMG2TEXT_PRETRAINED" ]]; then
  EXTRA_ARGS+=(--img2text-pretrained "$IMG2TEXT_PRETRAINED")
fi

torchrun --standalone --nproc_per_node="$GPUS" -- src/train_distillcir.py \
  --model "$MODEL" \
  --img2text-arch "$IMG2TEXT_ARCH" \
  --middle-dim "$MIDDLE_DIM" \
  --n-layer "$N_LAYER" \
  --droprate "$DROPRATE" \
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
  --wds-shuffle "$WDS_SHUFFLE" \
  --wds-shardshuffle "$WDS_SHARDSHUFFLE" \
  --workers "$WORKERS" \
  --lr "$LR" \
  --wd "$WD" \
  --warmup "$WARMUP" \
  --precision amp \
  --amp-init-scale "$AMP_INIT_SCALE" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --min-logit-scale "$MIN_LOGIT_SCALE" \
  --max-logit-scale "$MAX_LOGIT_SCALE" \
  --lora-r 64 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --logs "$LOGS_DIR" \
  --name "$RUN_NAME" \
  --log-interval 20 \
  "${PHYSICAL_GPU_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
