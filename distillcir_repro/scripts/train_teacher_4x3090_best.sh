#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1,2,7}"
export GPUS="${GPUS:-4}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-28}"
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
export LR="${LR:-5e-6}"
export AMP_INIT_SCALE="${AMP_INIT_SCALE:-1024}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export MAX_STEPS="${MAX_STEPS:-2807}"
export WORKERS="${WORKERS:-2}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_lora_lcom_4x3090_b28}"

bash scripts/train_teacher_llava_phi3_lora_8x3090.sh "$@"
