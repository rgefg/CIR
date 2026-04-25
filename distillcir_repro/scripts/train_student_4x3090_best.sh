#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Best measured 4-card student setup on this host:
# physical GPUs 0,1,2,7; per-GPU batch 32; global contrastive batch 128.
# Batch 48 ran into CUDA OOM during teardown, so 32 is the practical ceiling.

PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1,2,7}" \
PER_GPU_BATCH="${PER_GPU_BATCH:-32}" \
ACCUM_STEPS="${ACCUM_STEPS:-1}" \
WORKERS="${WORKERS:-4}" \
EPOCHS="${EPOCHS:-2}" \
WDS_EPOCH_STEPS="${WDS_EPOCH_STEPS:-16837}" \
LR="${LR:-1e-5}" \
WARMUP="${WARMUP:-1000}" \
AMP_INIT_SCALE="${AMP_INIT_SCALE:-1024}" \
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}" \
CONNECTOR="${CONNECTOR:-and}" \
REASON_CONNECTOR="${REASON_CONNECTOR:-that}" \
RUN_NAME="${RUN_NAME:-DistillCIR_Repro_ViTL14_4x3090_BS128_${CONNECTOR}}" \
./train_with_dropout.sh "$@"

