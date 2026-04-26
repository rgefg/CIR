#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Best measured 4-card student setup on this host:
# physical GPUs 0,1,2,3; per-GPU batch 48; global contrastive batch 192.
# This keeps the DistillCIR paper settings except for the smaller feasible batch.

PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1,2,3}" \
PER_GPU_BATCH="${PER_GPU_BATCH:-48}" \
ACCUM_STEPS="${ACCUM_STEPS:-1}" \
WORKERS="${WORKERS:-4}" \
EPOCHS="${EPOCHS:-2}" \
WDS_EPOCH_STEPS="${WDS_EPOCH_STEPS:-11225}" \
LR="${LR:-2e-5}" \
WARMUP="${WARMUP:-1000}" \
AMP_INIT_SCALE="${AMP_INIT_SCALE:-1024}" \
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}" \
CONNECTOR="${CONNECTOR:-and}" \
REASON_CONNECTOR="${REASON_CONNECTOR:-that}" \
RUN_NAME="${RUN_NAME:-DistillCIR_Repro_ViTL14_4x3090_BS192_${CONNECTOR}}" \
./train_with_dropout.sh "$@"
