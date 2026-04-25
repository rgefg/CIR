#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONNECTOR=that \
REASON_CONNECTOR=that \
RUN_NAME="${RUN_NAME:-DistillCIR_Repro_ViTL14_that_8x3090}" \
./train_with_dropout.sh "$@"

