#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

TEACHER_CACHE="${TEACHER_CACHE:-/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_cc3m_cache}"

python src/validate_distillcir_ready.py \
  --cc3m-cir-jsonl "${CC3M_JSONL:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl}" \
  --wds-shards "${WDS_SHARDS:-/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar}" \
  --pic2word-pretrained "${PIC2WORD_CKPT:-/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt}" \
  --teacher-cache "$TEACHER_CACHE"

./train_with_dropout.sh --dry-run-steps "${DRY_RUN_STEPS:-2}" --name "${RUN_NAME:-DistillCIR_DryRun}"
