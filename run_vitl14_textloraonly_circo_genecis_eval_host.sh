#!/usr/bin/env bash
set -euo pipefail

cd /data2/mingyu/composed_image_retrieval
export PYTHONPATH="/data2/mingyu/composed_image_retrieval:/data2/mingyu/composed_image_retrieval/src"

GPU_ID="${1:-4}"
DATASETS="${2:-circo,genecis}"
OUTPUT_JSON="${3:-/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp/text_lora_only_step1400_eval.json}"
RUN_NAME="${4:-text_lora_only_multidata}"
SUITE_BATCH_SIZE="${5:-32}"
GENECIS_BATCH_SIZE="${6:-32}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
/data2/mingyu/miniconda3/envs/torch/bin/python \
  /data2/mingyu/composed_image_retrieval/data/eval_branch_ablation_suite.py \
  --resume /data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp/checkpoints/epoch_0_step_1400.pt \
  --output-json "${OUTPUT_JSON}" \
  --variant text_lora_only \
  --suite-gpu 0 \
  --model ViT-L/14 \
  --img2text-arch phi \
  --middle-dim 3072 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
  --suite-batch-size "${SUITE_BATCH_SIZE}" \
  --workers 2 \
  --genecis-batch-size "${GENECIS_BATCH_SIZE}" \
  --retrieval-prompt-connector that \
  --datasets "${DATASETS}" \
  --name "${RUN_NAME}"
