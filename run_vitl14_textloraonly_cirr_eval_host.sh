#!/usr/bin/env bash
set -euo pipefail

cd /data2/mingyu/composed_image_retrieval
export PYTHONPATH="/data2/mingyu/composed_image_retrieval:/data2/mingyu/composed_image_retrieval/src"

CUDA_VISIBLE_DEVICES="${1:-2}" \
/data2/mingyu/miniconda3/envs/torch/bin/python \
  /data2/mingyu/composed_image_retrieval/data/eval_branch_ablation_suite.py \
  --resume /data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp/checkpoints/epoch_0_step_1400.pt \
  --output-json /data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp/text_lora_only_step1400_eval.json \
  --variant text_lora_only \
  --cirr-gpu 0 \
  --model ViT-L/14 \
  --img2text-arch phi \
  --middle-dim 3072 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
  --cirr-batch-size 48 \
  --workers 2 \
  --retrieval-prompt-connector and \
  --datasets cirr \
  --name text_lora_only_cirr
