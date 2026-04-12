#!/usr/bin/env bash
set -euo pipefail

cd /data2/mingyu/composed_image_retrieval
export PYTHONPATH="/data2/mingyu/composed_image_retrieval:/data2/mingyu/composed_image_retrieval/src"

CUDA_VISIBLE_DEVICES="${1:-4}" \
/data2/mingyu/miniconda3/envs/torch/bin/python \
  /data2/mingyu/composed_image_retrieval/data/eval_branch_ablation_suite.py \
  --resume /data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp/checkpoints/epoch_0_step_1400.pt \
  --output-json /data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp/text_lora_only_step1400_eval.json \
  --variant text_lora_only \
  --suite-gpu 0 \
  --model ViT-L/14 \
  --img2text-arch phi \
  --middle-dim 3072 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
  --suite-batch-size 32 \
  --workers 2 \
  --genecis-batch-size 32 \
  --retrieval-prompt-connector that \
  --datasets circo,genecis \
  --name text_lora_only_multidata
