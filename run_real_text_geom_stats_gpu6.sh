#!/usr/bin/env bash
set -euo pipefail

cd /data2/mingyu/composed_image_retrieval

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

OUT_DIR="logs/DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_And_NoDrop_SharedB_CIRR_MergeCmp"
OUT_JSON="${OUT_DIR}/real_text_geom_stats_2048.json"

/data2/mingyu/miniconda3/envs/torch/bin/python data/real_text_geom_stats.py \
  --retrieval-ckpt "${OUT_DIR}/checkpoints/epoch_0_step_1400.pt" \
  --merged-ckpt "/tmp/vitb_sharedb_cirr_step1400_ties_realstats.pt" \
  --output-json "${OUT_JSON}" \
  --model "ViT-B/32" \
  --img2text-arch phi \
  --middle-dim 2048 \
  --gpu 0 \
  --num-samples 2048 \
  --batch-size 128 \
  --workers 4 \
  --wds-shards '/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar' \
  --cc3m-cir-jsonl /data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2_with_reverse.jsonl
