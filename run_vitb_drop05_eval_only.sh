#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB16_EMA1400_And_Drop0p5_GeneCIS_CIRCO"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
GENECIS_JSONL="${LOG_DIR}/genecis_merged.jsonl"
FINAL_MERGED="/tmp/${RUN_NAME}_step1400_merged.pt"
GENECIS_GPU="${GENECIS_GPU:-4}"
CIRCO_GPU="${CIRCO_GPU:-5}"

: > "${GENECIS_JSONL}"

for STEP in 600 800 1000 1200 1400; do
  BASE_CKPT="${CKPT_DIR}/epoch_0_step_${STEP}.pt"
  GEO_CKPT="${CKPT_DIR}/epoch_0_step_${STEP}_geo_lora_ema.pt"
  MERGED_CKPT="/tmp/${RUN_NAME}_step${STEP}_merged.pt"
  OUT_JSON="/tmp/${RUN_NAME}_step${STEP}_genecis.json"

  "${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${BASE_CKPT}" \
    --ckpt-b "${GEO_CKPT}" \
    --output "${MERGED_CKPT}" \
    --weights 0.5 0.5 \
    --density 0.9 \
    --text-only \
    --base a \
    --alpha-a 16 --rank-a 64 \
    --alpha-b 16 --rank-b 64

  CUDA_VISIBLE_DEVICES="${GENECIS_GPU}" "${PYTHON_BIN}" "${ROOT}/data/eval_multidataset_suite.py" \
    --resume "${MERGED_CKPT}" \
    --output-json "${OUT_JSON}" \
    --gpu 0 \
    --model "ViT-B/16" \
    --batch-size 32 \
    --workers 2 \
    --genecis-batch-size 32 \
    --datasets "genecis"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

out_path = Path("${OUT_JSON}")
jsonl_path = Path("${GENECIS_JSONL}")
record = {"step": ${STEP}}
record.update(json.loads(out_path.read_text(encoding="utf-8")))
with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\\n")
PY

  if [[ "${STEP}" != "1400" ]]; then
    rm -f "${MERGED_CKPT}"
  fi
  rm -f "${OUT_JSON}"
done

"${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
  --ckpt-a "${CKPT_DIR}/epoch_0_step_1400.pt" \
  --ckpt-b "${CKPT_DIR}/epoch_0_step_1400_geo_lora_ema.pt" \
  --output "${FINAL_MERGED}" \
  --weights 0.5 0.5 \
  --density 0.9 \
  --text-only \
  --base a \
  --alpha-a 16 --rank-a 64 \
  --alpha-b 16 --rank-b 64

mkdir -p "${LOG_DIR}/circo_final_step1400"
CUDA_VISIBLE_DEVICES="${CIRCO_GPU}" "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
  --resume "${FINAL_MERGED}" \
  --openai-pretrained \
  --model "ViT-B/16" \
  --eval-mode circo \
  --gpu 0 \
  --batch-size 32 \
  --workers 2 \
  --logs "${LOG_DIR}" \
  --name "circo_final_step1400"

cp "${LOG_DIR}/circo_final_step1400/circo_submission.json" "${LOG_DIR}/circo_submission_step1400.json"
rm -rf "${LOG_DIR}/circo_final_step1400"
rm -f "${FINAL_MERGED}"
