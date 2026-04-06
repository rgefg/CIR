#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
RUN_NAME="DistillCIR_ParallelDualLoRA_BS256_Accum2_ViTB32_SEARLEPhi_That_NoDrop_FashionIQ600_PadCmp"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
PYTHON_BIN="/data2/mingyu/miniconda3/envs/torch/bin/python"

export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

mkdir -p /tmp/fashioniq600_padcmp
rm -f "${LOG_DIR}/fashioniq_official_nopad_200_400_600.jsonl" "${LOG_DIR}/fashioniq_official_pad_200_400_600.jsonl"

run_fashion_block() {
  local STEP="$1"
  local EVAL_GPU="$2"
  local MODE="$3"
  local TARGET_PAD_FLAG="$4"
  local OUT_JSONL="$5"

  local MERGED="/tmp/fashioniq600_padcmp/epoch0_step${STEP}_raw_plus_geoema_merged_${MODE}.pt"
  "${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${CKPT_DIR}/epoch_0_step_${STEP}.pt" \
    --ckpt-b "${CKPT_DIR}/epoch_0_step_${STEP}_geo_lora_ema.pt" \
    --output "${MERGED}" \
    --text-only \
    --base a \
    --weights 0.5 0.5 \
    --density 0.9

  local DLOG="${LOG_DIR}/fashioniq_${MODE}_step${STEP}_dress.log"
  local SLOG="${LOG_DIR}/fashioniq_${MODE}_step${STEP}_shirt.log"
  local TLOG="${LOG_DIR}/fashioniq_${MODE}_step${STEP}_toptee.log"

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --model "ViT-B/32" \
    --openai-pretrained \
    --img2text-arch phi \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
    --middle_dim 2048 \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --resume "${MERGED}" \
    --eval-mode fashion \
    --source-data dress \
    ${TARGET_PAD_FLAG} > "${DLOG}" 2>&1

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --model "ViT-B/32" \
    --openai-pretrained \
    --img2text-arch phi \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
    --middle_dim 2048 \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --resume "${MERGED}" \
    --eval-mode fashion \
    --source-data shirt \
    ${TARGET_PAD_FLAG} > "${SLOG}" 2>&1

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --model "ViT-B/32" \
    --openai-pretrained \
    --img2text-arch phi \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
    --middle_dim 2048 \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --resume "${MERGED}" \
    --eval-mode fashion \
    --source-data toptee \
    ${TARGET_PAD_FLAG} > "${TLOG}" 2>&1

  "${PYTHON_BIN}" - "${OUT_JSONL}" "${MODE}" "${STEP}" "${DLOG}" "${SLOG}" "${TLOG}" <<'PY'
import json, pathlib, re, sys
out_jsonl, mode, step_raw, dlog, slog, tlog = sys.argv[1:]
step = int(step_raw)
pat = re.compile(r"Eval composed FeatureR@1:\s*([0-9.]+)\s*R@5:\s*([0-9.]+)\s*R@10:\s*([0-9.]+)\s*R@50:\s*([0-9.]+)\s*R@100:\s*([0-9.]+)")
def parse(path):
    txt = pathlib.Path(path).read_text()
    m = pat.search(txt)
    if not m:
        raise SystemExit(f"missing composed metrics in {path}")
    r1, r5, r10, r50, r100 = map(float, m.groups())
    return {"R@1": r1, "R@5": r5, "R@10": r10, "R@50": r50, "R@100": r100}
record = {
    "mode": mode,
    "step": step,
    "fashioniq": {
        "dress": parse(dlog),
        "shirt": parse(slog),
        "toptee": parse(tlog),
    },
}
with open(out_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False))
PY

  rm -f "${MERGED}"
}

for STEP in 200 400 600; do
  run_fashion_block "${STEP}" "6" "nopad" "" "${LOG_DIR}/fashioniq_official_nopad_200_400_600.jsonl" &
  pid_nopad=$!
  run_fashion_block "${STEP}" "7" "pad" "--target-pad" "${LOG_DIR}/fashioniq_official_pad_200_400_600.jsonl" &
  pid_pad=$!
  wait "${pid_nopad}"
  wait "${pid_pad}"
done
