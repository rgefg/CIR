#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

WAIT_HOURS="${WAIT_HOURS:-6}"

VITB_RUN="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTB32_SEARLEPhi_And_Drop0p5_GeneCIS"
VITL_RUN="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_CIRR"

VITB_DIR="${ROOT}/logs/${VITB_RUN}"
VITL_DIR="${ROOT}/logs/${VITL_RUN}"

OUT_DIR="${ROOT}/logs/delayed_searle_eval_$(date +%F_%H%M%S)"
mkdir -p "${OUT_DIR}"

exec > >(tee -a "${OUT_DIR}/run.log") 2>&1

wait_for_file() {
  local path="$1"
  local label="$2"
  while [[ ! -f "${path}" ]]; do
    echo "[$(date '+%F %T')] waiting for ${label}: ${path}"
    sleep 60
  done
  echo "[$(date '+%F %T')] found ${label}: ${path}"
}

run_merge() {
  local ckpt_a="$1"
  local ckpt_b="$2"
  local output="$3"

  "${PYTHON_BIN}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${ckpt_a}" \
    --ckpt-b "${ckpt_b}" \
    --output "${output}" \
    --weights 0.5 0.5 \
    --density 0.9 \
    --text-only \
    --base a \
    --alpha-a 16 \
    --rank-a 64 \
    --alpha-b 16 \
    --rank-b 64 \
    --slim-output
}

run_circo() {
  local merged="${OUT_DIR}/vitb32_step1400_merged.pt"
  local eval_name="vitb32_circo_step1400"
  run_merge \
    "${VITB_DIR}/checkpoints/epoch_0_step_1400.pt" \
    "${VITB_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
    "${merged}"

  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --resume "${merged}" \
    --openai-pretrained \
    --model "ViT-B/32" \
    --img2text-arch phi \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-B32.pt \
    --middle_dim 2048 \
    --eval-mode circo \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --logs "${OUT_DIR}" \
    --name "${eval_name}" > "${OUT_DIR}/circo_eval.log" 2>&1

  cp "${OUT_DIR}/${eval_name}/circo_submission.json" "${OUT_DIR}/circo_submission_step1400.json"
}

run_cirr_test() {
  local merged="${OUT_DIR}/vitl14_step1400_merged.pt"
  local eval_name="vitl14_cirr_test_step1400"
  local result_dir="${OUT_DIR}/cirr_test_step1400"
  run_merge \
    "${VITL_DIR}/checkpoints/epoch_0_step_1400.pt" \
    "${VITL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
    "${merged}"

  CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --resume "${merged}" \
    --openai-pretrained \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
    --middle_dim 3072 \
    --eval-mode cirr_test \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --logs "${OUT_DIR}" \
    --name "${eval_name}" \
    --cirr-output-dir "${result_dir}" > "${OUT_DIR}/cirr_test_eval.log" 2>&1

  cp "${result_dir}/composed.json" "${OUT_DIR}/cirr_step1400_composed.json"
  cp "${result_dir}/subset_composed.json" "${OUT_DIR}/cirr_step1400_subset_composed.json"
}

run_fashion_one() {
  local gpu="$1"
  local cloth="$2"
  local merged="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --resume "${merged}" \
    --openai-pretrained \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
    --middle_dim 3072 \
    --eval-mode fashion \
    --source-data "${cloth}" \
    --target-pad \
    --gpu 0 \
    --batch-size 32 \
    --workers 2 \
    --logs "${OUT_DIR}" \
    --name "vitl14_fashion_step600_${cloth}" > "${OUT_DIR}/fashioniq_${cloth}.log" 2>&1
}

run_fashioniq() {
  local merged="${OUT_DIR}/vitl14_step600_merged.pt"
  run_merge \
    "${VITL_DIR}/checkpoints/epoch_0_step_600.pt" \
    "${VITL_DIR}/checkpoints/epoch_0_step_600_geo_lora_ema.pt" \
    "${merged}"

  run_fashion_one 4 dress "${merged}" &
  local dress_pid=$!
  (
    run_fashion_one 5 shirt "${merged}"
    run_fashion_one 5 toptee "${merged}"
  ) &
  local shirt_toptee_pid=$!

  wait "${dress_pid}"
  wait "${shirt_toptee_pid}"
}

echo "[$(date '+%F %T')] delayed eval script created by winter run"
echo "[$(date '+%F %T')] sleeping for ${WAIT_HOURS} hours before launching evals"
sleep "${WAIT_HOURS}h"

wait_for_file "${VITB_DIR}/checkpoints/epoch_0_step_1400.pt" "ViT-B raw step1400"
wait_for_file "${VITB_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" "ViT-B geo ema step1400"
wait_for_file "${VITL_DIR}/checkpoints/epoch_0_step_1400.pt" "ViT-L raw step1400"
wait_for_file "${VITL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" "ViT-L geo ema step1400"
wait_for_file "${VITL_DIR}/checkpoints/epoch_0_step_600.pt" "ViT-L raw step600"
wait_for_file "${VITL_DIR}/checkpoints/epoch_0_step_600_geo_lora_ema.pt" "ViT-L geo ema step600"

echo "[$(date '+%F %T')] launching CIRCO on GPU2"
run_circo &
circo_pid=$!

echo "[$(date '+%F %T')] launching CIRR test on GPU3"
run_cirr_test &
cirr_pid=$!

echo "[$(date '+%F %T')] launching FashionIQ on GPU4/5"
run_fashioniq &
fashion_pid=$!

wait "${circo_pid}"
wait "${cirr_pid}"
wait "${fashion_pid}"

echo "[$(date '+%F %T')] all delayed eval jobs finished"
echo "outputs:"
echo "  CIRCO:   ${OUT_DIR}/circo_submission_step1400.json"
echo "  CIRR:    ${OUT_DIR}/cirr_step1400_composed.json"
echo "  CIRRsub: ${OUT_DIR}/cirr_step1400_subset_composed.json"
echo "  FashionIQ logs:"
echo "    ${OUT_DIR}/fashioniq_dress.log"
echo "    ${OUT_DIR}/fashioniq_shirt.log"
echo "    ${OUT_DIR}/fashioniq_toptee.log"
