#!/bin/bash
set -euo pipefail

REPO_ROOT="/data2/mingyu/composed_image_retrieval"
TRAIN_SCRIPT="${REPO_ROOT}/train_with_dropout.sh"
LOG_ROOT="${REPO_ROOT}/logs"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-1800}"
STABLE_WAIT_SECONDS="${STABLE_WAIT_SECONDS:-120}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-DistillCIR_ParallelDualLoRA_BS56_Accum8_EMA1000_QKV_StrictLoss_NoDrop_FashionGeneCIS_Exp1}"
MONITOR_LOG="${MONITOR_LOG:-/tmp/monitor_fashioniq_exp1.log}"

gpu_has_compute_process() {
  local gpu_idx="$1"
  local out
  out="$(nvidia-smi -i "${gpu_idx}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ -n "${out}" ]]
}

find_two_free_gpus() {
  local free=()
  local idx
  for idx in 0 1 2 3 4 5 6 7; do
    if ! gpu_has_compute_process "${idx}"; then
      free+=("${idx}")
    fi
  done
  if [[ "${#free[@]}" -ge 2 ]]; then
    echo "${free[0]},${free[1]}"
    return 0
  fi
  return 1
}

echo "[$(date '+%F %T')] monitor started" >> "${MONITOR_LOG}"

while true; do
  pair="$(find_two_free_gpus || true)"
  if [[ -n "${pair}" ]]; then
    IFS=, read -r gpu_a gpu_b <<< "${pair}"
    stamp="$(date '+%F_%H%M%S')"
    run_name="${RUN_NAME_PREFIX}_${stamp}"
    launcher_log="/tmp/${run_name}_launcher.log"
    run_dir="${LOG_ROOT}/${run_name}"

    echo "[$(date '+%F %T')] launching ${run_name} on GPUs ${gpu_a},${gpu_b}" >> "${MONITOR_LOG}"

    (
      cd "${REPO_ROOT}"
      TRAIN_CUDA_DEVICES="${gpu_a},${gpu_b}" \
      POSTHOC_STANDALONE_GPU="${gpu_a}" \
      POSTHOC_MERGED_GPU="${gpu_b}" \
      RUN_NAME="${run_name}" \
      TRAIN_EPOCH_STEPS=1000 \
      SAVE_STEP_START=200 \
      SAVE_STEP_END=1000 \
      SAVE_STEP_INTERVAL=200 \
      INSTRUCTION_DROPOUT_PROB=0.0 \
      ENABLE_MULTIDATASET_STANDALONE_WATCHER=0 \
      ENABLE_MULTIDATASET_MERGED_WATCHER=0 \
      RUN_POSTHOC_STANDALONE_EVAL=1 \
      RUN_POSTHOC_MERGED_EVAL=1 \
      bash "${TRAIN_SCRIPT}"
    ) > "${launcher_log}" 2>&1 &
    launcher_pid=$!

    sleep "${STABLE_WAIT_SECONDS}"

    if pgrep -af "${run_name}|src/main.py" >/dev/null 2>&1 && [[ -f "${run_dir}/out.log" ]]; then
      echo "[$(date '+%F %T')] ${run_name} is stable, launcher_pid=${launcher_pid}" >> "${MONITOR_LOG}"
      echo "${run_name}" > /tmp/fashioniq_exp1_active_run.txt
      echo "${gpu_a},${gpu_b}" > /tmp/fashioniq_exp1_active_gpus.txt
      exit 0
    fi

    echo "[$(date '+%F %T')] ${run_name} failed to stabilize, cleaning ${run_dir}" >> "${MONITOR_LOG}"
    python -c "import shutil; shutil.rmtree('${run_dir}', ignore_errors=True)"
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
