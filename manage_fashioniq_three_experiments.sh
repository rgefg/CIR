#!/bin/bash
set -euo pipefail

REPO_ROOT="/data2/mingyu/composed_image_retrieval"
TRAIN_SCRIPT="${REPO_ROOT}/train_with_dropout.sh"
LOG_ROOT="${REPO_ROOT}/logs"
QUEUE_LOG="${QUEUE_LOG:-/tmp/manage_fashioniq_three_experiments.log}"
POLL_SECONDS="${POLL_SECONDS:-180}"
STABLE_WAIT_SECONDS="${STABLE_WAIT_SECONDS:-180}"
GPU_MEMORY_IDLE_MAX_MB="${GPU_MEMORY_IDLE_MAX_MB:-1024}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_LOG}"
}

gpu_is_idle() {
  local gpu_idx="$1"
  local compute_pids mem_used
  compute_pids="$(nvidia-smi -i "${gpu_idx}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  mem_used="$(nvidia-smi -i "${gpu_idx}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ -z "${compute_pids}" ]] && [[ -n "${mem_used}" ]] && [[ "${mem_used}" =~ ^[0-9]+$ ]] && (( mem_used <= GPU_MEMORY_IDLE_MAX_MB ))
}

find_two_idle_gpus() {
  local idle=()
  local gpu_idx
  for gpu_idx in 0 1 2 3 4 5 6 7; do
    if gpu_is_idle "${gpu_idx}"; then
      idle+=("${gpu_idx}")
    fi
  done
  if [[ "${#idle[@]}" -ge 2 ]]; then
    printf '%s,%s\n' "${idle[0]}" "${idle[1]}"
    return 0
  fi
  return 1
}

wait_for_gpu_pair() {
  local pair=""
  while true; do
    pair="$(find_two_idle_gpus || true)"
    if [[ -n "${pair}" ]]; then
      printf '%s\n' "${pair}"
      return 0
    fi
    log "no two idle GPUs yet; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

launch_experiment() {
  local tag="$1"
  local prompt_style="$2"
  local reset_logit_scale="$3"

  local pair gpu_a gpu_b stamp run_name run_dir session_name wrapper
  pair="$(wait_for_gpu_pair)"
  IFS=, read -r gpu_a gpu_b <<< "${pair}"
  stamp="$(date '+%F_%H%M%S')"
  run_name="DistillCIR_ParallelDualLoRA_BS56_Accum8_EMA1000_QKV_StrictLoss_${tag}_${stamp}"
  run_dir="${LOG_ROOT}/${run_name}"
  session_name="fashioniq_${tag}_${stamp}"
  wrapper="/tmp/${run_name}_wrapper.sh"

  mkdir -p "${run_dir}"

  cat > "${wrapper}" <<EOF
#!/bin/bash
set -euo pipefail
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
INSTRUCTION_PROMPT_STYLE="${prompt_style}" \
RESET_LOGIT_SCALE="${reset_logit_scale}" \
ENABLE_MULTIDATASET_STANDALONE_WATCHER=0 \
ENABLE_MULTIDATASET_MERGED_WATCHER=0 \
RUN_POSTHOC_STANDALONE_EVAL=1 \
RUN_POSTHOC_MERGED_EVAL=1 \
bash "${TRAIN_SCRIPT}"
status=\$?
echo "\${status}" > "${run_dir}/queue_exit_code.txt"
exit "\${status}"
EOF
  chmod +x "${wrapper}"

  log "launching ${run_name} on GPUs ${gpu_a},${gpu_b} prompt_style=${prompt_style} reset_logit=${reset_logit_scale}"
  tmux new-session -d -s "${session_name}" "bash '${wrapper}'"

  sleep "${STABLE_WAIT_SECONDS}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    log "${run_name} exited before stabilization; cleaning failed directory"
    rm -f "${wrapper}"
    rm -rf "${run_dir}"
    return 1
  fi
  if [[ ! -f "${run_dir}/out.log" ]] || ! rg -q "Start epoch 0|Train Epoch:" "${run_dir}/out.log"; then
    log "${run_name} did not show a valid training start; killing session and cleaning"
    tmux kill-session -t "${session_name}" 2>/dev/null || true
    rm -f "${wrapper}"
    rm -rf "${run_dir}"
    return 1
  fi

  log "${run_name} is stable in tmux session ${session_name}"

  while tmux has-session -t "${session_name}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done

  rm -f "${wrapper}"

  if [[ -f "${run_dir}/queue_exit_code.txt" ]] && [[ "$(cat "${run_dir}/queue_exit_code.txt")" == "0" ]]; then
    log "${run_name} finished successfully"
    return 0
  fi

  log "${run_name} failed after launch; cleaning failed directory"
  rm -rf "${run_dir}"
  return 1
}

run_until_success() {
  local tag="$1"
  local prompt_style="$2"
  local reset_logit_scale="$3"
  while true; do
    if launch_experiment "${tag}" "${prompt_style}" "${reset_logit_scale}"; then
      return 0
    fi
    log "${tag} did not complete successfully; retrying after the next idle GPU window"
  done
}

log "starting sequential FashionIQ/GeneCIS overnight queue"
run_until_success "Exp1_NoDrop" "single" "1"
run_until_success "Exp2_PromptAlign" "duplicate_and" "1"
run_until_success "Exp3_NoResetLogit" "single" "0"
log "all queued experiments completed"
