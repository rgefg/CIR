#!/bin/bash
set -euo pipefail

REPO_ROOT="/data2/mingyu/composed_image_retrieval"
TRAIN_SCRIPT="${REPO_ROOT}/train_with_dropout.sh"
LOG_ROOT="${REPO_ROOT}/logs"
QUEUE_LOG="${QUEUE_LOG:-/tmp/manage_fashioniq_three_experiments.log}"
POLL_SECONDS="${POLL_SECONDS:-180}"
STABLE_WAIT_SECONDS="${STABLE_WAIT_SECONDS:-90}"
GPU_MEMORY_IDLE_MAX_MB="${GPU_MEMORY_IDLE_MAX_MB:-1024}"

declare -a EXP_TAGS=("Exp1_NoDrop" "Exp2_PromptAlign" "Exp3_NoResetLogit")
declare -a EXP_PROMPT_STYLES=("single" "duplicate_and" "single")
declare -a EXP_RESET_LOGIT=("1" "1" "0")

declare -a EXP_STATUS=("pending" "pending" "pending")
declare -a EXP_SESSION=("" "" "")
declare -a EXP_RUN=("" "" "")
declare -a EXP_GPUS=("" "" "")

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

first_pending_index() {
  local idx
  for idx in "${!EXP_TAGS[@]}"; do
    if [[ "${EXP_STATUS[$idx]}" == "pending" ]]; then
      printf '%s\n' "${idx}"
      return 0
    fi
  done
  return 1
}

all_done() {
  local idx
  for idx in "${!EXP_TAGS[@]}"; do
    if [[ "${EXP_STATUS[$idx]}" != "done" ]]; then
      return 1
    fi
  done
  return 0
}

cleanup_failed_experiment() {
  local idx="$1"
  local run_dir="${LOG_ROOT}/${EXP_RUN[$idx]}"
  local session_name="${EXP_SESSION[$idx]}"
  tmux kill-session -t "${session_name}" 2>/dev/null || true
  rm -f "/tmp/${EXP_RUN[$idx]}_wrapper.sh"
  rm -rf "${run_dir}"
  EXP_STATUS[$idx]="pending"
  EXP_SESSION[$idx]=""
  EXP_RUN[$idx]=""
  EXP_GPUS[$idx]=""
}

mark_done_experiment() {
  local idx="$1"
  rm -f "/tmp/${EXP_RUN[$idx]}_wrapper.sh"
  EXP_STATUS[$idx]="done"
}

poll_running_experiments() {
  local idx run_dir exit_code
  for idx in "${!EXP_TAGS[@]}"; do
    if [[ "${EXP_STATUS[$idx]}" != "running" ]]; then
      continue
    fi
    if tmux has-session -t "${EXP_SESSION[$idx]}" 2>/dev/null; then
      continue
    fi
    run_dir="${LOG_ROOT}/${EXP_RUN[$idx]}"
    exit_code=""
    if [[ -f "${run_dir}/queue_exit_code.txt" ]]; then
      exit_code="$(cat "${run_dir}/queue_exit_code.txt")"
    fi
    if [[ "${exit_code}" == "0" ]]; then
      log "${EXP_RUN[$idx]} finished successfully on GPUs ${EXP_GPUS[$idx]}"
      mark_done_experiment "${idx}"
    else
      log "${EXP_RUN[$idx]} failed after launch on GPUs ${EXP_GPUS[$idx]}; cleaning and re-queueing"
      cleanup_failed_experiment "${idx}"
    fi
  done
}

launch_experiment_index() {
  local idx="$1"
  local pair="$2"
  local gpu_a gpu_b stamp run_name run_dir session_name wrapper prompt_style reset_logit
  IFS=, read -r gpu_a gpu_b <<< "${pair}"

  prompt_style="${EXP_PROMPT_STYLES[$idx]}"
  reset_logit="${EXP_RESET_LOGIT[$idx]}"
  stamp="$(date '+%F_%H%M%S')"
  run_name="DistillCIR_ParallelDualLoRA_BS56_Accum8_EMA1000_QKV_StrictLoss_${EXP_TAGS[$idx]}_${stamp}"
  run_dir="${LOG_ROOT}/${run_name}"
  session_name="fashioniq_${EXP_TAGS[$idx]}_${stamp}"
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
RESET_LOGIT_SCALE="${reset_logit}" \
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

  log "launching ${run_name} on GPUs ${gpu_a},${gpu_b} prompt_style=${prompt_style} reset_logit=${reset_logit}"
  tmux new-session -d -s "${session_name}" "bash '${wrapper}'"

  EXP_STATUS[$idx]="starting"
  EXP_SESSION[$idx]="${session_name}"
  EXP_RUN[$idx]="${run_name}"
  EXP_GPUS[$idx]="${gpu_a},${gpu_b}"

  sleep "${STABLE_WAIT_SECONDS}"

  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    log "${run_name} exited before stabilization"
    cleanup_failed_experiment "${idx}"
    return 1
  fi
  if [[ ! -f "${run_dir}/out.log" ]] || ! rg -q "Start epoch 0|Train Epoch:" "${run_dir}/out.log"; then
    log "${run_name} did not show a valid training start"
    cleanup_failed_experiment "${idx}"
    return 1
  fi

  EXP_STATUS[$idx]="running"
  log "${run_name} is stable in tmux session ${session_name}"
  return 0
}

log "starting parallel FashionIQ/GeneCIS overnight queue"

while true; do
  poll_running_experiments

  if all_done; then
    log "all queued experiments completed"
    exit 0
  fi

  launched_any=0
  while true; do
    pending_idx="$(first_pending_index || true)"
    if [[ -z "${pending_idx}" ]]; then
      break
    fi
    pair="$(find_two_idle_gpus || true)"
    if [[ -z "${pair}" ]]; then
      break
    fi
    if launch_experiment_index "${pending_idx}" "${pair}"; then
      launched_any=1
      continue
    fi
  done

  if [[ "${launched_any}" == "0" ]]; then
    status_line=()
    for idx in "${!EXP_TAGS[@]}"; do
      status_line+=("${EXP_TAGS[$idx]}=${EXP_STATUS[$idx]}")
    done
    log "waiting for resources; statuses: ${status_line[*]}"
  fi
  sleep "${POLL_SECONDS}"
done
