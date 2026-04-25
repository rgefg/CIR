#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/mingyu/composed_image_retrieval}"
RUN_ID="${RUN_ID:-decir_followup_20260425_173438}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/logs/${RUN_ID}}"
TMP_DIR="${TMP_DIR:-/tmp/${RUN_ID}}"
SOURCE_ONLY=1

source "${ROOT}/run_decir_followup_experiments_host.sh"

log_recovery() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${RESULT_DIR}/recovery_queue.log" "${RESULT_DIR}/queue.log"
}

cleanup_failed_loss_dirs() {
  local d
  for d in \
    "${ROOT}/logs/DistillCIR_Followup_ViTL14_SharedB12_fwd_only_CIRR" \
    "${ROOT}/logs/DistillCIR_Followup_ViTL14_SharedB12_fwd_rev_nozero_CIRR" \
    "${ROOT}/logs/DistillCIR_Followup_ViTL14_SharedB12_fwd_only_CIRCO_GeneCIS" \
    "${ROOT}/logs/DistillCIR_Followup_ViTL14_SharedB12_fwd_rev_nozero_CIRCO_GeneCIS"; do
    if [[ -d "${d}" && ! -e "${d}/checkpoints/epoch_0_step_${TRAIN_EPOCH_STEPS}.pt" ]]; then
      rm -rf "${d}"
      log_recovery "removed incomplete loss dir ${d}"
    fi
  done
}

run_loss_recovery_job() {
  local job_id="$1" dataset="$2" variant="$3" train_gpus="$4" port="$5"
  local result="${RESULT_DIR}/records/${job_id}_result.json"
  local log_path="${RESULT_DIR}/job_logs/${job_id}.log"
  run_job "${job_id}" "transition_loss_ablation" "${dataset}" "${variant}" "${train_gpus}" "${result}" "${log_path}" \
    train_loss_ablation_job "${dataset}" "${variant}" "${train_gpus}" "${port}" "${result}" "${log_path}"
}

main() {
  cleanup_failed_loss_dirs
  log_recovery "loss recovery batch 1 start: cirr variants on gpu pairs 3,4 and 5,6"
  run_loss_recovery_job "loss_cirr_fwd_only_rerun" "cirr" "fwd_only" "3,4" "7811" &
  pid1=$!
  run_loss_recovery_job "loss_cirr_fwd_rev_nozero" "cirr" "fwd_rev_nozero" "5,6" "7812" &
  pid2=$!
  wait "${pid1}"
  wait "${pid2}"
  log_recovery "loss recovery batch 1 done"

  log_recovery "loss recovery batch 2 start: multi variants on gpu pairs 3,4 and 5,6"
  run_loss_recovery_job "loss_multi_fwd_only" "multi" "fwd_only" "3,4" "7813" &
  pid3=$!
  run_loss_recovery_job "loss_multi_fwd_rev_nozero" "multi" "fwd_rev_nozero" "5,6" "7814" &
  pid4=$!
  wait "${pid3}"
  wait "${pid4}"
  log_recovery "loss recovery batch 2 done"
  log_recovery "loss recovery completed"
}

main "$@"
