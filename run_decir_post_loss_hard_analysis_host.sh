#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/mingyu/composed_image_retrieval}"
RUN_ID="${RUN_ID:-decir_followup_20260425_173438}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/logs/${RUN_ID}}"
TMP_DIR="${TMP_DIR:-/tmp/${RUN_ID}}"
POLL_SECONDS="${POLL_SECONDS:-120}"
HARD_CIRR_BATCH_SIZE="${HARD_CIRR_BATCH_SIZE:-24}"
HARD_GENECIS_BATCH_SIZE="${HARD_GENECIS_BATCH_SIZE:-16}"
SOURCE_ONLY=1

source "${ROOT}/run_decir_followup_experiments_host.sh"

post_log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${RESULT_DIR}/post_hard_queue.log" "${RESULT_DIR}/queue.log" >&2
}

wait_for_loss_recovery() {
  post_log "post-loss hard analysis watcher started"
  while true; do
    if [[ -f "${RESULT_DIR}/recovery_queue.log" ]] && grep -q "loss recovery completed" "${RESULT_DIR}/recovery_queue.log"; then
      post_log "loss recovery completed; preparing hard analysis"
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

select_one_idle_gpu() {
  local -a idle=()
  local gpu
  while true; do
    mapfile -t idle < <(discover_idle_gpus)
    if (( ${#idle[@]} > 0 )); then
      gpu="${idle[0]}"
      post_log "selected idle gpu ${gpu} for hard analysis"
      printf '%s\n' "${gpu}"
      return 0
    fi
    post_log "no idle gpu for hard analysis yet; waiting"
    sleep "${POLL_SECONDS}"
  done
}

run_post_hard_analysis() {
  mkdir -p "${RESULT_DIR}/hard_analysis" "${RESULT_DIR}/records" "${RESULT_DIR}/job_logs"
  local gpu="$1"
  CIRR_BATCH_SIZE="${HARD_CIRR_BATCH_SIZE}"
  GENECIS_BATCH_SIZE="${HARD_GENECIS_BATCH_SIZE}"
  SUITE_BATCH_SIZE="${HARD_CIRR_BATCH_SIZE}"
  run_job "hard_analysis_recovery" "hard_analysis" "cirr_circo_genecis" "strict_merged_wins" "${gpu}" \
    "${RESULT_DIR}/records/hard_analysis_result.json" "${RESULT_DIR}/job_logs/hard_analysis_recovery.wrapper.log" run_hard_analysis "${gpu}"
}

write_followup_summary() {
  "${PY}" "${ROOT}/data/summarize_decir_followup.py" \
    --result-dir "${RESULT_DIR}" \
    --output "${RESULT_DIR}/FOLLOWUP_RESULTS_SUMMARY.md"
  post_log "wrote ${RESULT_DIR}/FOLLOWUP_RESULTS_SUMMARY.md"
}

main() {
  wait_for_loss_recovery
  gpu="$(select_one_idle_gpu)"
  post_log "hard analysis start on gpu ${gpu}"
  run_post_hard_analysis "${gpu}"
  post_log "hard analysis finished"
  write_followup_summary
  nvidia-smi > "${RESULT_DIR}/host_gpu_snapshot_post_hard_end.txt" 2>&1 || true
  post_log "post-loss hard analysis completed"
}

main "$@"
