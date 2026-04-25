#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/mingyu/composed_image_retrieval}"
RUN_ID="${RUN_ID:-decir_followup_20260425_173438}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/logs/${RUN_ID}}"
TMP_DIR="${TMP_DIR:-/tmp/${RUN_ID}}"
ACCEL_GPUS="${ACCEL_GPUS:-3,4,5,6}"
MAX_ACTIVE_GPUS="${MAX_ACTIVE_GPUS:-4}"
CONTROL_PARENT_PID="${CONTROL_PARENT_PID:-2949885}"
SOURCE_ONLY=1

source "${ROOT}/run_decir_followup_experiments_host.sh"

ACCEL_STATE_DIR="${RESULT_DIR}/accelerator_state"
mkdir -p "${ACCEL_STATE_DIR}"

IFS=',' read -r -a GPU_POOL <<< "${ACCEL_GPUS}"
declare -A USED_GPU=()
declare -A PID_JOB=()
declare -A PID_TASK=()
declare -A PID_DATASET=()
declare -A PID_LABEL=()
declare -A PID_GPUS=()
declare -A PID_RESULT=()
declare -A PID_LOG=()
declare -A PID_STARTED=()
declare -A ACTIVE_JOB=()
declare -A FAILED_JOB=()

status_has_job() {
  local job_id="$1"
  [[ -f "${STATUS_TSV}" ]] && awk -F '\t' -v id="${job_id}" '$1 == id && $6 == "ok" {found=1} END {exit !found}' "${STATUS_TSV}"
}

status_has_terminal_job() {
  local job_id="$1"
  [[ -f "${STATUS_TSV}" ]] && awk -F '\t' -v id="${job_id}" '$1 == id && ($6 == "ok" || $6 == "failed") {found=1} END {exit !found}' "${STATUS_TSV}"
}

json_result_exists() {
  local path="$1"
  [[ -s "${path}" ]] && "${PY}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

try:
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
PY
}

mark_gpus_used() {
  local csv="$1" gpu
  IFS=',' read -r -a parts <<< "${csv}"
  for gpu in "${parts[@]}"; do
    USED_GPU["${gpu}"]=1
  done
}

mark_gpus_free() {
  local csv="$1" gpu
  IFS=',' read -r -a parts <<< "${csv}"
  for gpu in "${parts[@]}"; do
    unset "USED_GPU[${gpu}]"
  done
}

free_gpus_for() {
  local need="$1" gpu out=()
  for gpu in "${GPU_POOL[@]}"; do
    if [[ -z "${USED_GPU[${gpu}]:-}" ]]; then
      out+=("${gpu}")
      if (( ${#out[@]} == need )); then
        local IFS=,
        printf '%s\n' "${out[*]}"
        return 0
      fi
    fi
  done
  return 1
}

active_gpu_count() {
  printf '%s\n' "${!USED_GPU[@]}" | awk 'NF {n++} END {print n+0}'
}

reserve_external_rank_suite_k48() {
  local result="${RESULT_DIR}/records/rank_suite_k48_result.json"
  if status_has_job "rank_suite_k48"; then
    return 0
  fi
  if [[ -n "${CONTROL_PARENT_PID}" ]] && ps -p "${CONTROL_PARENT_PID}" > /dev/null 2>&1; then
    kill -STOP "${CONTROL_PARENT_PID}" 2>/dev/null || true
    log "accelerator paused original runner pid=${CONTROL_PARENT_PID}; active rank_suite_k48 remains on gpu=4"
  fi
  if ! json_result_exists "${result}"; then
    USED_GPU["4"]=1
  fi
}

record_external_rank_suite_k48_if_done() {
  local result="${RESULT_DIR}/records/rank_suite_k48_result.json"
  local log_path="${RESULT_DIR}/job_logs/rank_suite_k48.log"
  local pid_on_gpu
  if status_has_job "rank_suite_k48"; then
    mark_gpus_free "4"
    return 0
  fi
  if ! json_result_exists "${result}"; then
    return 1
  fi
  pid_on_gpu="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -x '3028552' || true)"
  if [[ -n "${pid_on_gpu}" ]]; then
    return 1
  fi
  local started finished
  started="2026-04-25 19:51:55"
  finished="$(date '+%F %T')"
  append_status "rank_suite_k48" "rank_ablation" "circo_genecis" "svd_k48" "4" "ok" "0" "${started}" "${finished}"
  write_record "rank_suite_k48" "rank_ablation" "circo_genecis" "svd_k48" "4" "ok" "0" "${started}" "${finished}" "${result}" "${log_path}"
  log "accelerator recorded external rank_suite_k48 and released gpu=4"
  rm -f "${TMP_DIR}/rank48_suite.pt"
  if [[ -n "${CONTROL_PARENT_PID}" ]] && ps -p "${CONTROL_PARENT_PID}" > /dev/null 2>&1; then
    kill -KILL "${CONTROL_PARENT_PID}" 2>/dev/null || true
    log "accelerator stopped original runner pid=${CONTROL_PARENT_PID}; remaining jobs are managed by accelerator"
  fi
  mark_gpus_free "4"
  return 0
}

start_job() {
  local job_id="$1" task="$2" dataset="$3" label="$4" need="$5" kind="$6" arg1="${7:-}" arg2="${8:-}"
  local gpus result log_path started
  if status_has_terminal_job "${job_id}" || [[ -n "${ACTIVE_JOB[${job_id}]:-}" ]] || [[ -n "${FAILED_JOB[${job_id}]:-}" ]]; then
    return 0
  fi
  gpus="$(free_gpus_for "${need}")" || return 1
  if (( $(active_gpu_count) + need > MAX_ACTIVE_GPUS )); then
    return 1
  fi
  result="${RESULT_DIR}/records/${job_id}_result.json"
  log_path="${RESULT_DIR}/job_logs/${job_id}.log"
  started="$(date '+%F %T')"
  mark_gpus_used "${gpus}"
  log "ACCEL_START ${job_id} task=${task} dataset=${dataset} label=${label} gpu=${gpus}"
  (
    set +e
    case "${kind}" in
      rank_cirr)
        rank_cirr_job "${gpus}" "${arg1}" "${result}" "${log_path}"
        ;;
      rank_suite)
        rank_suite_job "${gpus}" "${arg1}" "${result}" "${log_path}"
        ;;
      hard)
        run_hard_analysis "${gpus}" > "${log_path}" 2>&1
        ;;
      loss)
        local port="${arg2}"
        train_loss_ablation_job "${dataset}" "${label}" "${gpus}" "${port}" "${result}" "${log_path}"
        ;;
      *)
        echo "unknown accelerator job kind: ${kind}" >&2
        exit 2
        ;;
    esac
    echo "$?" > "${ACCEL_STATE_DIR}/${job_id}.exit"
  ) &
  local pid=$!
  PID_JOB["${pid}"]="${job_id}"
  PID_TASK["${pid}"]="${task}"
  PID_DATASET["${pid}"]="${dataset}"
  PID_LABEL["${pid}"]="${label}"
  PID_GPUS["${pid}"]="${gpus}"
  PID_RESULT["${pid}"]="${result}"
  PID_LOG["${pid}"]="${log_path}"
  PID_STARTED["${pid}"]="${started}"
  ACTIVE_JOB["${job_id}"]="${pid}"
  return 0
}

reap_done_jobs() {
  local pid job_id task dataset label gpus result log_path started finished rc status
  for pid in "${!PID_JOB[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    set +e
    wait "${pid}"
    set -e
    job_id="${PID_JOB[${pid}]}"
    task="${PID_TASK[${pid}]}"
    dataset="${PID_DATASET[${pid}]}"
    label="${PID_LABEL[${pid}]}"
    gpus="${PID_GPUS[${pid}]}"
    result="${PID_RESULT[${pid}]}"
    log_path="${PID_LOG[${pid}]}"
    started="${PID_STARTED[${pid}]}"
    finished="$(date '+%F %T')"
    rc="1"
    if [[ -s "${ACCEL_STATE_DIR}/${job_id}.exit" ]]; then
      rc="$(cat "${ACCEL_STATE_DIR}/${job_id}.exit")"
    fi
    status="ok"
    if [[ "${rc}" != "0" ]]; then
      status="failed"
      FAILED_JOB["${job_id}"]=1
    fi
    append_status "${job_id}" "${task}" "${dataset}" "${label}" "${gpus}" "${status}" "${rc}" "${started}" "${finished}"
    write_record "${job_id}" "${task}" "${dataset}" "${label}" "${gpus}" "${status}" "${rc}" "${started}" "${finished}" "${result}" "${log_path}"
    log "ACCEL_DONE ${job_id} status=${status} exit=${rc}"
    mark_gpus_free "${gpus}"
    unset "ACTIVE_JOB[${job_id}]"
    unset "PID_JOB[${pid}]" "PID_TASK[${pid}]" "PID_DATASET[${pid}]" "PID_LABEL[${pid}]" "PID_GPUS[${pid}]" "PID_RESULT[${pid}]" "PID_LOG[${pid}]" "PID_STARTED[${pid}]"
  done
}

all_queue_done() {
  local job
  for job in \
    rank_suite_k48 \
    rank_cirr_k64 \
    rank_suite_k64 \
    hard_analysis \
    loss_cirr_fwd_only \
    loss_cirr_fwd_rev_nozero \
    loss_multi_fwd_only \
    loss_multi_fwd_rev_nozero; do
    status_has_terminal_job "${job}" || return 1
  done
  return 0
}

main() {
  log "accelerator starting: gpu_pool=${ACCEL_GPUS} max_active_gpus=${MAX_ACTIVE_GPUS}"
  reserve_external_rank_suite_k48
  local base_port=$((7400 + RANDOM % 1000))
  while ! all_queue_done; do
    record_external_rank_suite_k48_if_done || true
    reap_done_jobs
    start_job "rank_cirr_k64" "rank_ablation" "cirr" "svd_k64" "1" "rank_cirr" "64" || true
    start_job "rank_suite_k64" "rank_ablation" "circo_genecis" "svd_k64" "1" "rank_suite" "64" || true
    start_job "hard_analysis" "hard_analysis" "cirr_circo_genecis" "strict_merged_wins" "1" "hard" || true
    start_job "loss_cirr_fwd_only" "transition_loss_ablation" "cirr" "fwd_only" "2" "loss" "" "$((base_port + 1))" || true
    start_job "loss_cirr_fwd_rev_nozero" "transition_loss_ablation" "cirr" "fwd_rev_nozero" "2" "loss" "" "$((base_port + 2))" || true
    start_job "loss_multi_fwd_only" "transition_loss_ablation" "multi" "fwd_only" "2" "loss" "" "$((base_port + 3))" || true
    start_job "loss_multi_fwd_rev_nozero" "transition_loss_ablation" "multi" "fwd_rev_nozero" "2" "loss" "" "$((base_port + 4))" || true
    sleep 20
  done
  reap_done_jobs
  nvidia-smi > "${RESULT_DIR}/host_gpu_snapshot_accelerator_end.txt" 2>&1 || true
  log "accelerator completed all remaining follow-up jobs"
}

main "$@"
