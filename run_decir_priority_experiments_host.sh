#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/mingyu/composed_image_retrieval}"
PY="${PY:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
SEARLE="${SEARLE:-/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt}"
RUN_ID="${RUN_ID:-decir_priority_$(date '+%Y%m%d_%H%M%S')}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/logs/${RUN_ID}}"
TMP_DIR="${TMP_DIR:-/tmp/${RUN_ID}}"
GPU_PREFERENCE="${GPU_PREFERENCE:-3,4,5,6,7,0,1,2}"
GPU_MEMORY_IDLE_MAX_MB="${GPU_MEMORY_IDLE_MAX_MB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-20}"
SVD_TOPK_RANK="${SVD_TOPK_RANK:-32}"
WORKERS="${WORKERS:-1}"
CIRR_BATCH_SIZE="${CIRR_BATCH_SIZE:-48}"
SUITE_BATCH_SIZE="${SUITE_BATCH_SIZE:-32}"
GENECIS_BATCH_SIZE="${GENECIS_BATCH_SIZE:-32}"
QUERYDIR_BATCH_SIZE="${QUERYDIR_BATCH_SIZE:-48}"
QUERYDIR_MAX_SAMPLES="${QUERYDIR_MAX_SAMPLES:-0}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

CIRR_RUN_DIR="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp"
CIRR_JOINT_CKPT="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_JointSingle_CIRR/checkpoints/epoch_0_step_1400.pt"
MULTI_RUN_DIR="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp"
MULTI_JOINT_CKPT="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_JointSingle_CIRCO_GeneCIS/checkpoints/epoch_0_step_1400.pt"

RESULTS_JSONL="${RESULT_DIR}/results.jsonl"
PROGRESS_MD="${RESULT_DIR}/PROGRESS.md"
STATUS_TSV="${RESULT_DIR}/status.tsv"
RECORD_DIR="${RESULT_DIR}/records"
LOG_DIR="${RESULT_DIR}/job_logs"

mkdir -p "${RESULT_DIR}" "${TMP_DIR}" "${RECORD_DIR}" "${LOG_DIR}"
: > "${RESULTS_JSONL}"
printf 'job_id\ttask\tdataset\tlabel\tgpu\tstatus\texit_code\tstarted_at\tfinished_at\n' > "${STATUS_TSV}"

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${RESULT_DIR}/queue.log"
  printf -- '- %s %s\n' "$(date '+%F %T')" "${msg}" >> "${PROGRESS_MD}"
}

write_header() {
  cat > "${PROGRESS_MD}" <<EOF
# DeCIR Priority Experiments

- run_id: ${RUN_ID}
- result_dir: ${RESULT_DIR}
- started_at: $(date '+%F %T')
- gpu_policy: use idle GPUs only; max 2 normally, max 6 only when all 8 GPUs are idle.
- priority_order: LRDM necessity ablation, merge-weight sweep, composed-query direction.

## Progress
EOF
}

gpu_is_idle() {
  local gpu_idx="$1"
  local pids mem_used
  pids="$(nvidia-smi -i "${gpu_idx}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
  mem_used="$(nvidia-smi -i "${gpu_idx}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
  [[ -z "${pids}" ]] && [[ "${mem_used}" =~ ^[0-9]+$ ]] && (( mem_used <= GPU_MEMORY_IDLE_MAX_MB ))
}

discover_idle_gpus() {
  local idle=()
  local pref gpu
  IFS=',' read -r -a pref <<< "${GPU_PREFERENCE}"
  for gpu in "${pref[@]}"; do
    if gpu_is_idle "${gpu}"; then
      idle+=("${gpu}")
    fi
  done
  printf '%s\n' "${idle[@]}"
}

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d '[:space:]'
}

selected_gpus_csv() {
  local -a idle=("$@")
  local total limit n
  total="$(gpu_count)"
  if [[ "${total}" != "8" ]]; then
    limit=2
  elif [[ "${#idle[@]}" -eq 8 ]]; then
    limit=6
  else
    limit=2
  fi
  n="${#idle[@]}"
  if (( n > limit )); then
    n="${limit}"
  fi
  if (( n == 0 )); then
    return 1
  fi
  local out=()
  local i
  for ((i = 0; i < n; i++)); do
    out+=("${idle[$i]}")
  done
  local IFS=,
  printf '%s\n' "${out[*]}"
}

require_host_gpu_visibility() {
  if ! nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "${RESULT_DIR}/host_gpu_probe.txt" 2>&1; then
    log "nvidia-smi is not available. Run this script in a real host tmux/session, not inside the restricted Codex sandbox."
    cat "${RESULT_DIR}/host_gpu_probe.txt" >&2
    exit 12
  fi
}

json_record() {
  local output="$1"
  local job_id="$2"
  local task="$3"
  local dataset="$4"
  local label="$5"
  local gpu="$6"
  local status="$7"
  local exit_code="$8"
  local started="$9"
  local finished="${10}"
  local result_path="${11:-}"
  local log_path="${12:-}"
  local extra_json="${13:-{}}"
  "${PY}" - "$output" "$job_id" "$task" "$dataset" "$label" "$gpu" "$status" "$exit_code" "$started" "$finished" "$result_path" "$log_path" "$extra_json" <<'PY'
import json
import sys
from pathlib import Path

output, job_id, task, dataset, label, gpu, status, exit_code, started, finished, result_path, log_path, extra = sys.argv[1:]
record = {
    "job_id": job_id,
    "task": task,
    "dataset": dataset,
    "label": label,
    "gpu": gpu,
    "status": status,
    "exit_code": int(exit_code),
    "started_at": started,
    "finished_at": finished,
}
if result_path:
    record["result_path"] = result_path
    path = Path(result_path)
    if path.exists() and path.stat().st_size > 0:
        try:
            record["result"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            record["result_parse_error"] = str(exc)
if log_path:
    record["log_path"] = log_path
try:
    record.update(json.loads(extra))
except Exception as exc:
    record["extra_parse_error"] = str(exc)
Path(output).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

append_status() {
  local job_id="$1" task="$2" dataset="$3" label="$4" gpu="$5" status="$6" exit_code="$7" started="$8" finished="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${exit_code}" "${started}" "${finished}" >> "${STATUS_TSV}"
}

parse_cirr_log_to_json() {
  local log_path="$1"
  local output_json="$2"
  "${PY}" - "$log_path" "$output_json" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path, output_json = sys.argv[1:]
feature_re = re.compile(r"Eval\s+(\w+)\s+Feature")
metric_re = re.compile(r"([A-Za-z0-9_@]+):\s*([0-9.]+)")
metrics = {}
current = None
for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
    match = feature_re.search(line)
    if match:
        current = match.group(1).lower()
        metrics.setdefault(current, {})
    if current:
        for name, value in metric_re.findall(line):
            metrics[current][name] = float(value)
Path(output_json).write_text(json.dumps({"cirr": {"metrics": metrics}}, indent=2), encoding="utf-8")
PY
}

merge_checkpoint() {
  local base_ckpt="$1"
  local geo_ckpt="$2"
  local merged_ckpt="$3"
  local mode="$4"
  local wa="$5"
  local wb="$6"
  "${PY}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${base_ckpt}" \
    --ckpt-b "${geo_ckpt}" \
    --output "${merged_ckpt}" \
    --weights "${wa}" "${wb}" \
    --density 0.9 \
    --text-only \
    --base a \
    --alpha-a 16 --rank-a 64 \
    --alpha-b 16 --rank-b 64 \
    --merge-mode "${mode}" \
    --shared-b-num-layers 12 \
    --svd-topk-rank "${SVD_TOPK_RANK}"
}

run_p1_cirr() {
  local gpu="$1" label="$2" mode="$3" wa="$4" wb="$5" result_json="$6" log_path="$7"
  local ckpt_dir="${CIRR_RUN_DIR}/checkpoints"
  local base_ckpt="${ckpt_dir}/epoch_0_step_1400.pt"
  local geo_ckpt="${ckpt_dir}/epoch_0_step_1400_geo_lora_ema.pt"
  local eval_ckpt="${base_ckpt}"
  local merged_ckpt=""
  if [[ "${mode}" != "none" ]]; then
    merged_ckpt="${TMP_DIR}/${label}_cirr.pt"
    merge_checkpoint "${base_ckpt}" "${geo_ckpt}" "${merged_ckpt}" "${mode}" "${wa}" "${wb}" >> "${log_path}" 2>&1
    eval_ckpt="${merged_ckpt}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${ROOT}/src/eval_retrieval.py" \
    --resume "${eval_ckpt}" \
    --openai-pretrained \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --middle_dim 3072 \
    --img2text-pretrained "${SEARLE}" \
    --eval-mode cirr \
    --gpu 0 \
    --batch-size "${CIRR_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --retrieval-prompt-connector and >> "${log_path}" 2>&1
  parse_cirr_log_to_json "${log_path}" "${result_json}"
  if [[ -n "${merged_ckpt}" ]]; then
    rm -f "${merged_ckpt}"
  fi
}

run_p1_suite() {
  local gpu="$1" label="$2" mode="$3" wa="$4" wb="$5" result_json="$6" log_path="$7"
  local ckpt_dir="${MULTI_RUN_DIR}/checkpoints"
  local base_ckpt="${ckpt_dir}/epoch_0_step_1400.pt"
  local geo_ckpt="${ckpt_dir}/epoch_0_step_1400_geo_lora_ema.pt"
  local eval_ckpt="${base_ckpt}"
  local merged_ckpt=""
  if [[ "${mode}" != "none" ]]; then
    merged_ckpt="${TMP_DIR}/${label}_suite.pt"
    merge_checkpoint "${base_ckpt}" "${geo_ckpt}" "${merged_ckpt}" "${mode}" "${wa}" "${wb}" >> "${log_path}" 2>&1
    eval_ckpt="${merged_ckpt}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${ROOT}/data/eval_multidataset_suite.py" \
    --resume "${eval_ckpt}" \
    --output-json "${result_json}" \
    --gpu 0 \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --img2text-pretrained "${SEARLE}" \
    --middle-dim 3072 \
    --batch-size "${SUITE_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --genecis-batch-size "${GENECIS_BATCH_SIZE}" \
    --retrieval-prompt-connector that \
    --datasets "circo,genecis" >> "${log_path}" 2>&1
  if [[ -n "${merged_ckpt}" ]]; then
    rm -f "${merged_ckpt}"
  fi
}

run_querydir_direct() {
  local gpu="$1" dataset="$2" label="$3" ckpt="$4" connector="$5" result_json="$6" log_path="$7"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${ROOT}/data/eval_composed_query_direction.py" \
    --resume "${ckpt}" \
    --output-json "${result_json}" \
    --gpu 0 \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --middle-dim 3072 \
    --img2text-pretrained "${SEARLE}" \
    --batch-size "${QUERYDIR_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --retrieval-prompt-connector "${connector}" \
    --datasets "${dataset}" \
    --max-samples "${QUERYDIR_MAX_SAMPLES}" \
    --name "querydir_${label}_${dataset}" >> "${log_path}" 2>&1
}

run_querydir_merged() {
  local gpu="$1" dataset="$2" label="$3" run_dir="$4" connector="$5" result_json="$6" log_path="$7"
  local ckpt_dir="${run_dir}/checkpoints"
  local base_ckpt="${ckpt_dir}/epoch_0_step_1400.pt"
  local geo_ckpt="${ckpt_dir}/epoch_0_step_1400_geo_lora_ema.pt"
  local merged_ckpt="${TMP_DIR}/${label}_${dataset}_querydir.pt"
  merge_checkpoint "${base_ckpt}" "${geo_ckpt}" "${merged_ckpt}" "shared_b_svd_a" "0.5" "0.5" >> "${log_path}" 2>&1
  run_querydir_direct "${gpu}" "${dataset}" "${label}" "${merged_ckpt}" "${connector}" "${result_json}" "${log_path}"
  rm -f "${merged_ckpt}"
}

run_one_job() {
  local spec="$1" gpu="$2"
  local job_id task dataset label a b c d
  IFS='|' read -r job_id task dataset label a b c d <<< "${spec}"
  local started finished log_path result_json record_json rc
  started="$(date '+%F %T')"
  log_path="${LOG_DIR}/${job_id}.log"
  result_json="${RECORD_DIR}/${job_id}_result.json"
  record_json="${RECORD_DIR}/${job_id}.json"
  log "START ${job_id} task=${task} dataset=${dataset} label=${label} gpu=${gpu}"
  set +e
  (
    set -e
    case "${task}" in
      p1_cirr)
        run_p1_cirr "${gpu}" "${label}" "${a}" "${b}" "${c}" "${result_json}" "${log_path}"
        ;;
      p1_suite)
        run_p1_suite "${gpu}" "${label}" "${a}" "${b}" "${c}" "${result_json}" "${log_path}"
        ;;
      querydir_direct)
        run_querydir_direct "${gpu}" "${dataset}" "${label}" "${a}" "${b}" "${result_json}" "${log_path}"
        ;;
      querydir_merged)
        run_querydir_merged "${gpu}" "${dataset}" "${label}" "${a}" "${b}" "${result_json}" "${log_path}"
        ;;
      *)
        echo "unknown task: ${task}" >> "${log_path}"
        false
        ;;
    esac
  )
  rc=$?
  set -e
  finished="$(date '+%F %T')"
  local status="ok"
  if [[ "${rc}" != "0" ]]; then
    status="failed"
    rm -f "${TMP_DIR}/${label}"*.pt
  fi
  json_record "${record_json}" "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${rc}" "${started}" "${finished}" "${result_json}" "${log_path}" "{}"
  {
    flock 9
    cat "${record_json}" >> "${RESULTS_JSONL}"
    append_status "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${rc}" "${started}" "${finished}"
  } 9>"${RESULT_DIR}/results.lock"
  log "DONE ${job_id} status=${status} exit=${rc}"
  return "${rc}"
}

build_jobs() {
  JOBS=()
  local labels modes weights
  labels=("retrieval_only" "avg_coeff_no_svd_w050_050" "weighted_no_svd_w075_025" "weighted_no_svd_w025_075" "lrdm_full_svd_w050_050")
  modes=("none" "shared_b_sum_a" "shared_b_sum_a" "shared_b_sum_a" "shared_b_svd_a")
  weights=("0.0|0.0" "0.5|0.5" "0.75|0.25" "0.25|0.75" "0.5|0.5")
  local i wa wb
  for i in "${!labels[@]}"; do
    IFS='|' read -r wa wb <<< "${weights[$i]}"
    JOBS+=("p1_cirr_${labels[$i]}|p1_cirr|cirr|${labels[$i]}|${modes[$i]}|${wa}|${wb}|")
    JOBS+=("p1_suite_${labels[$i]}|p1_suite|circo_genecis|${labels[$i]}|${modes[$i]}|${wa}|${wb}|")
  done
  JOBS+=("querydir_cirr_retrieval_only|querydir_direct|cirr|retrieval_only|${CIRR_RUN_DIR}/checkpoints/epoch_0_step_1400.pt|and||")
  JOBS+=("querydir_cirr_joint|querydir_direct|cirr|joint_training|${CIRR_JOINT_CKPT}|and||")
  JOBS+=("querydir_cirr_decir_merged|querydir_merged|cirr|decir_merged|${CIRR_RUN_DIR}|and||")
  JOBS+=("querydir_circo_retrieval_only|querydir_direct|circo|retrieval_only|${MULTI_RUN_DIR}/checkpoints/epoch_0_step_1400.pt|that||")
  JOBS+=("querydir_circo_joint|querydir_direct|circo|joint_training|${MULTI_JOINT_CKPT}|that||")
  JOBS+=("querydir_circo_decir_merged|querydir_merged|circo|decir_merged|${MULTI_RUN_DIR}|that||")
}

validate_inputs() {
  local missing=0
  local paths=(
    "${CIRR_RUN_DIR}/checkpoints/epoch_0_step_1400.pt"
    "${CIRR_RUN_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt"
    "${CIRR_JOINT_CKPT}"
    "${MULTI_RUN_DIR}/checkpoints/epoch_0_step_1400.pt"
    "${MULTI_RUN_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt"
    "${MULTI_JOINT_CKPT}"
    "${SEARLE}"
  )
  local p
  for p in "${paths[@]}"; do
    if [[ ! -f "${p}" ]]; then
      echo "missing input: ${p}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" == "1" ]]; then
    exit 2
  fi
}

main() {
  write_header
  require_host_gpu_visibility
  validate_inputs
  build_jobs

  mapfile -t idle_gpus < <(discover_idle_gpus)
  if [[ "${#idle_gpus[@]}" -eq 0 ]]; then
    log "no idle GPUs found; exiting without launching jobs"
    exit 3
  fi
  selected="$(selected_gpus_csv "${idle_gpus[@]}")"
  IFS=',' read -r -a GPUS <<< "${selected}"
  log "idle GPUs: ${idle_gpus[*]}"
  log "selected GPUs: ${GPUS[*]} (parallel=${#GPUS[@]})"
  log "jobs queued: ${#JOBS[@]}"
  nvidia-smi > "${RESULT_DIR}/host_gpu_snapshot_start.txt" 2>&1 || true

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "${JOBS[@]}" > "${RESULT_DIR}/dry_run_jobs.txt"
    log "dry run complete"
    exit 0
  fi

  local next=0
  declare -A pid_to_gpu=()
  declare -A pid_to_job=()
  while (( next < ${#JOBS[@]} || ${#pid_to_gpu[@]} > 0 )); do
    local gpu busy pid spec
    for gpu in "${GPUS[@]}"; do
      busy=0
      for pid in "${!pid_to_gpu[@]}"; do
        if [[ "${pid_to_gpu[$pid]}" == "${gpu}" ]]; then
          busy=1
          break
        fi
      done
      if (( busy == 1 || next >= ${#JOBS[@]} )); then
        continue
      fi
      if ! gpu_is_idle "${gpu}"; then
        continue
      fi
      spec="${JOBS[$next]}"
      ( run_one_job "${spec}" "${gpu}" ) &
      pid=$!
      pid_to_gpu["${pid}"]="${gpu}"
      pid_to_job["${pid}"]="${spec%%|*}"
      next=$((next + 1))
    done

    for pid in "${!pid_to_gpu[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" || true
        unset 'pid_to_gpu[$pid]'
        unset 'pid_to_job[$pid]'
      fi
    done
    sleep "${POLL_SECONDS}"
  done

  nvidia-smi > "${RESULT_DIR}/host_gpu_snapshot_end.txt" 2>&1 || true
  log "all jobs completed; unified results: ${RESULTS_JSONL}"
}

main "$@"
