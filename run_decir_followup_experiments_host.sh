#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/mingyu/composed_image_retrieval}"
PY="${PY:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
SEARLE="${SEARLE:-/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt}"
RUN_ID="${RUN_ID:-decir_followup_$(date '+%Y%m%d_%H%M%S')}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/logs/${RUN_ID}}"
TMP_DIR="${TMP_DIR:-/tmp/${RUN_ID}}"
GPU_PREFERENCE="${GPU_PREFERENCE:-3,4,5,6,7,1,2,0}"
GPU_MEMORY_IDLE_MAX_MB="${GPU_MEMORY_IDLE_MAX_MB:-1024}"
WORKERS="${WORKERS:-1}"
CIRR_BATCH_SIZE="${CIRR_BATCH_SIZE:-48}"
SUITE_BATCH_SIZE="${SUITE_BATCH_SIZE:-32}"
GENECIS_BATCH_SIZE="${GENECIS_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-56}"
TRAIN_ACCUM_STEPS="${TRAIN_ACCUM_STEPS:-8}"
TRAIN_EPOCH_STEPS="${TRAIN_EPOCH_STEPS:-1400}"
SVD_RANKS="${SVD_RANKS:-16,32,48,64}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash run_decir_followup_experiments_host.sh

Environment overrides:
  RUN_ID, RESULT_DIR, GPU_PREFERENCE, SVD_RANKS, DRY_RUN=1

Runs:
  1. SVD rank ablation.
  2. Hard distractor case mining.
  3. Transition-loss component ablation training/eval.
EOF
  exit 0
fi

export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

CIRR_FULL_DIR="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp"
CIRR_JOINT_CKPT="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_JointSingle_CIRR/checkpoints/epoch_0_step_1400.pt"
MULTI_FULL_DIR="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp"
MULTI_JOINT_CKPT="${ROOT}/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_JointSingle_CIRCO_GeneCIS/checkpoints/epoch_0_step_1400.pt"

mkdir -p "${RESULT_DIR}/records" "${RESULT_DIR}/job_logs" "${TMP_DIR}"
RESULTS_JSONL="${RESULT_DIR}/results.jsonl"
STATUS_TSV="${RESULT_DIR}/status.tsv"
PROGRESS_MD="${RESULT_DIR}/PROGRESS.md"
if [[ "${SOURCE_ONLY:-0}" != "1" ]]; then
  : > "${RESULTS_JSONL}"
  printf 'job_id\ttask\tdataset\tlabel\tgpu\tstatus\texit_code\tstarted_at\tfinished_at\n' > "${STATUS_TSV}"
else
  touch "${RESULTS_JSONL}"
  if [[ ! -f "${STATUS_TSV}" ]]; then
    printf 'job_id\ttask\tdataset\tlabel\tgpu\tstatus\texit_code\tstarted_at\tfinished_at\n' > "${STATUS_TSV}"
  fi
fi

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${RESULT_DIR}/queue.log"
  printf -- '- %s %s\n' "$(date '+%F %T')" "${msg}" >> "${PROGRESS_MD}"
}

write_header() {
  cat > "${PROGRESS_MD}" <<EOF
# DeCIR Follow-up Experiments

- run_id: ${RUN_ID}
- result_dir: ${RESULT_DIR}
- started_at: $(date '+%F %T')
- gpu_policy: use idle GPUs only; max 2 normally, max 6 only when all 8 GPUs are idle.
- queued_work: SVD rank ablation, hard distractor analysis, transition-loss component ablation.

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
  local pref gpu
  IFS=',' read -r -a pref <<< "${GPU_PREFERENCE}"
  for gpu in "${pref[@]}"; do
    if gpu_is_idle "${gpu}"; then
      printf '%s\n' "${gpu}"
    fi
  done
}

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d '[:space:]'
}

select_gpus() {
  local -a idle=("$@")
  local total limit n i out=()
  total="$(gpu_count)"
  if [[ "${total}" == "8" && "${#idle[@]}" -eq 8 ]]; then
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
  for ((i = 0; i < n; i++)); do
    out+=("${idle[$i]}")
  done
  local IFS=,
  printf '%s\n' "${out[*]}"
}

append_status() {
  local job_id="$1" task="$2" dataset="$3" label="$4" gpu="$5" status="$6" exit_code="$7" started="$8" finished="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${exit_code}" "${started}" "${finished}" >> "${STATUS_TSV}"
}

write_record() {
  local job_id="$1" task="$2" dataset="$3" label="$4" gpu="$5" status="$6" exit_code="$7" started="$8" finished="$9" result_path="${10}" log_path="${11}"
  "${PY}" - "${RESULTS_JSONL}" "${RESULT_DIR}/records/${job_id}.json" "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${exit_code}" "${started}" "${finished}" "${result_path}" "${log_path}" <<'PY'
import json
import sys
from pathlib import Path

results_jsonl, record_json, job_id, task, dataset, label, gpu, status, exit_code, started, finished, result_path, log_path = sys.argv[1:]
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
    "result_path": result_path,
    "log_path": log_path,
}
if result_path:
    p = Path(result_path)
    if p.exists() and p.stat().st_size > 0:
        try:
            record["result"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            record["result_parse_error"] = str(exc)
Path(record_json).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with open(results_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

run_job() {
  local job_id="$1" task="$2" dataset="$3" label="$4" gpu="$5"
  shift 5
  local result_path="$1" log_path="$2"
  shift 2
  local started finished rc status
  started="$(date '+%F %T')"
  log "START ${job_id} task=${task} dataset=${dataset} label=${label} gpu=${gpu}"
  set +e
  "$@" > "${log_path}" 2>&1
  rc=$?
  set -e
  finished="$(date '+%F %T')"
  status="ok"
  if [[ "${rc}" != "0" ]]; then
    status="failed"
  fi
  append_status "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${rc}" "${started}" "${finished}"
  write_record "${job_id}" "${task}" "${dataset}" "${label}" "${gpu}" "${status}" "${rc}" "${started}" "${finished}" "${result_path}" "${log_path}"
  log "DONE ${job_id} status=${status} exit=${rc}"
  return "${rc}"
}

parse_cirr_log_to_json() {
  local log_path="$1" output_json="$2"
  "${PY}" - "${log_path}" "${output_json}" <<'PY'
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
    m = feature_re.search(line)
    if m:
        current = m.group(1).lower()
        metrics.setdefault(current, {})
    if current:
        for name, value in metric_re.findall(line):
            metrics[current][name] = float(value)
Path(output_json).write_text(json.dumps({"cirr": {"metrics": metrics}}, indent=2), encoding="utf-8")
PY
}

merge_checkpoint() {
  local base_ckpt="$1" geo_ckpt="$2" out_ckpt="$3" svd_rank="$4"
  "${PY}" "${ROOT}/data/merge_lora_ties.py" \
    --ckpt-a "${base_ckpt}" \
    --ckpt-b "${geo_ckpt}" \
    --output "${out_ckpt}" \
    --weights 0.5 0.5 \
    --density 0.9 \
    --text-only \
    --base a \
    --alpha-a 16 --rank-a 64 \
    --alpha-b 16 --rank-b 64 \
    --merge-mode shared_b_svd_a \
    --shared-b-num-layers 12 \
    --svd-topk-rank "${svd_rank}"
}

eval_cirr_checkpoint() {
  local gpu="$1" ckpt="$2" result_json="$3" connector="$4" log_path="$5"
  "${PY}" "${ROOT}/src/eval_retrieval.py" \
    --resume "${ckpt}" \
    --openai-pretrained \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --middle_dim 3072 \
    --img2text-pretrained "${SEARLE}" \
    --eval-mode cirr \
    --gpu "${gpu}" \
    --batch-size "${CIRR_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --retrieval-prompt-connector "${connector}" >> "${log_path}" 2>&1
  parse_cirr_log_to_json "${log_path}" "${result_json}"
}

eval_suite_checkpoint() {
  local gpu="$1" ckpt="$2" result_json="$3" connector="$4" log_path="$5"
  "${PY}" "${ROOT}/data/eval_multidataset_suite.py" \
    --resume "${ckpt}" \
    --output-json "${result_json}" \
    --gpu "${gpu}" \
    --batch-size "${SUITE_BATCH_SIZE}" \
    --genecis-batch-size "${GENECIS_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --model "ViT-L/14" \
    --img2text-arch phi \
    --middle-dim 3072 \
    --img2text-pretrained "${SEARLE}" \
    --retrieval-prompt-connector "${connector}" \
    --datasets circo,genecis >> "${log_path}" 2>&1
}

rank_cirr_job() {
  local gpu="$1" rank="$2" result_json="$3" log_path="$4"
  local merged="${TMP_DIR}/rank${rank}_cirr.pt"
  merge_checkpoint \
    "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
    "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
    "${merged}" \
    "${rank}" >> "${log_path}" 2>&1
  eval_cirr_checkpoint "${gpu}" "${merged}" "${result_json}" and "${log_path}"
  rm -f "${merged}"
}

rank_suite_job() {
  local gpu="$1" rank="$2" result_json="$3" log_path="$4"
  local merged="${TMP_DIR}/rank${rank}_suite.pt"
  merge_checkpoint \
    "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
    "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
    "${merged}" \
    "${rank}" >> "${log_path}" 2>&1
  eval_suite_checkpoint "${gpu}" "${merged}" "${result_json}" that "${log_path}"
  rm -f "${merged}"
}

run_rank_ablation() {
  local -a gpulist=("$@")
  local rank gpu_cirr gpu_suite result log_path
  IFS=',' read -r -a ranks <<< "${SVD_RANKS}"
  gpu_cirr="${gpulist[0]}"
  gpu_suite="${gpulist[1]:-${gpulist[0]}}"
  for rank in "${ranks[@]}"; do
    result="${RESULT_DIR}/records/rank_cirr_k${rank}_result.json"
    log_path="${RESULT_DIR}/job_logs/rank_cirr_k${rank}.log"
    run_job "rank_cirr_k${rank}" "rank_ablation" "cirr" "svd_k${rank}" "${gpu_cirr}" "${result}" "${log_path}" \
      rank_cirr_job "${gpu_cirr}" "${rank}" "${result}" "${log_path}"

    result="${RESULT_DIR}/records/rank_suite_k${rank}_result.json"
    log_path="${RESULT_DIR}/job_logs/rank_suite_k${rank}.log"
    run_job "rank_suite_k${rank}" "rank_ablation" "circo_genecis" "svd_k${rank}" "${gpu_suite}" "${result}" "${log_path}" \
      rank_suite_job "${gpu_suite}" "${rank}" "${result}" "${log_path}"
  done
}

run_hard_analysis() {
  local gpu="$1"
  local cirr_merged="${TMP_DIR}/hard_cirr_lrdm.pt"
  local multi_merged="${TMP_DIR}/hard_multi_lrdm.pt"
  local result_json="${RESULT_DIR}/hard_analysis/hard_cases_all.json"
  local log_path="${RESULT_DIR}/job_logs/hard_analysis.log"
  mkdir -p "${RESULT_DIR}/hard_analysis"
  (
    merge_checkpoint \
      "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
      "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
      "${cirr_merged}" \
      32
    merge_checkpoint \
      "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
      "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
      "${multi_merged}" \
      32
    "${PY}" "${ROOT}/data/find_hard_distractor_cases.py" \
      --output-dir "${RESULT_DIR}/hard_analysis" \
      --gpu "${gpu}" \
      --batch-size "${CIRR_BATCH_SIZE}" \
      --genecis-batch-size "${GENECIS_BATCH_SIZE}" \
      --workers "${WORKERS}" \
      --max-cases 30 \
      --img2text-pretrained "${SEARLE}" \
      --cirr-retrieval "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
      --cirr-joint "${CIRR_JOINT_CKPT}" \
      --cirr-merged "${cirr_merged}" \
      --multi-retrieval "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
      --multi-joint "${MULTI_JOINT_CKPT}" \
      --multi-merged "${multi_merged}"
    rm -f "${cirr_merged}" "${multi_merged}"
  ) > "${log_path}" 2>&1
  cp "${result_json}" "${RESULT_DIR}/records/hard_analysis_result.json"
}

train_loss_ablation_job() {
  local dataset="$1" variant="$2" train_gpus="$3" dist_port="$4" result_json="$5" log_path="$6"
  local connector dropout run_name log_dir ckpt_dir use_rev zero_weight
  if [[ "${dataset}" == "cirr" ]]; then
    connector="and"
    dropout="0.0"
    run_name="DistillCIR_Followup_ViTL14_SharedB12_${variant}_CIRR"
  else
    connector="that"
    dropout="0.5"
    run_name="DistillCIR_Followup_ViTL14_SharedB12_${variant}_CIRCO_GeneCIS"
  fi
  if [[ "${variant}" == "fwd_only" ]]; then
    use_rev="0"
    zero_weight="0.0"
  elif [[ "${variant}" == "fwd_rev_nozero" ]]; then
    use_rev="1"
    zero_weight="0.0"
  else
    echo "unknown variant: ${variant}" >&2
    return 2
  fi
  log_dir="${ROOT}/logs/${run_name}"
  ckpt_dir="${log_dir}/checkpoints"
  MODEL_NAME="ViT-L/14" \
  OPENAI_PRETRAINED="1" \
  PIC2WORD_CKPT="" \
  IMG2TEXT_ARCH="phi" \
  IMG2TEXT_PRETRAINED="${SEARLE}" \
  MIDDLE_DIM="3072" \
  RETRIEVAL_PROMPT_CONNECTOR="${connector}" \
  TRAIN_CUDA_DEVICES="${train_gpus}" \
  DIST_URL="tcp://127.0.0.1:${dist_port}" \
  RUN_NAME="${run_name}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  TRAIN_ACCUM_STEPS="${TRAIN_ACCUM_STEPS}" \
  LR="2e-5" \
  GEO_LR="2e-5" \
  INSTRUCTION_DROPOUT_PROB="${dropout}" \
  TRAIN_EPOCH_STEPS="${TRAIN_EPOCH_STEPS}" \
  SAVE_STEP_START="${TRAIN_EPOCH_STEPS}" \
  SAVE_STEP_END="${TRAIN_EPOCH_STEPS}" \
  SAVE_STEP_INTERVAL="200" \
  SHARED_B_LORA="1" \
  SHARED_B_NUM_LAYERS="12" \
  SHARED_B_RETRIEVAL_ONLY_UPDATE="1" \
  GEO_REVERSE_WEIGHT="0.0" \
  GEO_USE_REVERSE_ALIGNMENT="${use_rev}" \
  GEO_ZERO_LOSS_WEIGHT="${zero_weight}" \
  RUN_POSTHOC_MERGED_EVAL="0" \
  RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
  RUN_POSTHOC_CIRR_EVAL="0" \
  MULTIDATASET_EVAL_EVERY="0" \
  CIRR_VAL_EVAL_EVERY="0" \
  bash "${ROOT}/train_with_dropout.sh" >> "${log_path}" 2>&1

  local merged="${TMP_DIR}/${run_name}_merged.pt"
  merge_checkpoint \
    "${ckpt_dir}/epoch_0_step_${TRAIN_EPOCH_STEPS}.pt" \
    "${ckpt_dir}/epoch_0_step_${TRAIN_EPOCH_STEPS}_geo_lora_ema.pt" \
    "${merged}" \
    32 >> "${log_path}" 2>&1
  if [[ "${dataset}" == "cirr" ]]; then
    eval_cirr_checkpoint "${train_gpus%%,*}" "${merged}" "${result_json}" "${connector}" "${log_path}"
  else
    eval_suite_checkpoint "${train_gpus%%,*}" "${merged}" "${result_json}" "${connector}" "${log_path}"
  fi
  rm -f "${merged}"
}

run_transition_loss_ablation() {
  local -a gpulist=("$@")
  if (( ${#gpulist[@]} < 2 )); then
    log "transition loss training needs 2 idle GPUs; only ${#gpulist[@]} selected"
    return 4
  fi
  local train_gpus="${gpulist[0]},${gpulist[1]}"
  local base_port=$((6400 + RANDOM % 1000))
  local dataset variant job_id result log_path port label
  local idx=0
  for dataset in cirr multi; do
    for variant in fwd_only fwd_rev_nozero; do
      idx=$((idx + 1))
      port=$((base_port + idx))
      label="${variant}"
      job_id="loss_${dataset}_${variant}"
      result="${RESULT_DIR}/records/${job_id}_result.json"
      log_path="${RESULT_DIR}/job_logs/${job_id}.log"
      run_job "${job_id}" "transition_loss_ablation" "${dataset}" "${label}" "${train_gpus}" "${result}" "${log_path}" \
        train_loss_ablation_job "${dataset}" "${variant}" "${train_gpus}" "${port}" "${result}" "${log_path}"
    done
  done
}

validate_inputs() {
  local missing=0 p
  for p in \
    "${SEARLE}" \
    "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
    "${CIRR_FULL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
    "${CIRR_JOINT_CKPT}" \
    "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400.pt" \
    "${MULTI_FULL_DIR}/checkpoints/epoch_0_step_1400_geo_lora_ema.pt" \
    "${MULTI_JOINT_CKPT}"; do
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
  validate_inputs
  if ! nvidia-smi > "${RESULT_DIR}/host_gpu_snapshot_start.txt" 2>&1; then
    log "nvidia-smi unavailable; abort"
    exit 12
  fi
  mapfile -t idle_gpus < <(discover_idle_gpus)
  if [[ "${#idle_gpus[@]}" -eq 0 ]]; then
    log "no idle GPUs found; abort"
    exit 3
  fi
  selected="$(select_gpus "${idle_gpus[@]}")"
  IFS=',' read -r -a GPUS <<< "${selected}"
  log "idle GPUs: ${idle_gpus[*]}"
  log "selected GPUs: ${GPUS[*]} (parallel_limit=${#GPUS[@]})"
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      echo "selected=${selected}"
      echo "rank_ablation=${SVD_RANKS}"
      echo "hard_analysis=1"
      echo "transition_loss_jobs=loss_cirr_fwd_only,loss_cirr_fwd_rev_nozero,loss_multi_fwd_only,loss_multi_fwd_rev_nozero"
    } > "${RESULT_DIR}/dry_run_plan.txt"
    log "dry run complete"
    exit 0
  fi
  log "stage 1/3 rank ablation: ranks=${SVD_RANKS}"
  run_rank_ablation "${GPUS[@]}"
  log "stage 2/3 hard distractor analysis"
  run_job "hard_analysis" "hard_analysis" "cirr_circo_genecis" "strict_merged_wins" "${GPUS[0]}" \
    "${RESULT_DIR}/records/hard_analysis_result.json" "${RESULT_DIR}/job_logs/hard_analysis.wrapper.log" run_hard_analysis "${GPUS[0]}"
  log "stage 3/3 transition loss ablation training/eval"
  run_transition_loss_ablation "${GPUS[@]}"
  nvidia-smi > "${RESULT_DIR}/host_gpu_snapshot_end.txt" 2>&1 || true
  log "all follow-up jobs completed"
}

if [[ "${SOURCE_ONLY:-0}" != "1" ]]; then
  main "$@"
fi
