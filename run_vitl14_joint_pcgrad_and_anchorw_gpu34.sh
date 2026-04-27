#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
PYTHON_BIN="${PYTHON_BIN:-/data2/mingyu/miniconda3/envs/torch/bin/python}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-3,4}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${ROOT}/logs/vitl14_joint_pcgrad_anchorw_gpu34_${STAMP}.log"
ANCHOR_RESULT_DIR="${ROOT}/docs/experiments/anchor_w_sensitivity_cirr_${STAMP}/results"
mkdir -p "$(dirname "${MASTER_LOG}")" "${ANCHOR_RESULT_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

gpu_snapshot() {
  log "Physical GPU snapshot:"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "${MASTER_LOG}"
}

parse_cirr_eval() {
  local run_dir="$1"
  local eval_log="$2"
  local out_json="$3"
  "${PYTHON_BIN}" - "${run_dir}" "${eval_log}" "${out_json}" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
log_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])
text = log_path.read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"Eval composed FeatureR@1: ([0-9.]+), R@5: ([0-9.]+), R@10: ([0-9.]+), R@50: ([0-9.]+), R@100: ([0-9.]+), R_subset@1: ([0-9.]+), R_subset@2: ([0-9.]+), R_subset@3: ([0-9.]+)",
    text,
)
payload = {
    "run_dir": str(run_dir),
    "eval_log": str(log_path),
}
if match:
    vals = [float(x) for x in match.groups()]
    payload.update({
        "status": "ok",
        "metrics": {
            "composed": {
                "FeatureR@1": vals[0],
                "R@5": vals[1],
                "R@10": vals[2],
                "R@50": vals[3],
                "R@100": vals[4],
                "R_subset@1": vals[5],
                "R_subset@2": vals[6],
                "R_subset@3": vals[7],
            }
        },
    })
else:
    payload.update({"status": "parse_failed", "tail": text[-4000:]})
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(out_path)
PY
}

eval_joint_cirr() {
  local run_dir="$1"
  local ckpt="${run_dir}/checkpoints/epoch_0_step_1400.pt"
  local eval_log="${run_dir}/joint_pcgrad_step1400_cirr_eval.log"
  local out_json="${run_dir}/joint_pcgrad_step1400_cirr_eval.json"
  if [[ ! -f "${ckpt}" ]]; then
    log "Missing CIRR checkpoint: ${ckpt}"
    return 1
  fi
  log "Evaluating joint-single PCGrad CIRR on physical GPU 3: ${ckpt}"
  "${PYTHON_BIN}" "${ROOT}/src/eval_retrieval.py" \
    --resume "${ckpt}" \
    --eval-mode cirr \
    --gpu 3 \
    --batch-size 48 \
    --workers 2 \
    --model ViT-L/14 \
    --img2text-arch phi \
    --middle_dim 3072 \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
    --retrieval-prompt-connector and \
    > "${eval_log}" 2>&1
  parse_cirr_eval "${run_dir}" "${eval_log}" "${out_json}" | tee -a "${MASTER_LOG}"
}

eval_joint_multi() {
  local run_dir="$1"
  local ckpt="${run_dir}/checkpoints/epoch_0_step_1400.pt"
  local out_json="${run_dir}/joint_pcgrad_step1400_circo_genecis_eval.json"
  local eval_log="${run_dir}/joint_pcgrad_step1400_circo_genecis_eval.log"
  if [[ ! -f "${ckpt}" ]]; then
    log "Missing CIRCO/GeneCIS checkpoint: ${ckpt}"
    return 1
  fi
  log "Evaluating joint-single PCGrad CIRCO+GeneCIS on physical GPU 4: ${ckpt}"
  "${PYTHON_BIN}" "${ROOT}/data/eval_multidataset_suite.py" \
    --resume "${ckpt}" \
    --output-json "${out_json}" \
    --gpu 4 \
    --batch-size 32 \
    --workers 2 \
    --genecis-batch-size 32 \
    --model ViT-L/14 \
    --img2text-arch phi \
    --middle-dim 3072 \
    --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
    --retrieval-prompt-connector that \
    --datasets circo,genecis \
    --name joint_pcgrad_step1400 \
    > "${eval_log}" 2>&1
}

run_joint_cirr() {
  local run_name="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_JointSinglePCGrad_CIRR"
  log "Start joint-single+PCGrad CIRR training on physical GPUs ${PHYSICAL_GPUS}"
  rm -rf "${ROOT}/logs/${run_name}"
  MODEL_NAME="ViT-L/14" \
  OPENAI_PRETRAINED="1" \
  PIC2WORD_CKPT="" \
  IMG2TEXT_ARCH="phi" \
  IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt" \
  MIDDLE_DIM="3072" \
  RETRIEVAL_PROMPT_CONNECTOR="and" \
  TRAIN_CUDA_DEVICES="${PHYSICAL_GPUS}" \
  DIST_URL="tcp://127.0.0.1:6434" \
  RUN_NAME="${run_name}" \
  TRAIN_BATCH_SIZE="56" \
  TRAIN_ACCUM_STEPS="8" \
  LR="2e-5" \
  GEO_LR="2e-5" \
  INSTRUCTION_DROPOUT_PROB="0.0" \
  TRAIN_EPOCH_STEPS="1400" \
  SAVE_STEP_START="1400" \
  SAVE_STEP_END="1400" \
  SAVE_STEP_INTERVAL="1400" \
  SHARED_B_LORA="0" \
  SHARED_B_RETRIEVAL_ONLY_UPDATE="0" \
  JOINT_SINGLE_BRANCH="1" \
  ENABLE_GEO_CONFLICT_PROJECTION="1" \
  GEO_REVERSE_WEIGHT="0.0" \
  GEO_ZERO_LOSS_WEIGHT="1.0" \
  GEO_SRC_ANCHOR_MODE="blend" \
  GEO_SRC_IMAGE_WEIGHT="0.25" \
  GEO_SRC_ANCHOR_DETACH="0" \
  RUN_POSTHOC_MERGED_EVAL="0" \
  RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
  RUN_POSTHOC_CIRR_EVAL="0" \
  MULTIDATASET_EVAL_EVERY="0" \
  CIRR_VAL_EVAL_EVERY="0" \
  bash "${ROOT}/train_with_dropout.sh" 2>&1 | tee -a "${MASTER_LOG}"
  eval_joint_cirr "${ROOT}/logs/${run_name}"
}

run_joint_circo_genecis() {
  local run_name="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_JointSinglePCGrad_CIRCO_GeneCIS"
  log "Start joint-single+PCGrad CIRCO/GeneCIS training on physical GPUs ${PHYSICAL_GPUS}"
  rm -rf "${ROOT}/logs/${run_name}"
  MODEL_NAME="ViT-L/14" \
  OPENAI_PRETRAINED="1" \
  PIC2WORD_CKPT="" \
  IMG2TEXT_ARCH="phi" \
  IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt" \
  MIDDLE_DIM="3072" \
  RETRIEVAL_PROMPT_CONNECTOR="that" \
  TRAIN_CUDA_DEVICES="${PHYSICAL_GPUS}" \
  DIST_URL="tcp://127.0.0.1:6436" \
  RUN_NAME="${run_name}" \
  TRAIN_BATCH_SIZE="56" \
  TRAIN_ACCUM_STEPS="8" \
  LR="2e-5" \
  GEO_LR="2e-5" \
  INSTRUCTION_DROPOUT_PROB="0.5" \
  TRAIN_EPOCH_STEPS="1400" \
  SAVE_STEP_START="1400" \
  SAVE_STEP_END="1400" \
  SAVE_STEP_INTERVAL="1400" \
  SHARED_B_LORA="0" \
  SHARED_B_RETRIEVAL_ONLY_UPDATE="0" \
  JOINT_SINGLE_BRANCH="1" \
  ENABLE_GEO_CONFLICT_PROJECTION="1" \
  GEO_REVERSE_WEIGHT="0.0" \
  GEO_ZERO_LOSS_WEIGHT="1.0" \
  GEO_SRC_ANCHOR_MODE="blend" \
  GEO_SRC_IMAGE_WEIGHT="0.25" \
  GEO_SRC_ANCHOR_DETACH="0" \
  RUN_POSTHOC_MERGED_EVAL="0" \
  RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
  RUN_POSTHOC_CIRR_EVAL="0" \
  MULTIDATASET_EVAL_EVERY="0" \
  CIRR_VAL_EVAL_EVERY="0" \
  bash "${ROOT}/train_with_dropout.sh" 2>&1 | tee -a "${MASTER_LOG}"
  eval_joint_multi "${ROOT}/logs/${run_name}"
}

run_anchor_w_point() {
  local w="$1"
  local tag="${w//./p}"
  local port
  case "${tag}" in
    0) port="6440" ;;
    0p25) port="6441" ;;
    0p5) port="6442" ;;
    0p75) port="6443" ;;
    1) port="6444" ;;
    *) port="6449" ;;
  esac
  local run_name="DistillCIR_ViTL14_SharedB12_BlendW${tag}_CIRR1000_Sensitivity"
  local result_json="${ANCHOR_RESULT_DIR}/blend_w_${tag}.json"
  log "Start source-anchor w=${w} sensitivity run on physical GPUs ${PHYSICAL_GPUS}"
  RUN_NAME="${run_name}" \
  TRAIN_CUDA_DEVICES="${PHYSICAL_GPUS}" \
  DIST_URL="tcp://127.0.0.1:${port}" \
  RESULT_JSON="${result_json}" \
  RETRIEVAL_PROMPT_CONNECTOR="and" \
  GEO_SRC_ANCHOR_MODE="blend" \
  GEO_SRC_IMAGE_WEIGHT="${w}" \
  GEO_SRC_ANCHOR_DETACH="0" \
  TRAIN_STEPS="1000" \
  EVAL_STEP="1000" \
  bash "${ROOT}/run_vitl14_sharedb12_cirr1000_geoexp.sh" 2>&1 | tee -a "${MASTER_LOG}"
}

summarize_anchor_w() {
  local summary_json="${ANCHOR_RESULT_DIR}/anchor_w_sensitivity_summary.json"
  "${PYTHON_BIN}" - "${ANCHOR_RESULT_DIR}" "${summary_json}" <<'PY'
import json
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = []
for path in sorted(result_dir.glob("blend_w_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {"file": str(path), "status": data.get("status")}
    w = path.stem.replace("blend_w_", "").replace("p", ".")
    row["w"] = float(w)
    metrics = data.get("metrics", {}).get("composed", {})
    for key in ["R_subset@1", "R_subset@2", "R_subset@3", "R@1", "R@5", "R@10", "R@50"]:
        if key in metrics:
            row[key] = metrics[key]
    rows.append(row)
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(summary_path)
PY
}

log "Experiment dispatcher started. Physical GPUs requested: ${PHYSICAL_GPUS}"
gpu_snapshot

run_joint_cirr
gpu_snapshot

run_joint_circo_genecis
gpu_snapshot

for w in 0 0.25 0.5 0.75 1; do
  run_anchor_w_point "${w}"
  gpu_snapshot
done

summarize_anchor_w | tee -a "${MASTER_LOG}"
log "All requested jobs finished."
