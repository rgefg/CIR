#!/bin/bash
set -euo pipefail

export PYTHONPATH="/data2/mingyu/composed_image_retrieval:/data2/mingyu/composed_image_retrieval/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DIST_URL="${DIST_URL:-tcp://127.0.0.1:6146}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-6,7}}"
RUN_NAME="${RUN_NAME:-DistillCIR_ParallelDualLoRA_BS56_Accum8_EMA1700_QKV_StrictLoss}"
EVAL_GPU="${EVAL_GPU:-2}"

TRAIN_JSON="${TRAIN_JSON:-/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2_with_reverse.jsonl}"
REVERSE_JSON="${REVERSE_JSON:-}"
if [[ -z "${WDS_SHARDS:-}" ]]; then
  WDS_SHARDS="/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar"
fi
PIC2WORD_CKPT="${PIC2WORD_CKPT:-/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-56}"
TRAIN_ACCUM_STEPS="${TRAIN_ACCUM_STEPS:-8}"
TRAIN_WORKERS="${TRAIN_WORKERS:-2}"
TRAIN_EPOCH_STEPS="${TRAIN_EPOCH_STEPS:-1700}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
SAVE_STEP_INTERVAL="${SAVE_STEP_INTERVAL:-200}"
SAVE_STEP_START="${SAVE_STEP_START:-200}"
SAVE_STEP_END="${SAVE_STEP_END:-1700}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
CIRR_VAL_EVAL_EVERY="${CIRR_VAL_EVAL_EVERY:-${SAVE_STEP_INTERVAL}}"
WDS_SHUFFLE="${WDS_SHUFFLE:-10000}"
WDS_SHARDSHUFFLE="${WDS_SHARDSHUFFLE:-1000}"
SEED="${SEED:-3407}"
ENABLE_WDS_DETERMINISTIC="${ENABLE_WDS_DETERMINISTIC:-0}"
ENABLE_DETERMINISTIC_TRAIN="${ENABLE_DETERMINISTIC_TRAIN:-0}"

LR="${LR:-2e-5}"
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.98}"
EPS="${EPS:-1e-6}"
PRECISION="${PRECISION:-amp}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
AMP_INIT_SCALE="${AMP_INIT_SCALE:-65536}"
AMP_GROWTH_FACTOR="${AMP_GROWTH_FACTOR:-2.0}"
AMP_BACKOFF_FACTOR="${AMP_BACKOFF_FACTOR:-0.5}"
AMP_GROWTH_INTERVAL="${AMP_GROWTH_INTERVAL:-2000}"
RETRIEVAL_EMA_DECAY="${RETRIEVAL_EMA_DECAY:-0.999}"

GEO_WEIGHT="${GEO_WEIGHT:-1.0}"
GEO_SEED="${GEO_SEED:-${SEED}}"
GEO_LR="${GEO_LR:-${LR}}"
GEO_BETA1="${GEO_BETA1:-${BETA1}}"
GEO_BETA2="${GEO_BETA2:-${BETA2}}"
GEO_EPS="${GEO_EPS:-${EPS}}"
GEO_WD="${GEO_WD:-${WD}}"
GEO_WARMUP_STEPS="${GEO_WARMUP_STEPS:-${WARMUP_STEPS}}"
GEO_LORA_R="${GEO_LORA_R:-${LORA_R}}"
GEO_LORA_ALPHA="${GEO_LORA_ALPHA:-${LORA_ALPHA}}"
GEO_LORA_DROPOUT="${GEO_LORA_DROPOUT:-${LORA_DROPOUT}}"
GEO_AMP_INIT_SCALE="${GEO_AMP_INIT_SCALE:-${AMP_INIT_SCALE}}"
GEO_AMP_GROWTH_FACTOR="${GEO_AMP_GROWTH_FACTOR:-${AMP_GROWTH_FACTOR}}"
GEO_AMP_BACKOFF_FACTOR="${GEO_AMP_BACKOFF_FACTOR:-${AMP_BACKOFF_FACTOR}}"
GEO_AMP_GROWTH_INTERVAL="${GEO_AMP_GROWTH_INTERVAL:-${AMP_GROWTH_INTERVAL}}"
GEO_EMA_DECAY="${GEO_EMA_DECAY:-${RETRIEVAL_EMA_DECAY}}"
ENABLE_GEO_CONFLICT_PROJECTION="${ENABLE_GEO_CONFLICT_PROJECTION:-1}"
GEO_REVERSE_WEIGHT="${GEO_REVERSE_WEIGHT:-0.25}"
GEO_REVERSE_MARGIN="${GEO_REVERSE_MARGIN:-0.0}"
GEO_ZERO_LOSS_WEIGHT="${GEO_ZERO_LOSS_WEIGHT:-1.0}"
GEO_EMBED_NORM_EPS="${GEO_EMBED_NORM_EPS:-1e-6}"
GEO_DELTA_NORM_EPS="${GEO_DELTA_NORM_EPS:-1e-4}"
GEO_DELTA_MIN_NORM="${GEO_DELTA_MIN_NORM:-1e-3}"
ENABLE_EMA_EVAL="${ENABLE_EMA_EVAL:-1}"
ENABLE_EMA_SAVE_CHECKPOINTS="${ENABLE_EMA_SAVE_CHECKPOINTS:-1}"

MERGE_RETRIEVAL_WEIGHT="${MERGE_RETRIEVAL_WEIGHT:-0.5}"
MERGE_GEO_WEIGHT="${MERGE_GEO_WEIGHT:-0.5}"
MERGE_DENSITY="${MERGE_DENSITY:-0.9}"
ENABLE_PARALLEL_MERGE_EVAL="${ENABLE_PARALLEL_MERGE_EVAL:-0}"
WATCHER_CPU_AFFINITY="${WATCHER_CPU_AFFINITY:-48-63}"
WATCHER_NICE="${WATCHER_NICE:-15}"
WATCHER_CPU_THREADS="${WATCHER_CPU_THREADS:-1}"
WATCHER_EVAL_WORKERS="${WATCHER_EVAL_WORKERS:-1}"

LOG_DIR="/data2/mingyu/composed_image_retrieval/logs/${RUN_NAME}"
CKPT_DIR="${LOG_DIR}/checkpoints"
STANDALONE_JSONL="${LOG_DIR}/cirr_val_eval.log"
MERGED_JSONL="${LOG_DIR}/cirr_val_parallel_merged_eval.jsonl"
SUMMARY_TSV="${LOG_DIR}/cirr_val_summary.tsv"

mkdir -p "${CKPT_DIR}"

WATCHER_PID=""
cleanup() {
  if [[ -n "${WATCHER_PID}" ]] && kill -0 "${WATCHER_PID}" 2>/dev/null; then
    kill "${WATCHER_PID}" 2>/dev/null || true
    wait "${WATCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Run name: ${RUN_NAME}"
echo "Training GPUs: ${TRAIN_CUDA_DEVICES}"
echo "Eval GPU: ${EVAL_GPU}"
echo "Base retrieval JSON: ${TRAIN_JSON}"
echo "Reverse sidecar JSON: ${REVERSE_JSON}"
echo "Train batch size per GPU: ${TRAIN_BATCH_SIZE}"
echo "Accumulation steps: ${TRAIN_ACCUM_STEPS}"
echo "Train workers: ${TRAIN_WORKERS}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "WDS shuffle: samples=${WDS_SHUFFLE}, shards=${WDS_SHARDSHUFFLE}"
echo "CIRR val every: ${CIRR_VAL_EVAL_EVERY}"
echo "Seed: ${SEED}"
echo "Retrieval optim: lr=${LR}, wd=${WD}, betas=(${BETA1}, ${BETA2}), eps=${EPS}"
echo "Retrieval LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Retrieval AMP: init_scale=${AMP_INIT_SCALE}, growth=${AMP_GROWTH_FACTOR}, backoff=${AMP_BACKOFF_FACTOR}, interval=${AMP_GROWTH_INTERVAL}"
echo "Retrieval EMA decay: ${RETRIEVAL_EMA_DECAY}"
echo "Deterministic WDS: ${ENABLE_WDS_DETERMINISTIC}"
echo "Deterministic cuDNN: ${ENABLE_DETERMINISTIC_TRAIN}"
echo "Geo weight: ${GEO_WEIGHT}"
echo "Geo seed: ${GEO_SEED}"
echo "Geo optim: lr=${GEO_LR}, wd=${GEO_WD}, betas=(${GEO_BETA1}, ${GEO_BETA2}), eps=${GEO_EPS}, warmup=${GEO_WARMUP_STEPS}"
echo "Geo LoRA: r=${GEO_LORA_R}, alpha=${GEO_LORA_ALPHA}, dropout=${GEO_LORA_DROPOUT}"
echo "Geo AMP: init_scale=${GEO_AMP_INIT_SCALE}, growth=${GEO_AMP_GROWTH_FACTOR}, backoff=${GEO_AMP_BACKOFF_FACTOR}, interval=${GEO_AMP_GROWTH_INTERVAL}"
echo "Geo EMA decay: ${GEO_EMA_DECAY}"
echo "EMA eval: ${ENABLE_EMA_EVAL}"
echo "EMA save checkpoints: ${ENABLE_EMA_SAVE_CHECKPOINTS}"
echo "Geo strict loss: reverse_weight=${GEO_REVERSE_WEIGHT}, reverse_margin=${GEO_REVERSE_MARGIN}, zero_loss_weight=${GEO_ZERO_LOSS_WEIGHT}"
echo "Geo norm eps: embed=${GEO_EMBED_NORM_EPS}, delta=${GEO_DELTA_NORM_EPS}, min_delta=${GEO_DELTA_MIN_NORM}"
echo "Watcher isolation: affinity=${WATCHER_CPU_AFFINITY}, nice=${WATCHER_NICE}, cpu_threads=${WATCHER_CPU_THREADS}, eval_workers=${WATCHER_EVAL_WORKERS}"

echo "Step save interval: ${SAVE_STEP_INTERVAL}"
echo "If OOM: lower TRAIN_BATCH_SIZE in this order: 56 -> 52 -> 48 -> 44, and keep effective contrastive batch reasonably large."

if [[ "${ENABLE_PARALLEL_MERGE_EVAL}" == "1" && "${GEO_WEIGHT}" != "0" && "${GEO_WEIGHT}" != "0.0" ]]; then
  python data/watch_cirr_parallel_merge_eval.py \
    --checkpoint-dir "${CKPT_DIR}" \
    --output-jsonl "${MERGED_JSONL}" \
    --eval-gpu "${EVAL_GPU}" \
    --batch-size 56 \
    --workers "${WATCHER_EVAL_WORKERS}" \
    --retrieval-weight "${MERGE_RETRIEVAL_WEIGHT}" \
    --geo-weight "${MERGE_GEO_WEIGHT}" \
    --density "${MERGE_DENSITY}" \
    --lora-alpha 16 \
    --lora-rank 64 \
    --nice "${WATCHER_NICE}" \
    --cpu-affinity "${WATCHER_CPU_AFFINITY}" \
    --cpu-threads "${WATCHER_CPU_THREADS}" \
    --poll-interval 30 \
    --timeout 1800 \
    --stop-on-final &
  WATCHER_PID=$!
fi

EXTRA_ARGS=()
if [[ "${ENABLE_GEO_CONFLICT_PROJECTION}" == "1" ]]; then
  EXTRA_ARGS+=(--geo-conflict-projection)
fi
if [[ -n "${REVERSE_JSON}" ]]; then
  EXTRA_ARGS+=(--cc3m-cir-reverse-jsonl "${REVERSE_JSON}")
fi
if [[ "${ENABLE_WDS_DETERMINISTIC}" == "1" ]]; then
  EXTRA_ARGS+=(--wds-deterministic)
fi
if [[ "${ENABLE_DETERMINISTIC_TRAIN}" == "1" ]]; then
  EXTRA_ARGS+=(--deterministic-train)
fi
if [[ "${ENABLE_EMA_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=(--ema-eval)
fi
if [[ "${ENABLE_EMA_SAVE_CHECKPOINTS}" == "1" ]]; then
  EXTRA_ARGS+=(--ema-save-checkpoints)
fi

CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_DEVICES}" python -u src/main.py \
  --name "${RUN_NAME}" \
  --dataset-type cc3m_cir_wds \
  --cc3m-cir-jsonl "${TRAIN_JSON}" \
  --train-data "dummy" \
  --wds-shards "${WDS_SHARDS}" \
  --wds-epoch-steps "${TRAIN_EPOCH_STEPS}" \
  --wds-shuffle "${WDS_SHUFFLE}" \
  --wds-shardshuffle "${WDS_SHARDSHUFFLE}" \
  --model ViT-L/14 \
  --pic2word-pretrained "${PIC2WORD_CKPT}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --accum-steps "${TRAIN_ACCUM_STEPS}" \
  --epochs 1 \
  --seed "${SEED}" \
  --lr "${LR}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --eps "${EPS}" \
  --wd "${WD}" \
  --warmup "${WARMUP_STEPS}" \
  --precision "${PRECISION}" \
  --amp-init-scale "${AMP_INIT_SCALE}" \
  --amp-growth-factor "${AMP_GROWTH_FACTOR}" \
  --amp-backoff-factor "${AMP_BACKOFF_FACTOR}" \
  --amp-growth-interval "${AMP_GROWTH_INTERVAL}" \
  --retrieval-ema-decay "${RETRIEVAL_EMA_DECAY}" \
  --workers "${TRAIN_WORKERS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --reset-logit-scale \
  --logit-scale-clamp-min 9.0 \
  --logit-scale-clamp-max 36.6 \
  --logit-scale-freeze-percent 0.3 \
  --save-frequency 1 \
  --save-step-start "${SAVE_STEP_START}" \
  --save-step-end "${SAVE_STEP_END}" \
  --save-step-interval "${SAVE_STEP_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --cirr-val-eval-every "${CIRR_VAL_EVAL_EVERY}" \
  --geo-weight "${GEO_WEIGHT}" \
  --geo-seed "${GEO_SEED}" \
  --geo-lr "${GEO_LR}" \
  --geo-beta1 "${GEO_BETA1}" \
  --geo-beta2 "${GEO_BETA2}" \
  --geo-eps "${GEO_EPS}" \
  --geo-wd "${GEO_WD}" \
  --geo-warmup "${GEO_WARMUP_STEPS}" \
  --geo-lora-r "${GEO_LORA_R}" \
  --geo-lora-alpha "${GEO_LORA_ALPHA}" \
  --geo-lora-dropout "${GEO_LORA_DROPOUT}" \
  --geo-amp-init-scale "${GEO_AMP_INIT_SCALE}" \
  --geo-amp-growth-factor "${GEO_AMP_GROWTH_FACTOR}" \
  --geo-amp-backoff-factor "${GEO_AMP_BACKOFF_FACTOR}" \
  --geo-amp-growth-interval "${GEO_AMP_GROWTH_INTERVAL}" \
  --geo-ema-decay "${GEO_EMA_DECAY}" \
  --geo-reverse-weight "${GEO_REVERSE_WEIGHT}" \
  --geo-reverse-margin "${GEO_REVERSE_MARGIN}" \
  --geo-zero-loss-weight "${GEO_ZERO_LOSS_WEIGHT}" \
  --geo-embed-norm-eps "${GEO_EMBED_NORM_EPS}" \
  --geo-delta-norm-eps "${GEO_DELTA_NORM_EPS}" \
  --geo-delta-min-norm "${GEO_DELTA_MIN_NORM}" \
  --dist-url "${DIST_URL}" \
  "${EXTRA_ARGS[@]}"

if [[ -n "${WATCHER_PID}" ]]; then
  wait "${WATCHER_PID}"
  WATCHER_PID=""
fi

if [[ -f "${STANDALONE_JSONL}" && -f "${MERGED_JSONL}" ]]; then
  python data/build_cirr_eval_table.py \
    --standalone-log "${STANDALONE_JSONL}" \
    --merged-log "${MERGED_JSONL}" \
    --output "${SUMMARY_TSV}"
  echo "Saved CIRR summary table to ${SUMMARY_TSV}"
fi
