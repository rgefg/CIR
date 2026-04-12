#!/bin/bash
set -euo pipefail

ROOT="/data2/mingyu/composed_image_retrieval"
RUN_NAME="DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_GeoOnly_Drop0p2_200_All3Eval"

MODEL_NAME="ViT-L/14" \
OPENAI_PRETRAINED="1" \
PIC2WORD_CKPT="" \
IMG2TEXT_ARCH="phi" \
IMG2TEXT_PRETRAINED="/data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt" \
MIDDLE_DIM="3072" \
RETRIEVAL_PROMPT_CONNECTOR="that" \
TRAIN_CUDA_DEVICES="4,5" \
DIST_URL="tcp://127.0.0.1:6362" \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="56" \
TRAIN_ACCUM_STEPS="8" \
LR="2e-5" \
GEO_LR="2e-5" \
INSTRUCTION_DROPOUT_PROB="0.2" \
TRAIN_EPOCH_STEPS="200" \
SAVE_STEP_START="200" \
SAVE_STEP_END="200" \
SAVE_STEP_INTERVAL="200" \
SHARED_B_LORA="0" \
SHARED_B_RETRIEVAL_ONLY_UPDATE="0" \
JOINT_SINGLE_BRANCH="1" \
GEO_ONLY_BRANCH="1" \
ENABLE_GEO_CONFLICT_PROJECTION="0" \
GEO_REVERSE_WEIGHT="0.0" \
GEO_ZERO_LOSS_WEIGHT="1.0" \
RUN_POSTHOC_MERGED_EVAL="0" \
RUN_POSTHOC_SECOND_MERGED_EVAL="0" \
RUN_POSTHOC_CIRR_EVAL="0" \
MULTIDATASET_EVAL_EVERY="0" \
CIRR_VAL_EVAL_EVERY="0" \
bash "${ROOT}/train_with_dropout.sh"

CKPT="${ROOT}/logs/${RUN_NAME}/checkpoints/epoch_0_step_200.pt"
RUN_DIR="${ROOT}/logs/${RUN_NAME}"

CUDA_VISIBLE_DEVICES=4 \
/data2/mingyu/miniconda3/envs/torch/bin/python \
  "${ROOT}/src/eval_retrieval.py" \
  --resume "${CKPT}" \
  --eval-mode cirr \
  --gpu 0 \
  --batch-size 48 \
  --workers 2 \
  --model ViT-L/14 \
  --img2text-arch phi \
  --middle_dim 3072 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
  --retrieval-prompt-connector and \
  > "${RUN_DIR}/geo_only_step200_cirr.log" 2>&1 &

CIRR_PID=$!

CUDA_VISIBLE_DEVICES=5 \
/data2/mingyu/miniconda3/envs/torch/bin/python \
  "${ROOT}/data/eval_multidataset_suite.py" \
  --resume "${CKPT}" \
  --output-json "${RUN_DIR}/geo_only_step200_multidata.json" \
  --gpu 0 \
  --batch-size 32 \
  --workers 2 \
  --genecis-batch-size 32 \
  --model ViT-L/14 \
  --img2text-arch phi \
  --middle-dim 3072 \
  --img2text-pretrained /data2/mingyu/.cache/torch/hub/checkpoints/SEARLE_ViT-L14.pt \
  --retrieval-prompt-connector that \
  --datasets circo,genecis \
  --name geo_only_step200 &

MULTI_PID=$!

wait "${CIRR_PID}"
wait "${MULTI_PID}"

/data2/mingyu/miniconda3/envs/torch/bin/python - <<'PY'
import json
import pathlib
import re

run_dir = pathlib.Path("/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_GeoOnly_Drop0p2_200_All3Eval")
cirr_log = (run_dir / "geo_only_step200_cirr.log").read_text()
multidata = json.loads((run_dir / "geo_only_step200_multidata.json").read_text())
match = re.search(
    r"Eval composed FeatureR@1: ([0-9.]+), R@5: ([0-9.]+), R@10: ([0-9.]+), R@50: ([0-9.]+), R@100: ([0-9.]+), R_subset@1: ([0-9.]+), R_subset@2: ([0-9.]+), R_subset@3: ([0-9.]+)",
    cirr_log,
)
if not match:
    raise SystemExit("failed to parse CIRR log")
vals = [float(v) for v in match.groups()]
summary = {
    "variant": "geo_only",
    "resume": str(run_dir / "checkpoints" / "epoch_0_step_200.pt"),
    "cirr": {
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
    },
    "circo_val": multidata.get("circo_val"),
    "genecis": multidata.get("genecis"),
}
(run_dir / "geo_only_step200_eval.json").write_text(json.dumps(summary, indent=2))
print(run_dir / "geo_only_step200_eval.json")
PY
