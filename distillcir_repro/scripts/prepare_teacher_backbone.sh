#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-xtuner/llava-phi-3-mini-hf}"
LOCAL_DIR="${LOCAL_DIR:-/data2/mingyu/composed_image_retrieval/checkpoint/hf_models/xtuner_llava_phi3_mini_hf}"

mkdir -p "$LOCAL_DIR"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "Downloading teacher backbone: $MODEL_ID"
echo "Local dir: $LOCAL_DIR"
echo "Set HF_ENDPOINT=https://hf-mirror.com before running if the normal Hugging Face endpoint is slow."

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False
else
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$LOCAL_DIR",
    local_dir_use_symlinks=False,
)
PY
fi

python - <<PY
from pathlib import Path
root = Path("$LOCAL_DIR")
required = ["config.json"]
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit(f"teacher backbone download incomplete, missing: {missing}")
print(f"teacher backbone ready: {root}")
PY
