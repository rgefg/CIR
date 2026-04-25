#!/bin/bash

# ================= 配置区域 =================
# 权重路径
CHECKPOINT="/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_EMA1700_QKV_StrictLoss_dropout0.5/checkpoints/epoch_1.pt"

# 通用设置
GPU_ID=7
BATCH_SIZE=32
MODEL="ViT-L/14"
# ===========================================

echo "========================================================"
echo "Starting Evaluation using checkpoint:"
echo "$CHECKPOINT"
echo "========================================================"

# ---------------------------------------------------------
# 1. CIRCO Evaluation
# ---------------------------------------------------------
echo ""
echo "[1/4] Evaluating CIRCO..."
python src/eval_retrieval.py \
    --model $MODEL \
    --openai-pretrained \
    --gpu $GPU_ID \
    --batch-size $BATCH_SIZE \
    --resume "$CHECKPOINT" \
    --eval-mode circo

# ---------------------------------------------------------
# 2. GeneCIS Evaluation (4 Tasks)
# ---------------------------------------------------------
echo ""
echo "[2/4] Evaluating GeneCIS..."
for TASK in focus_attribute change_attribute focus_object change_object
do
    echo "  -> Running GeneCIS Task: $TASK"
    python src/eval_retrieval.py \
        --model $MODEL \
        --openai-pretrained \
        --gpu $GPU_ID \
        --batch-size $BATCH_SIZE \
        --resume "$CHECKPOINT" \
        --eval-mode genecis \
        --genecis-task $TASK
done

# ---------------------------------------------------------
# 3. FashionIQ Evaluation (3 Categories)
# ---------------------------------------------------------
echo ""
echo "[3/4] Evaluating FashionIQ..."
for CATEGORY in dress shirt toptee
do
    echo "  -> Running FashionIQ Category: $CATEGORY"
    python src/eval_retrieval.py \
        --model $MODEL \
        --openai-pretrained \
        --gpu $GPU_ID \
        --batch-size $BATCH_SIZE \
        --resume "$CHECKPOINT" \
        --eval-mode fashion \
        --source-data $CATEGORY \
        --target-pad
done

# ---------------------------------------------------------
# 4. CIRR Evaluation (Validation Set)
# ---------------------------------------------------------
echo ""
echo "[4/4] Evaluating CIRR (Validation)..."
python src/eval_retrieval.py \
    --model $MODEL \
    --openai-pretrained \
    --gpu $GPU_ID \
    --batch-size $BATCH_SIZE \
    --resume "$CHECKPOINT" \
    --eval-mode cirr

echo ""
echo "========================================================"
echo "All evaluations completed."
echo "========================================================"
