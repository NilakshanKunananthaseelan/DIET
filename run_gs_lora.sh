#!/usr/bin/env bash

# ==============================================================================
# Shell Script to Run GS-LoRA Unlearning Experiments
# ==============================================================================
# This script automates the process of running the main_gslora.py script
# with different configurations for semantic unlearning.
# It iterates through specified datasets and seeds, setting all required
# command-line arguments.
#
# To Run:
# 1. Make the script executable: chmod +x run_gs_lora.sh
# 2. Execute it: ./run_gs_lora.sh
# ==============================================================================

# --- 1. Configuration ---
PYTHON_SCRIPT="main_gslora.py"
DATA=$1
GPU_ID="1"
ROOT_PATH="."
SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_gslora_no_retain_results"
BACKBONE="ViT-B/16"
ITERS=20
BATCH_SIZE=32
LEARNING_RATE=2e-3
WORKERS=8
LORA_R=2
LORA_ALPHA=4
GS_ALPHA=0.1
GS_BETA=0.3
ENCODER="vision"
POSITION="all"
GROUP_TYPE="block"
SHOTS=16

DATASETS_TO_RUN="OxfordFlowers"
SEEDS_TO_RUN="1"
USE_RETAIN_TO_RUN="False"

echo "Starting GS-LoRA experiments..."

for DATASET in $DATASETS_TO_RUN; do
    for USE_RETAIN in $USE_RETAIN_TO_RUN; do
        if [ "$USE_RETAIN" = "True" ]; then
            USE_RETAIN_FLAG="--use_retain"
            SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_gslora_retain_results"
        else
            USE_RETAIN_FLAG=""
            SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_gslora_no_retain_results"
        fi

        for SEED in $SEEDS_TO_RUN; do
            DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
            FULL_SAVE_PATH="${SAVE_DIR_BASE}/${DATASET_LOWER}/seed_${SEED}"

            echo "=================================================="
            echo "RUNNING: Dataset: ${DATASET}, Seed: ${SEED}"
            echo "Saving checkpoints to: ${FULL_SAVE_PATH}"
            echo "=================================================="

            CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
                --root_path "${ROOT_PATH}" \
                --data_dir "${DATA}" \
                --save_path "${FULL_SAVE_PATH}" \
                --dataset "${DATASET}" \
                --backbone "${BACKBONE}" \
                --n_iters ${ITERS} \
                --batch_size ${BATCH_SIZE} \
                --lr ${LEARNING_RATE} \
                --seed ${SEED} \
                --workers ${WORKERS} \
                --r ${LORA_R} \
                --alpha ${LORA_ALPHA} \
                --encoder "${ENCODER}" \
                --position "${POSITION}" \
                --gs_alpha ${GS_ALPHA} \
                --gs_beta ${GS_BETA} \
                --group_type "${GROUP_TYPE}" \
                --shots ${SHOTS} \
                --num_experts 4 \
                --top_k 2 \
                --dropout_rate 0.1 \
                --no_aug \
                ${USE_RETAIN_FLAG}

            if [ $? -eq 0 ]; then
                echo "SUCCESS: Completed experiment for ${DATASET} with seed ${SEED}"
            else
                echo "ERROR: Failed experiment for ${DATASET} with seed ${SEED}"
            fi

            echo ""
        done
    done
done

echo "=================================================="
echo "All experiments finished."
echo "=================================================="