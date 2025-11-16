#!/bin/bash

# ==============================================================================
# Shell Script to Run Enhanced Euclidean LoRA Experiments
# ==============================================================================
# This script automates the process of running the main_lora_euclidean.py script
# with different configurations for semantic unlearning using Euclidean space
# with Optimal Transport and repulsive loss mechanisms.
# It iterates through specified datasets, shots, and seeds, setting all required
# command-line arguments.
#
# To Run:
# 1. Make the script executable: chmod +x run_euclidean_semantic.sh
# 2. Execute it: ./run_euclidean_semantic.sh
# ==============================================================================

# --- 1. Configuration ---
PYTHON_SCRIPT="main_lora_euclidean.py"
GPU_ID="0"
ROOT_PATH="/home/nilakshan/0-Unlearning/0-DIET/results"
DATA='/share_folder/nilakshan/FSL_data/'

# Model Configuration
BACKBONE="ViT-B/16"
BATCH_SIZE=32
UNLEARN_EPOCHS=30
POSITION="all"
ENCODER="vision"
R=4
ALPHA=1
NORM_R=1

# Loss Configuration
LAMBDA_HYP=20
LAMBDA_OT=1
LAMBDA_RETAIN=1

# Prototype and OT Configuration
PROTOTYPE_TYPE="eucl"
COST_TYPE="euclidean"
OT_TYPE="sinkhorn"
ASSIGNMENT="OT"
LOSS_TYPE="regular"
NPO_BETA=0.1

# Sinkhorn Configuration
SINKHORN_EPSILON=0.5
SINKHORN_MAX_ITER=2000

# Similarity Configuration
SIMILARITY_THRESHOLD=0.85

# Experiment Configuration
DATASETS_TO_RUN="OxfordFlowers OxfordPets Caltech101"
SHOTS_TO_RUN="16"
SEEDS_TO_RUN="1 2 3 4 5"
LRS_TO_RUN="0.01"

SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_eucl_lora_${LOSS_TYPE}"

# Check if ROOT_PATH exists, if not, create it
if [ ! -d "$ROOT_PATH" ]; then
    echo "ROOT_PATH '$ROOT_PATH' does not exist. Creating it..."
    mkdir -p "$ROOT_PATH"
fi

echo "Starting Enhanced Euclidean LoRA experiments..."
echo "Configuration:"
echo "  - Backbone: ${BACKBONE}"
echo "  - Prototype Type: ${PROTOTYPE_TYPE}"
echo "  - Cost Type: ${COST_TYPE}"
echo "  - OT Type: ${OT_TYPE}"
echo "  - Assignment: ${ASSIGNMENT}"
echo "  - Loss Type: ${LOSS_TYPE}"
echo "  - Lambda Hyp: ${LAMBDA_HYP}"
echo "  - Lambda OT: ${LAMBDA_OT}"
echo "  - Lambda Retain: ${LAMBDA_RETAIN}"
echo ""

for DATASET in $DATASETS_TO_RUN; do
    DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    echo "Dataset: $DATASET_LOWER"

    for SHOTS in $SHOTS_TO_RUN; do
        for SEED in $SEEDS_TO_RUN; do
            for LR in $LRS_TO_RUN; do
                SAVE_PATH="${SAVE_DIR_BASE}/${DATASET_LOWER}/${ENCODER}/${SHOTS}shots/seed${SEED}/"

                echo "=================================================="
                echo "RUNNING: Dataset: ${DATASET}, Shots: ${SHOTS}, Seed: ${SEED}, LR: ${LR}"
                echo "Saving checkpoints to: ${SAVE_PATH}"
                echo "=================================================="

                CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
                    --dataset "${DATASET}" \
                    --data_dir "${DATA}" \
                    --backbone "${BACKBONE}" \
                    --batch_size ${BATCH_SIZE} \
                    --unlearn_lr ${LR} \
                    --norm_r ${NORM_R} \
                    --lambda_hyp ${LAMBDA_HYP} \
                    --lambda_ot ${LAMBDA_OT} \
                    --lambda_retain ${LAMBDA_RETAIN} \
                    --assignment "${ASSIGNMENT}" \
                    --prototype_type "${PROTOTYPE_TYPE}" \
                    --cost_type "${COST_TYPE}" \
                    --seed ${SEED} \
                    --save_path "${SAVE_PATH}" \
                    --filename lora_weights \
                    --unlearn_epochs ${UNLEARN_EPOCHS} \
                    --ot_type "${OT_TYPE}" \
                    --position "${POSITION}" \
                    --encoder "${ENCODER}" \
                    --r ${R} \
                    --alpha ${ALPHA} \
                    --shots ${SHOTS} \
                    --loss_type "${LOSS_TYPE}" 
                    # --npo_beta ${NPO_BETA} \
                    # --sinkhorn_epsilon ${SINKHORN_EPSILON} \
                    # --sinkhorn_max_iter ${SINKHORN_MAX_ITER} \
                    # --similarity_threshold ${SIMILARITY_THRESHOLD}

                if [ $? -eq 0 ]; then
                    echo "SUCCESS: Completed experiment for ${DATASET} with seed ${SEED} and shots ${SHOTS}"
                else
                    echo "ERROR: Failed experiment for ${DATASET} with seed ${SEED} and shots ${SHOTS}"
                fi

                echo ""
            done
        done
    done
done

echo "=================================================="
echo "All enhanced Euclidean experiments finished."
echo "=================================================="
