#!/usr/bin/env bash

# ==============================================================================
# Shell Script to Run GS-LoRA Unlearning Experiments
# ==============================================================================
# This script automates the process of running the main_gslora.py script
# with different configurations. It iterates through specified datasets and seeds,
# setting all required command-line arguments.
#
# To Run:
# 1. Make the script executable: chmod +x run_gs_lora.sh
# 2. Execute it: ./run_gs_lora.sh
# ==============================================================================

# --- 1. Configuration ---
# Set the base parameters for your experiments here.

# The Python script to execute
PYTHON_SCRIPT="main_gslora.py"

# Specify which GPU to use
GPU_ID="1"

# Data directory
DATA_DIR=$1

# Base paths (adjust if your directory structure is different)
ROOT_PATH="."
CLIP_MODEL_PATH="${ROOT_PATH}/clip_models/ViT-B-16.pt"
# A new base directory for this script's outputs to keep things organized

# Training parameters
ITERS=20
BATCH_SIZE=32
LEARNING_RATE=2e-3
WORKERS=8

# LoRA & GS-LoRA parameters
LORA_R=2
LORA_ALPHA=4
GS_ALPHA=0.1
GS_BETA=0.3
ENCODER="vision"
POSITION="all"
GROUP_TYPE="block"

# --- 3. Experiment Loops ---
# Define the datasets and seeds you want to iterate over.

DATASETS_TO_RUN="OxfordFlowers OxfordPets StanfordCars  Food101 Caltech101 cifar10 cifar100 EuroSAT SUN397 UCF101 svhn"

# Seeds for reproducibility (space-separated list)
SEEDS_TO_RUN="1 2 3"

# --- 4. Execution ---
# The main loop that runs the experiments.

echo "Starting GS-LoRA experiments..."

for DATASET in $DATASETS_TO_RUN; do
    # Get the class_to_replace value for this dataset

    for shots in 1 2 4 8; do

    for USE_RETAIN in True False; do
        if [ "$USE_RETAIN" = "True" ]; then
            USE_RETAIN_FLAG="--use_retain"
            SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_gslora_retain_results"
        else
            USE_RETAIN_FLAG=""
            SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_gslora_no_retain_results"
        fi

        for SEED in $SEEDS_TO_RUN; do

            # Convert dataset name to lowercase for the path
            DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

            # Create a unique save path for each experiment run
            FULL_SAVE_PATH="${SAVE_DIR_BASE}/${DATASET_LOWER}/shots_${shots}/seed_${SEED}"

            echo "=================================================="
            echo "RUNNING: Dataset: ${DATASET}, Seed: ${SEED}"
            echo "Saving checkpoints to: ${FULL_SAVE_PATH}"
            echo "=================================================="

            # Set CUDA_VISIBLE_DEVICES and execute the python script
            # The command is broken into multiple lines with '\' for readability.
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
                --data_dir "${DATA_DIR}" \
                --root_path "${ROOT_PATH}" \
                --save_path "${FULL_SAVE_PATH}" \
                --dataset "${DATASET}" \
                --backbone "ViT-B/16" \
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
                --shots ${shots} \
                --num_experts 4 \
                --top_k 2 \
                --dropout_rate 0.1 \
                --no_aug \
                ${USE_RETAIN_FLAG}

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "SUCCESS: Completed experiment for ${DATASET} with seed ${SEED}"
            else
                echo "ERROR: Failed experiment for ${DATASET} with seed ${SEED}"
            fi

            echo ""

        done
    done
    done
done
echo "=================================================="
echo "All experiments finished."
echo "=================================================="