
# ==============================================================================
# Shell Script to Run Baseline LoRA Experiments (for CIFAR10 and CIFAR100)
# ==============================================================================

# Configuration
PYTHON_SCRIPT="main_lora_baseline.py"
GPU_ID="0"
ROOT_PATH="."
SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_baseline_new"
BACKBONE="ViT-B/16"
BATCH_SIZE=32
LEARNING_RATE=2e-4
UNLEARN_EPOCHS=30
POSITION="all"
LORA_R=4
LORA_ALPHA=4

DATASETS_TO_RUN="cifar10 cifar100"
SEEDS_TO_RUN="1 2 3 4 5"
ENCODERS_TO_RUN="vision"

echo "Starting Baseline LoRA experiments..."

for DATASET in $DATASETS_TO_RUN; do
    DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    echo "Dataset: $DATASET_LOWER"
    for ENCODER in $ENCODERS_TO_RUN; do
        for SEED in $SEEDS_TO_RUN; do
            SAVE_PATH="${SAVE_DIR_BASE}/${DATASET_LOWER}/${ENCODER}/${SEED}"

            echo "=================================================="
            echo "RUNNING: Dataset: ${DATASET}, Encoder: ${ENCODER}, Seed: ${SEED}"
            echo "Saving checkpoints to: ${SAVE_PATH}"
            echo "=================================================="

            CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
                --dataset "${DATASET}" \
                --backbone "${BACKBONE}" \
                --batch_size ${BATCH_SIZE} \
                --lr ${LEARNING_RATE} \
                --seed ${SEED} \
                --save_path "${SAVE_PATH}" \
                --filename lora_weights \
                --unlearn_epochs ${UNLEARN_EPOCHS} \
                --position "${POSITION}" \
                --encoder "${ENCODER}" \
                --r ${LORA_R} \
                --alpha ${LORA_ALPHA}

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
echo "All baseline experiments finished."
echo "=================================================="
