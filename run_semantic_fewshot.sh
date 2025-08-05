
# ==============================================================================
# Shell Script to Run Semantic LoRA Few-Shot Experiments (Hyp-Busemann)
# ==============================================================================
# This script automates the process of running the main_lora_hypebuseman.py script
# with different configurations for semantic unlearning with few-shot settings.
# It iterates through specified datasets, shots, and seeds, setting all required
# command-line arguments.
#
# To Run:
# 1. Make the script executable: chmod +x run_semantic_fewshot.sh
# 2. Execute it: ./run_semantic_fewshot.sh <DATA_ROOT>
# ==============================================================================

# --- 1. Configuration ---
PYTHON_SCRIPT="main_lora_hypebuseman.py"
GPU_ID="1"
ROOT_PATH="."
DATA=$1
SAVE_DIR_BASE="${ROOT_PATH}/checkpoints_hyp_lora_fewshots"
BACKBONE="ViT-B/16"
BATCH_SIZE=32
UNLEARN_EPOCHS=30
POSITION="all"
ENCODER="vision"
R=4
ALPHA=1
NORM_R=1
LAMBDA_HYP=30
LAMBDA_OT=1
LAMBDA_RETAIN=30
PROTOTYPE_TYPE="eucl"
COST_TYPE="busemann"
OT_TYPE="sinkhorn"

DATASETS_TO_RUN="Food101 cifar10 cifar100 svhn UCF101"
SHOTS_TO_RUN="1 2 4 8"
SEEDS_TO_RUN="1 2 3"
LRS_TO_RUN="0.0009"

echo "Starting Semantic LoRA Few-Shot (Hyp-Busemann) experiments..."

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
                    --shots ${SHOTS}

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
echo "All semantic few-shot experiments finished."
echo "=================================================="