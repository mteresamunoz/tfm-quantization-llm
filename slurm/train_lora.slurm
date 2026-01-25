#!/bin/bash
#SBATCH --job-name=train_lora
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20-00:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/training/%x_%j.out
#SBATCH --error=logs/training/%x_%j.err
#SBATCH --mail-type=END,FAIL

# Load configuration
source "$(dirname "$0")/../config.sh"

# Activate virtual environment
source ${VENV_PATH}

# Display GPU info
nvidia-smi

# Run LoRA training (full precision, no quantization)
python src/LoRa/scripts/train.py \
    --dataset_path ${DATASET_PATH} \
    --model ${BASE_MODEL_ID} \
    --model_type "causal" \
    --lr ${LEARNING_RATE} \
    --save_path ${LORA_SAVE_PATH}
