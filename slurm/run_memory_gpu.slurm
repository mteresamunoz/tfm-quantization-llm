#!/bin/bash
#SBATCH --job-name=memmory_gpu
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/memory/%x_%j.out
#SBATCH --error=logs/memory/%x_%j.err
#SBATCH --mail-type=END,FAIL

source "$(dirname "$0")/../config.sh"

source ${VENV_PATH}

export HF_TOKEN=${HF_TOKEN}

nvidia-smi
# Usage examples:
# - Base model: --base_id <model_id>
# - QLoRA model: --adapter_id <adapter_id> --base_id <base_id> --quant_type <fp4|int8|nf4_dq>
# - Post-quantized: --adapter_id <merge_id> --quant_type <int8|fp4|nf4>

srun python src/memory/memory_usage_qlora.py \
    --adapter_id ${ADAPTER_ID} \
    --base_id ${BASE_MODEL_ID} \
    --quant_type ${QUANT_TYPE}