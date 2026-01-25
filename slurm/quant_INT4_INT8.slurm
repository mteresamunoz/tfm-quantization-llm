#!/usr/bin/env bash
#SBATCH --job-name=quantization_INT4_INT8
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/quantization/%x_%j.out
#SBATCH --error=logs/quantization/%x_%j.err

# Load configuration
source "$(dirname "$0")/../config.sh"

# Activate virtual environment
source ${VENV_PATH}

# Run NF4 quantization script
srun python src/quantization/NF4/quantizationNF4.py
