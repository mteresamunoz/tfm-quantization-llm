#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/merge/%x_%j.out
#SBATCH --error=logs/merge/%x_%j.err
#SBATCH --mail-type=END,FAIL

# Load configuration
source "$(dirname "$0")/../config.sh"

# Activate virtual environment
source ${VENV_PATH}

# Run merge script
srun python src/merge/merge.py
