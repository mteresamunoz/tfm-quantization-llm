#!/bin/bash
#SBATCH --job-name=infer_measure
#SBATCH --partition=GPU-ilara
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference/%x_%j.out
#SBATCH --error=logs/inference/%x_%j.err
#SBATCH --mail-type=END,FAIL

echo "========================================"
echo "Inference VRAM/latency measurement"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Load configuration
source "$(dirname "$0")/../config.sh"

# Activate virtual environment
source ${VENV_PATH}

# Environment check
echo ""
echo "Environment Check:"
python --version
pip list | grep -E "torch|transformers|peft|trl|llmcompressor|compressed-tensors"

echo ""
echo "GPU Information (initial):"
nvidia-smi

# Run inference measurement
echo ""
echo "Running inference measurement..."

python src/VRAM_y_tiempo_de_inferencia/VRAM_tiempo_infere.py \
    --model_path ${INFERENCE_MODEL_PATH} \
    --prompt "${INFERENCE_PROMPT}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE}

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Finished with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "========================================"

exit $EXIT_CODE
