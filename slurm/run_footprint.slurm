#!/bin/bash
#SBATCH --job-name=footprint          # Nombre del trabajo
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/logs/footprint/run_50_falta.out
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/logs/footprint/run_50_falta.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#module purge
#module load cuda/12.1 python/3.9  # Ajusta versiones
nvidia-smi --gpu-reset -i 0
# 1) Activar entorno (ajusta si usas otro)
source /gaueko1/users/mmartin/ENVIRONMENT/bin/activate

# pip install --user codecarbon

# 3) Ruta del modelo a evaluar (CAMBIAR para cada job)
#MODEL_PATH="/gaueko1/users/mmartin/tfm-quantization-llm/models/qLoRa/NF4/best_model"

#PROMPT="Kaixo, azaldu zer da LoRA euskaraz laburki."
#MAX_NEW_TOKENS=128

#echo "Measuring footprint for: ${MODEL_PATH}"
nvidia-smi

python /gaueko1/users/mmartin/tfm-quantization-llm/src/footprint/codeCarbon_inference.py #\
#  --model_path "${MODEL_PATH}" \
#  --prompt "${PROMPT}" \
#  --max_new_tokens ${MAX_NEW_TOKENS} \
#  --device cuda
