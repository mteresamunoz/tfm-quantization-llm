#!/bin/bash
#SBATCH --job-name=infer_vram_latxa8b      # Nombre del trabajo
#SBATCH --partition=GPU-ilara              # Ajusta a tu partición real
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --cpus-per-task=4                  # CPUs suficientes para tokenización
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G                          # RAM; no muy crítico aquí
#SBATCH --time=02:00:00                    # Tiempo máximo (suele sobrar)
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/logs/VRAM_tiempo_inferencia/infer_vram_postquantFP8_latxa8b.out
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/logs/VRAM_tiempo_inferencia/infer_vram_postquantFP8_latxa8b.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

echo "========================================"
echo "Inference VRAM/latency measurement"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Opcional: limitar a la GPU 0 del nodo asignado
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Activar el entorno donde tienes transformers + torch + compressed-tensors
source /gaueko1/users/mmartin/ENVIRONMENT/bin/activate

# Info rápida de entorno
echo ""
echo "Environment Check:"
python --version
pip list | grep -E "torch|transformers|peft|trl|llmcompressor|compressed-tensors"

echo ""
echo "GPU Information (initial):"
nvidia-smi

# Ejecutar script de medición
echo ""
echo "Running inference measurement..."

python /gaueko1/users/mmartin/tfm-quantization-llm/src/VRAM_y_tiempo_de_inferencia/VRAM_tiempo_infere.py \
    --model_path "/gaueko1/users/mmartin/tfm-quantization-llm/models/PostQuant/FP8/Latxa3.1_8b_lr1e-5--LoRaMIO-FP8-calibration-dynamic-asym" \
    --prompt "Kaixo, azaldu zer da LoRA euskaraz laburki." \
    --max_new_tokens 128 \
    --batch_size 1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Finished with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "========================================"

exit $EXIT_CODE
