#!/bin/bash
#SBATCH --job-name=gemma_lora                # Nombre del trabajo
#SBATCH --cpus-per-task=8                    # Número de CPUs por tarea
#SBATCH --nodes=1                            # Número de nodos
#SBATCH --ntasks-per-node=1                  # Número de tareas por nodo
#SBATCH --time=20-00:00:00                    # Tiempo máximo para el trabajo (24 horas)
#SBATCH --mem=64GB                           # Memoria asignada
#SBATCH --gres=gpu:1                       # Número de GPUs asignadas
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/logs/lora/train_gemma_lora_vram_tiempo.out  # Salida estándar
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/logs/lora/train_gemma_lora_vram_tiempo.err   # Errores estándar
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

# Activar el entorno virtual
source /gaueko1/users/mmartin/qloraTrain/bin/activate

nvidia-smi


# Lanzar entrenamiento distribuido
#pasar modelo ya cunatizado /gaueko1/users/mmartin/ENVIRONMENT/models/Latxa3.1_8b_lr1e-5_quantized4bit
python /gaueko1/users/mmartin/tfm-quantization-llm/src/LoRa/scripts/train.py \
    --dataset_path /gaueko1/users/mmartin/tfm-quantization-llm/data/openhermes_2eu.json \
    --model "google/gemma-2-9b" \
    --model_type "causal" \
    --lr 0.0005 \
    --save_path /gaueko1/users/mmartin/tfm-quantization-llm/models/LoRa/gemma-2-9b_lora_VRAM_tiempo 