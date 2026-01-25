#!/bin/bash
#SBATCH --job-name=qlora_4bit                # Nombre del trabajo
#SBATCH --cpus-per-task=8                    # Número de CPUs por tarea
#SBATCH --nodes=1                            # Número de nodos
#SBATCH --ntasks-per-node=1                  # Número de tareas por nodo
#SBATCH --time=10:00:00                    # Tiempo máximo para el trabajo (24 horas)
#SBATCH --mem=64GB                           # Memoria asignada
#SBATCH --gres=gpu:1                       # Número de GPUs asignadas
#SBATCH --output=logs/training/%x_%j.out
#SBATCH --error=logs/training/%x_%j.err
#SBATCH --mail-type=END,FAIL               # Tipo de notificación por correo

# Activar el entorno virtual
source "$(dirname "$0")/../config.sh"

source ${VENV_PATH}

# Run QLoRA training
python src/qLoRa/INT4/scripts/train.py \
    --dataset_path ${DATASET_PATH} \
    --model ${BASE_MODEL_ID} \
    --model_type "causal" \
    --lr ${LEARNING_RATE} \
    --save_path ${QLORA_INT4_SAVE_PATH}