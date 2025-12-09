#!/bin/bash
#SBATCH --job-name=24qlora4bit                # Nombre del trabajo
#SBATCH --cpus-per-task=8                    # Número de CPUs por tarea
#SBATCH --nodes=1                            # Número de nodos
#SBATCH --ntasks-per-node=1                  # Número de tareas por nodo
#SBATCH --time=20-00:00:00                    # Tiempo máximo para el trabajo (24 horas)
#SBATCH --mem=64GB                           # Memoria asignada
#SBATCH --gres=gpu:1                       # Número de GPUs asignadas
#SBATCH --output=/gaueko1/users/mmartin/qloraTrain/results_latxa8b_qlora4b/train_4bitbs24.out  # Salida estándar
#SBATCH --error=/gaueko1/users/mmartin/qloraTrain/results_latxa8b_qlora4b/train_4bitbs24.err   # Errores estándar
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

# Activar el entorno virtual
source /gaueko1/users/mmartin/qloraTrain/bin/activate

# Configuración (solo hace falta una vez, pero aquí por seguridad)
#accelerate config default

# Lanzar entrenamiento distribuido
#pasar modelo ya cunatizado /gaueko1/users/mmartin/ENVIRONMENT/models/Latxa3.1_8b_lr1e-5_quantized4bit
python /gaueko1/users/mmartin/qloraTrain/qlora-latxa-8b-4bit/scripts/train.py \
    --dataset_path /gaueko1/users/mmartin/qloraTrain/data/openhermes_2eu.json \
    --model "/proiektuak/ikergaitu-data/azabala106/model_evaluation/trained_models/Latxa3.1_8b_lr1e-5" \
    --model_type "causal" \
    --lr 0.0005 \
    --save_path /gaueko1/users/mmartin/qloraTrain/qlora-latxa-8b-4bit/models/bs24maxSeq512/