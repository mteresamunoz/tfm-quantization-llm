#!/usr/bin/env bash
#SBATCH --job-name=qNF4                # Nombre del trabajo
#SBATCH --cpus-per-task=4                    # Número de CPUs por tarea
#SBATCH --nodes=1                            # Número de nodos
#SBATCH --ntasks-per-node=1                  # Número de tareas por nodo
#SBATCH --time=24:00:00                      # Tiempo máximo para el trabajo (24 horas)
#SBATCH --mem=50GB                           # Memoria asignada
#SBATCH --gres=gpu:1                       # Número de GPUs asignadas
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/logs/otros/postquant_NF4_gemma.out  # Salida estándar
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/logs/otros/postquant_NF4_gemma.err   # Errores estándar

# Activar el entorno virtual
source /gaueko1/users/mmartin/qlora8bit/bin/activate

# Specify common cache for Transformers library
srun python /gaueko1/users/mmartin/tfm-quantization-llm/src/quantization/NF4/quantizationNF4.py