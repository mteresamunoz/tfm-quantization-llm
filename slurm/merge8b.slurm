#!/bin/bash
#SBATCH --job-name=merge                # Nombre del trabajo
#SBATCH --cpus-per-task=8                    # Número de CPUs por tarea
#SBATCH --nodes=1                            # Número de nodos
#SBATCH --ntasks-per-node=1                  # Número de tareas por nodo
#SBATCH --time=10-00:00:00                    # Tiempo máximo para el trabajo (24 horas)
#SBATCH --mem=50GB                           # Memoria asignada
#SBATCH --gres=gpu:1                       # Número de GPUs asignadas
#SBATCH --output=/gaueko1/users/mmartin/qloraTrain/resultsMerge/merge8b.out  # Salida estándar
#SBATCH --error=/gaueko1/users/mmartin/qloraTrain/resultsMerge/merge8b.err   # Errores estándar
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

# Activar el entorno virtual
source /gaueko1/users/mmartin/qloraTrain/bin/activate

# Ejecutar el script de entrenamiento
srun python /gaueko1/users/mmartin/qloraTrain/merge/mergeLoraBase8b.py