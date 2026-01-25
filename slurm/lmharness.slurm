#!/bin/bash
#SBATCH --job-name=evaluation_lmharness
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lm_harness/%x_%j.out
#SBATCH --error=logs/lm_harness/%x_%j.err
#SBATCH --mail-type=END,FAIL

source "$(dirname "$0")/../config.sh"

source ${VENV_PATH}

tasks_selected=(
    "belebele_eus_Latn"
    "eus_exams_eu"
    "eus_proficiency"
    "eus_reading"
    "eus_trivia"
)

export HF_TOKEN=${HF_TOKEN}

num_fewshot=5

for task_name in "${tasks_selected[@]}"; do
    # Set batch_size based on task
    if [[ ${task_name} == "eus_reading" || ${task_name} == "belebele_eus_Latn" || ${task_name} == "eus_exams_eu" ]]; then
        batch_size=1
    else
        batch_size="auto:10"
    fi

    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL_PATH} \
        --tasks $task_name \
        --device cuda \
        --output_path ${RESULTS_DIR} \
        --batch_size ${batch_size} \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
