#!/bin/bash
#SBATCH --job-name=eval_fp8
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/ptq_exp/logs/eval_Latxa3.1_8b_lr1e-5-FP8-calibration-dynamic-asym2.out
#SBATCH --error=/gaueko1/users/mmartin/ptq_exp/logs/eval_Latxa3.1_8b_lr1e-5-FP8-calibration-dynamic-asym2.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

source /gaueko1/users/mmartin/ENVIRONMENT/bin/activate

tasks_selected=(
    "belebele_eus_Latn"
    "eus_exams_eu"
    "eus_proficiency"
    "eus_reading"
    "eus_trivia"
)

MODEL="/gaueko1/users/mmartin/ptq_exp/models/Latxa3.1_8b_lr1e-5-FP8-calibration-dynamic-asym"
OUTPUT_DIR="/gaueko1/users/mmartin/ptq_exp/results/Latxa3.1_8b_lr1e-5-FP8-calibration-dynamic-asym2"

num_fewshot=5
for task_name in "${tasks_selected[@]}"; do
    if [[ ${task_name} == "eus_reading" || ${task_name} == "belebele_eus_Latn" || ${task_name} == "eus_exams_eu" ]]; then
        batch_size=1
    else
        batch_size="auto:10"
    fi

    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL},add_bos_token=True \
        --tasks ${task_name} \
        --device cuda \
        --output_path ${OUTPUT_DIR} \
        --batch_size ${batch_size} \
        --num_fewshot ${num_fewshot} \
        --log_samples
done

echo "Evaluación completada: ${OUTPUT_DIR}"