#!/bin/bash
#SBATCH --job-name=eval_llama3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/eval_lmharness/results_FP8/logs/latxa_loraMIO_fp8.out
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/eval_lmharness/results_FP8/logs/latxa_loraMIO_fp8.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

source /gaueko1/users/mmartin/ENVIRONMENT/bin/activate

tasks_selected=(
    "belebele_eus_Latn"
    "eus_exams_eu"
    "eus_proficiency"
    "eus_reading"
    "eus_trivia"
)

num_fewshot=5
for task_name in "${tasks_selected[@]}"; do
    # Set batch_size based on task
    if [[ ${task_name} == "eus_reading" || ${task_name} == "belebele_eus_Latn" || ${task_name} == "eus_exams_eu" ]]; then
        batch_size=1
    else
        batch_size="auto:10"
    fi
#si no deja evaluar directamente modelo, pasar pretrained=modelo base,peft=modelo lora o qlora, etc (NO DEJAR ESPACIOS)
#/gaueko1/users/mmartin/ENVIRONMENT/models/Latxa_8b_loraMIO_quantized4bit
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=/gaueko1/users/mmartin/tfm-quantization-llm/models/PostQuant/FP8/Latxa3.1_8b_lr1e-5--LoRaMIO-FP8-calibration-dynamic-asym \
        --tasks $task_name \
        --device cuda \
        --output_path ../results/ \
        --batch_size ${batch_size} \
        --num_fewshot ${num_fewshot} \
        --log_samples
done