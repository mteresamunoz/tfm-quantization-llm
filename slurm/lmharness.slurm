#!/bin/bash
#SBATCH --job-name=eval_ql8qwen
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/eval_lmharness/gemma/qlora_int8_eustrivia.out
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/eval_lmharness/gemma/qlora_int8_eustrivia.err
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

export HF_TOKEN=YOUR_TOKEN

num_fewshot=5
for task_name in "${tasks_selected[@]}"; do
    # Set batch_size based on task
    if [[ ${task_name} == "eus_reading" || ${task_name} == "belebele_eus_Latn" || ${task_name} == "eus_exams_eu" ]]; then
        batch_size=1
    else
        batch_size="auto:10"
    fi
#si no deja evaluar directamente modelo, pasar pretrained=modelo base,peft=modelo lora o qlora, etc (NO DEJAR ESPACIOS), o poner llama o latxa en el nombre
#/gaueko1/users/mmartin/ENVIRONMENT/models/Latxa_8b_loraMIO_quantized4bit
#google/gemma-2-9b,token=$HF_TOKEN
#${batch_size} \
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=/gaueko1/users/mmartin/tfm-quantization-llm/models/qLoRa/INT8/gemma/best_model \
        --tasks $task_name \
        --device cuda \
        --output_path ../results/ \
        --batch_size 1 \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
