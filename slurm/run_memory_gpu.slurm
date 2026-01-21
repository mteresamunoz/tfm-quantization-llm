#!/bin/bash
#SBATCH --job-name=mem_latxa
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/logs/memory_gpu/gemma/qlora_nf4_dq.out
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/logs/memory_gpu/gemma/qlora_nf4_dq.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

source /gaueko1/users/mmartin/ENVIRONMENT/bin/activate

export HF_TOKEN="API_KEY"

nvidia-smi
#si es base --> dar los base
#si es qlora --> dar qlora, base y tipo (fp4, int8, nf4_dq)
#si es postquant --> dar merge y tipo (int8, fp4, nf4)
srun python /gaueko1/users/mmartin/tfm-quantization-llm/src/memory/memory_usage_qlora.py --adapter_id maytemuma/qlora_NF4_gemma_2_9b --base_id google/gemma-3-12b-it --quant_type nf4_dq

