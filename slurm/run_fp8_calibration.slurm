#!/bin/bash
#SBATCH --job-name=fp8_latxa
#SBATCH --output=/gaueko1/users/mmartin/tfm-quantization-llm/logs/otros/quant_loraMIO_fp8_dynamic_asym_latxa.out
#SBATCH --error=/gaueko1/users/mmartin/tfm-quantization-llm/logs/otros/quant_loraMIO_fp8_dynamic_latxa.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=GPU-ilara
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

echo "========================================"
echo "FP8 Quantization dynamic asym - Latxa3.1_8b LoRA MIO"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Activar entorno virtual
source /gaueko1/users/mmartin/ptq_exp/ptq_env_py311/bin/activate

# Mostrar info de GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Ejecutar cuantización
python /gaueko1/users/mmartin/tfm-quantization-llm/src/quantization/FP8/fp8_calibration.py

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"