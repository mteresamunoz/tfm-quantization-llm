# QLoRA vs LoRA-Q Pipelines for Basque: a Multi-Model Efficiency Analysis

This repository contains the code, experiments, and results for a Master's Thesis (TFM - HiTZ GreenAI) investigating the impact of quantization techniques on Large Language Models (LLMs) for the Basque language.

## Project Overview

### Introduction

#### Motivation
The recent surge in Natural Language Processing (NLP) has been driven by the unprecedented scaling of Large Language Models (LLMs). State-of-the-art generalist models, such as Qwen and Gemma, have demonstrated that increasing parameter counts and training data leads to remarkable generalization capabilities. However, this progress has created a significant "computational wall." The hardware resources required to train and deploy these massive architectures are substantial, raising concerns about sustainability and accessibility.

This issue has catalyzed the Green AI movement, which advocates for making efficiency a core evaluation criterion alongside accuracy. The environmental cost of deep learning is non-negligible; the carbon footprint of training a single large transformer model can be equivalent to the lifetime emissions of multiple automobiles. Consequently, reducing the energy consumption of inference is not merely an engineering optimization but an ethical imperative for the field.

The implications of this resource barrier are particularly acute for low-resource languages like Basque (Euskera). While English-centric research benefits from massive industrial infrastructure, low-resource language communities often operate in a "data-rich, compute-poor" or "data-poor, compute-poor" regime. Developing competitive models for Basque, such as the recently released Latxa, requires maximizing the utility of limited hardware. In this context, the ability to run high-performance models on consumer-grade GPUs or limited academic clusters is essential for the democratization of language technology.

To address these constraints, the community relies on two primary techniques: **Quantization** and **Parameter-Efficient Fine-Tuning (PEFT)**. Quantization compresses the model by reducing the precision of its weights (e.g., from 16-bit floating points to 4-bit integers), which dramatically lowers memory requirements. Simultaneously, PEFT methods like LoRA (Low-Rank Adaptation) enable the adaptation of these frozen models to specific languages or tasks by training only a small subset of parameters.

The intersection of these techniques gave rise to **QLoRA**, a method that quantizes the model before fine-tuning to minimize training memory usage. While QLoRA has become a standard for accessibility, an important research gap remains: does the aggressive compression of the base model prior to adaptation hinder the learning process in linguistically complex, low-resource scenarios? It is unclear whether it is more effective to quantize first (as in QLoRA) or to perform adaptation in higher precision and quantize later (**Post-Training Quantization**).

This thesis investigates this trade-off. It explores whether the order of operations—quantization and adaptation—affects the downstream performance of LLMs in Basque. By analyzing both domain-specific models (such as Latxa) and generalist models (like Qwen and Gemma), this work aims to provide a robust, evidence-based guideline for deploying efficient, high-quality AI in low-resource environments.

### Research Questions and Hypothesis

The primary objective of this study is to determine the optimal inference architecture for the Basque language that balances computational efficiency with linguistic integrity. To achieve this, the research is guided by three hierarchical questions:

#### RQ1. The pipeline order dilemma (Core question)
**Is the Post-Training Quantization strategy (adapting in high precision, then compressing) superior to the QLoRA approach (compressing first, then adapting) for a low-resource language?**
To ensure a fair comparison, this study contrasts two pipelines that utilize identical quantization algorithms, differing only in the sequence of application.

*   **Hypothesis 1.1**: We hypothesize that **Pipeline A (QLoRA)** will yield superior accuracy and linguistic coherence compared to Pipeline B (LoRA-Q). By updating the adapter weights while the base model is already in its quantized representation, the training process can compensate for the quantization error ("quantization-aware adaptation"). In contrast, quantizing a model after adaptation (Pipeline B) introduces noise that the adapter weights were not optimized to handle, likely degrading performance in sensitive morphological tasks.
*   **Hypothesis 1.2**: We anticipate a dichotomy between memory and speed. While quantization (in both pipelines) will drastically reduce VRAM usage compared to the baseline, we hypothesize that **inference latency will increase relative to the uncompressed BF16 model**. This is due to the computational overhead required for on-the-fly dequantization during the forward pass. Furthermore, regarding training efficiency, we hypothesize that while QLoRA minimizes memory footprint, it may incur longer training times depending on the efficiency of the specific quantization kernel (e.g., Int8 vs. NF4) used during the backward pass.

#### RQ2. Generalization across model families
**Does the impact of the quantization pipeline vary between domain-specific models (pre-trained on Basque, e.g., Latxa) and generalist multilingual models (e.g., Qwen, Gemma)?**

*   **Hypothesis 2**: We hypothesize that domain-specific models like **Latxa** will exhibit greater robustness to quantization degradation than generalist models. Since Latxa's internal representations are already aligned with the target language, we expect it to retain a higher percentage of its baseline performance after compression. Conversely, generalist models, which rely more heavily on cross-lingual transfer, may suffer more severe "catastrophic forgetting" of Basque grammar when their weights are aggressively compressed.

#### RQ3. Impact of quantization methods
**Which quantization method offers the best trade-off between compression and accuracy for Basque text generation: Standard Integer Quantization (Int8), Floating Point (FP4), or Normal Float with Double Quantization (NF4+DQ)?**

*   **Hypothesis 3**: We hypothesize that the **NF4 (Normal Float 4-bit) + Double Quantization** format will outperform standard Int8 and FP4 in terms of accuracy. Since the weights of pre-trained LLMs typically follow a normal distribution, the NF4 data type is theoretically optimal for minimizing information loss. We expect this precision to be the determining factor in maintaining the complex morphological structure of Basque, outperforming the linear mapping of Int8.

---

## Repository Structure

The project is organized as follows:

- **`src/`**: Source code for training, quantization, and evaluation.
    - **`LoRa/`**: Scripts for standard Low-Rank Adaptation (Fine-tuning in high precision).
    - **`qLoRa/`**: Scripts for Quantized LoRA (Fine-tuning quantized models) supporting INT4, INT8, and NF4.
    - **`quantization/`**: Scripts for Post-Training Quantization strategies (INT4, INT8, NF4).
    - **`VRAM_y_tiempo_de_inferencia/`**: Benchmarking scripts to measure VRAM usage and inference latency.
    - **`graficas/`**: Python scripts for generating plots and visualizations.
- **`slurm/`**: SLURM job scripts for running experiments on HPC clusters (e.g., `train_latxa8b_nf4.slurm`, `lmharness.slurm`).
- **`results_csv/`**: Contains raw experimental results and metrics for different models (Gemma, Latxa, Qwen).
- **`figures/`**: Generated plots for analysis, including bar charts and scatter plots.

---

## Installation

Ensure you have Python 3.9+ and CUDA drivers installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/tfm-quantization-llm.git
    cd tfm-quantization-llm
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment or Conda environment.
    ```bash
    pip install torch transformers peft trl bitsandbytes accelerate scipy pandas seaborn matplotlib wandb
    ```

---

## Usage

### 1. Training (Fine-Tuning)

You can launch training jobs using the python scripts in `src` or the SLURM scripts for cluster execution.

**Example: Training with LoRA (High Precision Base)**
```bash
python src/LoRa/scripts/train.py \
    --dataset_path path/to/dataset.json \
    --model "google/gemma-2-9b" \
    --model_type "causal" \
    --lr 0.0002 \
    --save_path models/lora/gemma/
```

**Example: Training with QLoRA (NF4 Quantization)**
See `slurm/train_latxa8b_nf4.slurm` for the full configuration.
```bash
python src/qLoRa/NF4/train.py \
    --dataset_path path/to/dataset.json \
    --model "google/gemma-2-9b" \
    --model_type "causal" \
    --lr 0.0002 \
    --save_path models/qLoRa/NF4/gemma/
```

### 2. Measuring Efficiency (VRAM & Latency)
To verify Hypothesis 1.2 regarding memory and speed:
```bash
python src/VRAM_y_tiempo_de_inferencia/VRAM_tiempo_infere.py
```

### 3. Visualizing Results
Generate plots to compare performance across different configurations (RQ1, RQ2).
```bash
python src/graficas/generate_plots_advanced.py
```
Outputs will be saved to the `figures/` directory.

---

## Results & Artifacts

*   **Metrics**: Raw performance metrics for all experiments are stored in the `results_csv/` directory.
*   **Visualizations**: Validated figures supporting the hypotheses can be found in `figures/`.

---

## Contact

For questions or inquiries regarding this research, please contact the author.
