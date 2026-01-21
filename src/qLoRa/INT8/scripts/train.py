# INITIAL IMPORTS
import time
import subprocess
import numpy as np
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
import argparse
import wandb
from load_dataset import *
import os


def log_nvidia_smi(tag=""):
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]).decode("utf-8").strip()
        print(f"[nvidia-smi]{tag} {out}")
    except Exception as e:
        print(f"[nvidia-smi]{tag} error: {e}")


def train_formatting_function(data):
    formated_sen_chat = []
    system_found = False

    for conv in data["conversations"]:
        if isinstance(conv, dict):
            if conv.get("from") == "system":
                role = "system"
                system_found = True
            elif conv.get("from") == "human":
                role = "user"
            elif conv.get("from") == "gpt":
                role = "assistant"

            if not system_found:
                formated_sen_chat.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant who answers questions in Basque."
                })
                system_found = True

            formated_sen_chat.append({"role": role, "content": conv.get("value")})

    formated_sen = ""
    for msg in formated_sen_chat:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formated_sen += f"User: {content}\n"
        elif role == "assistant":
            formated_sen += f"Assistant: {content}\n"
        elif role == "system":
            formated_sen += f"[System Message]: {content}\n"

    return {"text": formated_sen}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    # WANDB
    #os.environ["WANDB_API_KEY"] = "API_KEY"
    #wandb.init(project="qlora-8b-8bit", name="qlora-8b-8bit-retrain")

    model_chk = args.model
    model_type = args.model_type
    lr = args.lr
    save_path = args.save_path
    dataset_path = args.dataset_path

    # SEEDS
    torch.manual_seed(42)
    np.random_seed = np.random.seed(42)

    # HYPERPARAMETERS
    bs = 24
    epochs = 5
    max_seq_length = 1024
    wd = 0.01

    print("=" * 60)
    print("TRAINING CONFIGURATION (QLoRA 8-bit)")
    print("=" * 60)
    print(f"Model: {model_chk}")
    print(f"Dataset: {dataset_path}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Weight decay: {wd}")
    print(f"Save path (logs/checkpoints): {save_path}")
    print("=" * 60)

    # BitsAndBytes config (8-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
        # load_in_8bit_fp32_cpu_offload=True,
    )

    # 1) Load model in 8-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_chk,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    print("✓ 8-bit base model loaded")

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_chk, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded")

    # 3) Prepare for k-bit training + LoRA
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("✓ LoRA configured on 8-bit model")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 4) Dataset
    print("\n[3/6] Loading dataset...")
    dataset = load_dataset(dataset_path)
    formated_train = dataset["train"].map(train_formatting_function, batched=False)
    formated_dev = dataset["dev"].map(train_formatting_function, batched=False)
    print("✓ Dataset loaded")
    print(f"  Train samples: {len(formated_train):,}")
    print(f"  Dev samples: {len(formated_dev):,}")

    def has_valid_text(example):
        """Filtra textos vacíos o muy cortos"""
        text = example.get("text", "").strip()
        return len(text) > 20  # Ajusta threshold si quieres

    print(formated_train[0])  # Verás que "text" está completo

    # Filtrar + remover columnas problemáticas
    formated_train = formated_train.filter(has_valid_text)
    formated_train = formated_train.remove_columns(["conversations"])  # Evita ChatML auto

    print(formated_train[0])  # Verás que "text" está completo

    formated_dev = formated_dev.filter(has_valid_text) 
    formated_dev = formated_dev.remove_columns(["conversations"])

    print(f"Train samples after filter: {len(formated_train):,}")
    print(f"Dev samples after filter: {len(formated_dev):,}")


    # 5) TrainingArguments + Trainer
    print("\n[4/6] Setting up trainer...")
    training_args = SFTConfig(
        output_dir=save_path + "gemma",
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        weight_decay=wd,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        #report_to="wandb",
        max_length=max_seq_length,
        dataset_num_proc=4,
        packing=False,
        #formatting_func=None,
    )

    trainer = SFTTrainer(
        model=model,
        #tokenizer=tokenizer,
        processing_class=tokenizer,
        args=training_args,
        #peft_config=peft_config,
        train_dataset=formated_train,
        eval_dataset=formated_dev,
        #dataset_text_field="text",
        #dataset_num_proc=1,
        #max_length=max_seq_length,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        #packing=False,
        
    )

    steps_per_epoch = len(formated_train) // (4 * 8)
    total_steps = steps_per_epoch * epochs
    print("✓ Trainer configured")
    print(f"  Steps per epoch: ~{steps_per_epoch:,}")
    print(f"  Total steps: ~{total_steps:,}")

    # 6) Medir VRAM + tiempo de ENTRENAMIENTO COMPLETO
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    log_nvidia_smi(tag="[before_train_qlora8b]")

    print("\n[5/6] Starting QLoRA-8bit training with metrics...")
    print("=" * 60)
    train_start = time.time()
    train_output = trainer.train()
    train_elapsed = time.time() - train_start
    print("=" * 60)
    print("✓ QLoRA-8bit training completed")

    peak_bytes_train = torch.cuda.max_memory_allocated(device)
    peak_gb_train = peak_bytes_train / 1024**3
    print(f"[METRICS][train] peak_train_vram_gb={peak_gb_train:.2f}")
    print(f"[METRICS][train] train_time_min={train_elapsed/60:.1f}")
    log_nvidia_smi(tag="[after_train_qlora8b]")

    print("\n[6/6] Saving final model...")
    final_model_path = os.path.join(save_path, "best_model")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Model saved to: {final_model_path}")

    # 7) No guardar modelo de nuevo
    model.config.use_cache = True
    print("\nFinal evaluation (for logging)...")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    print("\n" + "=" * 60)
    print("QLoRA-8bit TRAINING FINISHED SUCCESSFULLY")
    print("=" * 60)
