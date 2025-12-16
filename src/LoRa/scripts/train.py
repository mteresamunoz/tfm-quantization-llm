# INITIAL IMPORTS
#para medir VRAM en entrenamiento
import time
import subprocess
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTTrainer
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
    """
    Format the dataset for training with questions and answers in Basque.
    """

    #print(data)
    #full_text = []
    formated_sen_chat = []

    system_found = False

    for conv in data['conversations']:

        if isinstance(conv, dict):

            if conv.get("from") == "system":
                role = "system"
                system_found = True
            elif conv.get("from") == "human":
                role = "user"
            elif conv.get("from") == "gpt":
                role = "assistant"
            
            if not system_found:
                formated_sen_chat.insert(0, {"role":"system", "content": "You are a helpful assistant who answers questions in Basque."})
                system_found = True
            
            formated_sen_chat.append({"role": role, "content": conv.get('value')})

    

    #print(formated_sen_chat)
    #formated_sen = tokenizer.apply_chat_template(formated_sen_chat, add_generation_prompt=True, tokenize=False)

    formated_sen = ""
    for msg in formated_sen_chat:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formated_sen += f"User: {content}\n"
        elif role == "assistant":
            formated_sen += f"Assistant: {content}\n"
        elif role == "system":
            formated_sen += f"System Message: {content}\n"
  

    return {"text": formated_sen}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    # Inicializar wandb
    os.environ["WANDB_API_KEY"] = "YOUR-APY-KEY"
    wandb.init(project="lora-fp8-latxa3.1_8b", name="latxa3.1_8b-lora-fp8")

    # Variables
    model_chk = args.model
    model_type = args.model_type
    lr = args.lr
    save_path = args.save_path
    dataset_path = args.dataset_path

    # SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)
    
    # HYPERPARAMETERS
    epochs = 5
    max_seq_length = 512
    wd = 0.01
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {model_chk}")
    print(f"Dataset: {dataset_path}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Weight decay: {wd}")
    print(f"Save path: {save_path}")
    print("="*60)

    # Cargar modelo
    print("\n[1/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_chk, 
        device_map="auto",
        use_safetensors=True,
        torch_dtype=torch.bfloat16  # ← AÑADIDO para ahorrar memoria
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    print("✓ Model loaded")
    
    # Cargar tokenizer
    print("\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_chk, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded")

    # Preparar modelo para k-bit training (quitar si no es para bnb)
    #print("\n[3/6] Preparing model for training...")
    #model = prepare_model_for_kbit_training(model)

    # Configurar LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    # Imprimir parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA configured")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Cargar dataset
    print("\n[4/6] Loading dataset...")
    dataset = load_dataset(dataset_path)
    formated_train = dataset["train"].map(train_formatting_function, batched=False)
    formated_dev = dataset["dev"].map(train_formatting_function, batched=False)
    print(f"✓ Dataset loaded")
    print(f"  Train samples: {len(formated_train):,}")
    print(f"  Dev samples: {len(formated_dev):,}")

    # TRAINING
    print("\n[5/6] Setting up trainer...")
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        learning_rate=lr,
        weight_decay=wd,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        optim="adamw_torch",
        report_to="wandb",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=formated_train,
        eval_dataset=formated_dev,
        dataset_text_field="text",
        dataset_num_proc=4,  # ← CAMBIADO de 1 a 4
        max_seq_length=max_seq_length,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        packing=False,
    )

    # Calcular steps estimados
    steps_per_epoch = len(formated_train) // (8 * 4)  # batch_size * grad_accum
    total_steps = steps_per_epoch * epochs
    print(f"✓ Trainer configured")
    print(f"  Steps per epoch: ~{steps_per_epoch:,}")
    print(f"  Total steps: ~{total_steps:,}")
    print(f"  Estimated time: 6-8 hours")

    # --------------------------------------------------
    # MEDICIÓN DE VRAM Y TIEMPO EN UN MINI-STEP DE TRAIN
    # --------------------------------------------------
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)  # resetea contador de pico [web:25][web:51]
    log_nvidia_smi(tag="[before_warmup]")

    # Coger un solo batch representativo del DataLoader interno
    train_dataloader = trainer.get_train_dataloader()
    first_batch = next(iter(train_dataloader))

    model.train()
    start = time.time()
    # forward + loss
    outputs = model(
        input_ids=first_batch["input_ids"].to(device),
        attention_mask=first_batch["attention_mask"].to(device),
        labels=first_batch["input_ids"].to(device),
    )
    loss = outputs.loss
    loss.backward()
    # OJO: no llamamos optimizer.step() aquí para no interferir mucho con el Trainer

    elapsed = time.time() - start
    peak_bytes = torch.cuda.max_memory_allocated(device)  # pico desde el reset [web:25][web:56]
    peak_gb = peak_bytes / 1024**3

    print(f"[METRICS][warmup_step] time={elapsed:.3f}s, peak_vram={peak_gb:.2f}GB")
    log_nvidia_smi(tag="[after_warmup]")

    # Entrenar
    print("\n[6/6] Starting training...")
    print("="*60)
    trainer.train()
    print("="*60)
    print("✓ Training completed")

    # Guardar modelo final (CORREGIDO)
    print("\nSaving final model...")
    final_model_path = os.path.join(save_path, "best_model")  # ← FIX
    os.makedirs(final_model_path, exist_ok=True)
    
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Model saved to: {final_model_path}")
    
    # Evaluación final
    model.config.use_cache = True
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)
    
    print("\n" + "="*60)
    print("TRAINING FINISHED SUCCESSFULLY")
    print("="*60)
