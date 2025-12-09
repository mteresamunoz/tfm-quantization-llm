# INITIAL IMPORTS
from unsloth import FastLanguageModel

import numpy as np
import torch
import argparse
import wandb
import os
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from load_dataset import *



def train_formatting_function(data):
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
                formated_sen_chat.insert(0, {"role": "system", "content": "You are a helpful assistant who answers questions in Basque."})
                system_found = True
            formated_sen_chat.append({"role": role, "content": conv.get('value')})

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


#le paso el modelo cuantizado a 8 bit (latxa8b base ya cuantizado a 8bit) --> entreno ese modelo con lora y unsloth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = "9dfaae00b45401110e0e0024724781315433b031"
    wandb.init(project="lora-8b-8bit-unsloth", name="lora8b-8bit-unsloth")

    model_chk = args.model
    model_type = args.model_type
    lr = args.lr
    save_path = args.save_path
    dataset_path = args.dataset_path

    torch.manual_seed(42)
    np.random.seed(42)

    # HIPERPARÁMETROS
    bs = 32
    epochs = 5
    #512 max_seq?
    max_seq_length = 1024

    # CARGA SIMPLIFICADA CON UNSLOTH
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_chk,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False  # Puedes usar True si quieres usar 4bit
    )

    FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")

    # CONFIGURACIÓN LoRA PERSONALIZADA (opcional)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    print("Modelo y tokenizer cargados con Unsloth")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #tokenizer.add_eos_token = True

    # CARGA Y PREPARACIÓN DEL DATASET
    dataset = load_dataset(dataset_path)
    formated_train = dataset["train"].map(train_formatting_function, batched=True, num_proc=4)
    formated_dev = dataset["dev"].map(train_formatting_function, batched=True, num_proc=4)
    print("Dataset cargado")

    # CONFIGURACIÓN DE ENTRENAMIENTO
    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=save_path + "latxa8b_q8bit_instruct_unsloth",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        #save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=epochs,
        lr_scheduler_type="inverse_sqrt",
        warmup_ratio=0.1,
        bf16=True,
        #fp16=True, #cambiar a true para que sea mas rapido
        optim="paged_adamw_8bit",
        max_seq_length=max_seq_length,
        logging_steps=1,
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=formated_train,
        eval_dataset=formated_dev,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        packing=True, #cambiar a true si textos cortos
    )

    trainer.train()
    print("Entrenamiento completado")

    # GUARDAR MEJOR MODELO
    model.save_pretrained(args.save_path + "best_model")
    tokenizer.save_pretrained(args.save_path + "best_model")
    print(f"Mejor checkpoint: {trainer.state.best_model_checkpoint}")


    model.config.use_cache = True
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)
