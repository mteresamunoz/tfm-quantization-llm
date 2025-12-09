# INITIAL IMPORTS
import numpy as np
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback ,BitsAndBytesConfig
from trl import SFTTrainer #SFTConfig
import argparse
import wandb
from load_dataset import *
import os


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

    #inicializar wandb
    os.environ["WANDB_API_KEY"] = "9dfaae00b45401110e0e0024724781315433b031"
    wandb.init(project="qlora-8b-4bit", name="qlora-8b-4bit")

    model_chk = args.model
    model_type = args.model_type
    lr = args.lr
    save_path = args.save_path
    dataset_path = args.dataset_path


    #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)
    

    #HIPERPARAMETERS
    bs = 24 #32
    epochs = 5
    max_seq_length = 512 #1024
    wd = 0.01


    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True, #cambiar a 8bits
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )

    #model = AutoModelForCausalLM.from_pretrained(model_chk,quantization_config=bnb_config) #, device_map="auto")
    #checkpoint_path = "/gaueko1/users/mmartin/qloraTrain/qlora4bit/models/HiTZ/latxa-7b-v1.20.0005instruct4bits/checkpoint-43364"
    model = AutoModelForCausalLM.from_pretrained(
        model_chk,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    print("modelo cargado")

    tokenizer = AutoTokenizer.from_pretrained(model_chk, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    #tokenizer.add_eos_token = True
    #tokenizer.add_eos_token
    print("tokenizer cargado")

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
            )
    
    model = get_peft_model(model, peft_config)

    dataset = load_dataset(dataset_path)
    formated_train = dataset["train"].map(train_formatting_function, batched=False)
    formated_dev = dataset["dev"].map(train_formatting_function, batched=False)
    #formated_train = dataset["train"].map(train_formatting_function,batched=True, num_proc=4)
    #formated_dev = dataset["dev"].map(train_formatting_function, batched=True, num_proc=4)
    print("dataset cargado")

    #print("Example input:", formated_train[0]["text"][:500])
    #print("Num tokens:", len(tokenizer(formated_train[0]["text"])["input_ids"]))

    # TRAINING

        
    training_args = TrainingArguments(
            #dataset_text_field="text",
            output_dir=save_path + "latxa8b_q4bit_instruct",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=lr,
            weight_decay=wd,
            per_device_train_batch_size=4,   # menor tamaño para que quepa en memoria
            gradient_accumulation_steps=8,
            #per_device_train_batch_size=bs,
            per_device_eval_batch_size=4,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            num_train_epochs=epochs,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            bf16=True,
            #fp16=False,
            optim = "paged_adamw_8bit",
            logging_steps=1,
            report_to="wandb"
        )
    
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            peft_config=peft_config,
            train_dataset=formated_train,
            eval_dataset=formated_dev,
            dataset_text_field="text",
            dataset_num_proc=1,
            max_seq_length=max_seq_length,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
            packing=False,
        )

    trainer.train() #resume_from_checkpoint=checkpoint_path
    print("entrenado")

    #Guardar el mejor modelo al terminar
    trainer.save_model(args.save_path + "best_model")
    tokenizer.save_pretrained(args.save_path + "best_model")

    #para cargar despues el modelo:
    #model = AutoModelForCausalLM.from_pretrained("ruta/a/best_model", device_map="auto")
    #model = PeftModel.from_pretrained(model, "ruta/a/best_model")
    #tokenizer = AutoTokenizer.from_pretrained("ruta/a/best_model")

    model.config.use_cache = True
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)


#python scripts/train.py --dataset_path data/openhermes_2eu.json --model "HiTZ/latxa-7b-v1.2" --model_type "causal" --lr 0.0005 --save_path models/

#python /gaueko1/users/mmartin/qloraTrain/qlora4bit/scripts/train.py --dataset_path /gaueko1/users/mmartin/qloraTrain/qlora4bit/data/openhermes_2eu.json --model "HiTZ/latxa-7b-v1.2" --model_type "causal" --lr 0.0005 --save_path /gaueko1/users/mmartin/qloraTrain/qlora4bit/models/