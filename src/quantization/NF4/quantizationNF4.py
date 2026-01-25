from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import os
import torch
import safetensors.torch 

model = 'MODEL_TO_QUANTIZE'  

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model) #, use_fast=False) #--> si da bool, quitar use_fast
print(tokenizer)

if tokenizer is None:
    print("Error: El tokenizador no se cargó correctamente.")
else:
    print("Tokenizador cargado correctamente.")

#cuant 8bit
print("CARGANDO MODELO NF4")
# Latxa-8B en NF4 solo → 6GB VRAM
model = AutoModelForCausalLM.from_pretrained(
    model,  # Base model
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

path_model = "MODEL_NF4_OUTPUT_DIRECTORY"
#crea carpeta si no existe
os.makedirs(path_model, exist_ok=True)

#para gemma se guarda todo el modelo, no state_dict
#state_dict_8bit = model.state_dict()
safetensors.torch.save_model(model, os.path.join(path_model, "model.safetensors"))

model.save_pretrained(path_model, safe_serialization=True)

model.config.save_pretrained(path_model)

tokenizer.save_pretrained(path_model)

print("MODELO NF4 GUARDADO EN: ", os.listdir(path_model))
