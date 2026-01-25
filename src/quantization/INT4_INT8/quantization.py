#pip install "torch>=2.0.0" bitsandbytes --upgrade
#pip install "transformers[accelerate]>=4.43.0" --upgrade

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import safetensors.torch #pip install safetensors

model = 'MODEL_TO_QUANTIZE'  # Ejemplo: "path/to/your/fused/model"

#config cuant 8-bit
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,  #Cuant 4bit (cambiar a load_in_8bit para 8bit)
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
    #load_in_8bit_fp32_cpu_offload=True, #mueve algunas capas a CPU (por si no cabe en la GPU(xirimiri))
)

#config para quant en 4bit
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype=torch.float16,
)

#tokenizador
tokenizer = AutoTokenizer.from_pretrained(model) #, use_fast=False) #--> si da bool, quitar use_fast
print(tokenizer)

if tokenizer is None:
    print("Error: El tokenizador no se cargó correctamente.")
else:
    print("Tokenizador cargado correctamente.")

#cuant 8bit
print("CARGANDO MODELO 8BIT")
modelq8 = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config_8bit,
    device_map="auto",
    trust_remote_code=True
)

#modelo cuant 4-bit
print("CARGANDO MODELO 4BIT")
modelq4 = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config_4bit,
    device_map="auto",  #mapea automáticamente a GPU si está disponible
    trust_remote_code=True
    #offload_folder="offload",  # Mueve partes a disco
)


path_modelq8bit = "MODEL_Q8BIT_OUTPUT_DIRECTORY"  
#crea carpeta si no existe
os.makedirs(path_modelq8bit, exist_ok=True)
#guardar modelo con safetensors
#guarda pesos cuantizados y parametros entrenados pytorch
#state_dict_8bit = modelq8.state_dict()
safetensors.torch.save_model(modelq8, os.path.join(path_modelq8bit, "model.safetensors"))
#guardar config y tokenizador
#modelq8.config.save_pretrained(path_modelq8bit)
modelq8.save_pretrained(path_modelq8bit, safe_serialization=True)
modelq8.config.save_pretrained(path_modelq8bit)
#print(tokenizer)
tokenizer.save_pretrained(path_modelq8bit)
#tokenizer.config.save_pretrained(path_modelq8bit)
print("MODELO 8BIT GUARDADO EN: ", os.listdir(path_modelq8bit))



path_modelq4bit = "MODEL_Q4BIT_OUTPUT_DIRECTORY"
os.makedirs(path_modelq4bit, exist_ok=True)
#guardar modelo con safetensors
#guarda pesos cuantizados y parametros entrenados pytorch
#state_dict_4bit = modelq4.state_dict()
safetensors.torch.save_model(modelq4, os.path.join(path_modelq4bit, "model.safetensors"))
#guardar config y tokenizador
modelq4.save_pretrained(path_modelq4bit, safe_serialization=True)
modelq4.config.save_pretrained(path_modelq4bit)
tokenizer.save_pretrained(path_modelq4bit)
#tokenizer.config.save_pretrained(path_modelq4bit)

print("MODELO 4BIT GUARDADO EN: ", os.listdir(path_modelq4bit))

print("TERMINADO")
