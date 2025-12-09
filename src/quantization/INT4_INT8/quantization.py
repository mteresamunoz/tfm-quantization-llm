#pip install "torch>=2.0.0" bitsandbytes --upgrade
#pip install "transformers[accelerate]>=4.43.0" --upgrade

import os
import torch
#import triton.ops --> cambiar a triton
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import safetensors.torch #pip install safetensors

#try:
#    import triton
#    print("Triton importado correctamente.")
#except ImportError as e:
#    print(f"Error al importar Triton: {e}")
#model = "HiTZ/latxa-7b-v1.2" #1.2 --> 28GB
#model = '/gaueko1/users/mmartin/ENVIRONMENT/latxa70b/Latxa-Llama-3.1-70B-Instruct-exp_2_101' #--> 263G 
model = '/gaueko1/users/mmartin/qloraTrain/merge/models/Latxa3.1_8b_fusionado'

#config cuant 8-bit
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,  #Cuant 4bit (cambiar a load_in_8bit para 8bit)
    llm_int8_threshold=6.0,
    #load_in_8bit_fp32_cpu_offload=True, #mueve algunas capas a CPU (por si no cabe en la GPU(xirimiri))
)

#config para quant en 4bit
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit = True,
    #llm_int4_threshold=6.0,
    bnb_4bit_use_double_quant=True,
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


path_modelq8bit = "/gaueko1/users/mmartin/ENVIRONMENT/models/Latxa_8b_merge_quantized8bit"
#crea carpeta si no existe
os.makedirs(path_modelq8bit, exist_ok=True)
#guardar modelo con safetensors
#guarda pesos cuantizados y parametros entrenados pytorch
state_dict_8bit = modelq8.state_dict()
safetensors.torch.save_file(state_dict_8bit, os.path.join(path_modelq8bit, "model.safetensors"))
#guardar config y tokenizador
modelq8.config.save_pretrained(path_modelq8bit)
#modelq.save_pretrained(path)
#print(tokenizer)
tokenizer.save_pretrained(path_modelq8bit)
#tokenizer.config.save_pretrained(path_modelq8bit)
print("MODELO 8BIT GUARDADO EN: ", os.listdir(path_modelq8bit))

#du -sh /gaueko1/users/mmartin/maite_env/models/latxa-7b-v1.2-quantized8bit --> 6.6GB
#du -sh /gaueko1/users/mmartin/maite_env/models/latxa-7b-v1.2-quantized4bit --> 3.9G  
#print("Contenido de la carpeta del modelo:", os.listdir(path))
#du -sh /gaueko1/users/mmartin/maite_env/models/latxa-70b-quantized8bit --> 68G
#du -sh /gaueko1/users/mmartin/maite_env/models/latxa-70b-quantized4bit --> 37G



path_modelq4bit = "/gaueko1/users/mmartin/ENVIRONMENT/models/Latxa_8b_merge_quantized4bit"
os.makedirs(path_modelq4bit, exist_ok=True)
#guardar modelo con safetensors
#guarda pesos cuantizados y parametros entrenados pytorch
state_dict_4bit = modelq4.state_dict()
safetensors.torch.save_file(state_dict_4bit, os.path.join(path_modelq4bit, "model.safetensors"))
#guardar config y tokenizador
modelq4.config.save_pretrained(path_modelq4bit)
#modelq.save_pretrained(path)
tokenizer.save_pretrained(path_modelq4bit)
#tokenizer.config.save_pretrained(path_modelq4bit)

print("MODELO 4BIT GUARDADO EN: ", os.listdir(path_modelq4bit))

print("TERMINADO")
#du -sh /gaueko1/users/mmartin/maite_env/models/latxa-7b-v1.2-quantized8bit --> 6.6GB
#du -sh /gaueko1/users/mmartin/maite_env/models/latxa-7b-v1.2-quantized4bit --> 3.9G  
#print("Contenido de la carpeta del modelo:", os.listdir(path))

#verificar en que dispositivo esta el modelo
#print(modelq.hf_device_map)

#probar
#text = "Kimika analitikoan, zein da analisi kuantitatiboan barne estandarra erabiltzearen atzean dagoen printzipioa?\\nA. Lagin prestaketan eta tresnaren erantzunean dauden aldaerak konpentsatzen ditu.\\nB. Metodo analitikoaren sentikortasuna hobetzen du.\\nC. Metodo analitikoaren detekzio-muga murrizten du.\\nD. Kromatografian analitikoen arteko bereizmena handitzen du.\\nE. Goian aipatutakoetatik bat ere ez."
#inputs = tokenizer(text, return_tensors="pt").to("cuda")  # Asegurar que los tensores están en GPU

#with torch.no_grad():
#    output = modelq.generate(**inputs, max_new_tokens=100)

#print(tokenizer.decode(output[0], skip_special_tokens=True))

#Kimika analitikoan, zein da analisi kuantitatiboan barne estandarra erabiltzearen atzean dagoen printzipioa?\nA. Lagin prestaketan eta tresnaren erantzunean dauden aldaerak konpentsatzen ditu.\nB. Metodo analitikoaren sentikortasuna hobetzen du.\nC. Metodo analitikoaren detekzio-muga murrizten du.\nD. Kromatografian analitikoen arteko bereizmena handitzen du.\nE. Goian aipatutakoetatik bat ere ez.

#A. Lagin prestaketan eta tresnaren erantzunean dauden aldaerak konpentsatzen ditu.

#Zein da zehaztasunaren eta errorearen arteko desberdintasuna?

#Zein da zehaztasunaren eta errorearen arteko desberdintasuna?

#A. Zehaztasuna analisi bate