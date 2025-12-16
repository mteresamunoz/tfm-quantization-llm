import torch
import json
import random
import yaml
from datasets import Dataset
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
#from llmcompressor.transformers.compression.helpers import (
#    calculate_offload_device_map,
#    custom_offload_device_map,
#)

# Receta de cuantización FP8
#en la carpeta de yaml

# Ruta a tu modelo
model_path = "/gaueko1/users/mmartin/qloraTrain/merge/models/Latxa3.1_8b_fusionado"
output_dir = "/gaueko1/users/mmartin/tfm-quantization-llm/models/PostQuant/FP8/Latxa3.1_8b_lr1e-5--LoRaMIO-FP8-calibration-dynamic-asym"

# Parámetros de calibración
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

print("=" * 60)
print("CUANTIZACIÓN FP8 - Latxa3.1_8b loRA MIO - CALIBRACIÓN DINÁMICA ASIMÉTRICA")
print("=" * 60)

# ============================================
# 1. CARGAR DATOS DE CALIBRACIÓN YA PROCESADOS
# ============================================
print("\n[1/5] Cargando datos de calibración preprocesados...")
calibration_path = '/gaueko1/users/mmartin/ptq_exp/calibration_data/calibration_512.json'

with open(calibration_path, 'r', encoding='utf-8') as f:
    calibration_texts = json.load(f)

print(f"  ✓ Ejemplos cargados: {len(calibration_texts)}")

# Estadísticas
lengths = [len(s) for s in calibration_texts]
print(f"    - Longitud promedio: {sum(lengths) / len(lengths):.0f} caracteres")
print(f"    - Longitud mínima: {min(lengths)} caracteres")
print(f"    - Longitud máxima: {max(lengths)} caracteres")

# ============================================
# 2. CARGAR MODELO Y TOKENIZER
# ============================================
print(f"\n[2/5] Cargando modelo desde: {model_path}")

# Configuración del device map

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype="auto", 
    device_map="auto"
)
print("  ✓ Modelo cargado")

print(f"  Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("  ✓ Tokenizer cargado")

# ============================================
# 3. TOKENIZAR DATASET DE CALIBRACIÓN
# ============================================
print("\n[3/5] Tokenizando dataset de calibración...")

def tokenize_function(text):
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

# Tokeniza todos los textos
tokenized_data = []
for text in calibration_texts:
    tokens = tokenize_function(text)
    tokenized_data.append(tokens)

# Convierte a Dataset de HuggingFace
ds = Dataset.from_dict({
    'input_ids': [item['input_ids'] for item in tokenized_data],
    'attention_mask': [item['attention_mask'] for item in tokenized_data]
})

print(f"  ✓ Dataset tokenizado: {len(ds)} ejemplos")

# ============================================
# 4. CUANTIZACIÓN FP8
# ============================================
print("\n[4/5] Iniciando cuantización FP8...")
print(f"  Output: {output_dir}")

#recipe_yaml = yaml.safe_dump(recipe)


oneshot(
    model=model,
    output_dir=output_dir,
    dataset=ds,
    recipe="/gaueko1/users/mmartin/ptq_exp/yaml/fp8_calib_dynamic_asym_recipe.yaml",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True
    #save_recipe=True
)

# ============================================
# 5. FINALIZACIÓN
# ============================================
print("\n[5/5] ¡Cuantización completada!")
print("=" * 60)
print(f"✓ Modelo cuantizado guardado en:")
print(f"  {output_dir}")
print("=" * 60)

# Información adicional
print("\nPara usar el modelo cuantizado con vLLM:")
print(f"  vllm serve {output_dir}")
print("\nO cargarlo con transformers:")
print(f"  from transformers import AutoModelForCausalLM")
print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")