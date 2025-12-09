import json
import random

print("Cargando dataset...")

# Carga JSONL (una línea = un objeto JSON)
data = []
with open('/gaueko1/users/mmartin/lora_qlora_exp/data/openhermes_2eu.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"Total ejemplos en dataset: {len(data)}")

# Extrae 512 ejemplos aleatorios
random.seed(42)
selected_indices = random.sample(range(len(data)), min(512, len(data)))

calibration_samples = []
for idx in selected_indices:
    example = data[idx]
    
    # Formato según tu preprocessing
    formated_sen = ""
    system_found = False
    
    if 'conversations' in example and example['conversations']:
        for conv in example['conversations']:
            role = conv.get('from', '')
            content = conv.get('value', '')
            
            if role == 'system':
                formated_sen += f"System Message: {content}\n"
                system_found = True
            elif role == 'human':
                formated_sen += f"User: {content}\n"
            elif role == 'gpt':
                formated_sen += f"Assistant: {content}\n"
    
    # FUERA del loop: Si no había system, añádelo AL INICIO
    if not system_found and formated_sen.strip():
        formated_sen = "System Message: You are a helpful assistant who answers questions in Basque.\n" + formated_sen
    
    # Solo añade si tiene contenido
    if formated_sen.strip():
        calibration_samples.append(formated_sen)

print(f"Ejemplos procesados: {len(calibration_samples)}")

# Guarda
output_path = '/gaueko1/users/mmartin/ptq_exp/calibration_data/calibration_512.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(calibration_samples, f, ensure_ascii=False, indent=2)

print(f"✓ Guardados {len(calibration_samples)} ejemplos en {output_path}")

# Estadísticas
lengths = [len(s) for s in calibration_samples]
print(f"  Longitud promedio: {sum(lengths) / len(lengths):.0f} caracteres")
print(f"  Longitud mínima: {min(lengths)} caracteres")
print(f"  Longitud máxima: {max(lengths)} caracteres")

# Muestra ejemplo
print(f"\nEjemplo del primer sample:")
print(calibration_samples[0][:300])