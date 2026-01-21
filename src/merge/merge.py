from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ruta a tu modelo base
base_model_path = "google/gemma-2-9b"
# Ruta a los pesos LoRA que entrenaste
lora_model_path = "/gaueko1/users/mmartin/tfm-quantization-llm/models/LoRa/gemma-2-9b_lora_VRAM_tiempo/best_model"

# 1. Cargar modelo base (sin cuantización)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",  # o "cpu" si no tienes GPU
    torch_dtype="auto"
)

# 2. Cargar adaptadores LoRA encima del modelo base
model = PeftModel.from_pretrained(model, lora_model_path)

# 3. Fusionar los pesos LoRA dentro del modelo base
model = model.merge_and_unload()

# 4. Guardar el modelo ya fusionado (sin LoRA, listo para cuantizar o usar)
fusion_output_dir = "/gaueko1/users/mmartin/tfm-quantization-llm/models/merge/gemma"
model.save_pretrained(fusion_output_dir)

# (opcional) guardar también el tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(fusion_output_dir)
