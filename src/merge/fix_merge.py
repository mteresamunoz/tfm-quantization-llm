# GUARDAR COMO fix_gemma_shards.py y correr: python fix_gemma_shards.py
from safetensors import safe_open
from safetensors.torch import save_file
import glob, os

path = "/gaueko1/users/mmartin/tfm-quantization-llm/models/merge/gemma"
shards = sorted(glob.glob(os.path.join(path, "model-*.safetensors")))

for shard_path in shards:
    tensors = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    save_file(tensors, shard_path, metadata={"format": "pt"})  # Fix in-place
    print(f"✅ Fixed: {os.path.basename(shard_path)}")

print("✅ Merge listo para lm-eval!")
