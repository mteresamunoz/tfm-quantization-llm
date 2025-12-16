#!/usr/bin/env python
import argparse
import time
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

# ============================
# PATCH: evitar normal_ en pesos cuantizados (8bit)
# ============================

print(">>> Antes del patch, _init_weights =", LlamaPreTrainedModel._init_weights)

def patched_init_weights(self, module):
    """
    Versión parcheada de _init_weights para evitar llamar a normal_()
    sobre pesos cuantizados (Char / int8 / uint8).
    También replica la lógica del ejemplo del hilo: caso rank_head.
    """
    # Si el módulo tiene pesos y son cuantizados, no tocar
    if hasattr(module, "weight") and module.weight is not None:
        if module.weight.dtype in (
            torch.int8, torch.uint8,
            torch.int16, torch.int32, torch.int64
        ):
            # Saltamos la inicialización para evitar normal_kernel_cpu sobre Char
            return

    # Caso especial como en el hilo: rank_head
    if hasattr(self, "rank_head") and module is getattr(self, "rank_head", None):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(torch.float32)
        return

    # Para cualquier otro módulo, NO hacer nada (no llamamos a normal_)
    return

LlamaPreTrainedModel._init_weights = patched_init_weights
print(">>> Después del patch, _init_weights =", LlamaPreTrainedModel._init_weights)


# ============================
# Utilidades
# ============================

def log_nvidia_smi(tag: str = ""):
    """
    Imprime memoria usada/libre según nvidia-smi para dejarlo en el .out.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).decode("utf-8").strip()
        print(f"[nvidia-smi]{tag} {out}")
    except Exception as e:
        print(f"[nvidia-smi]{tag} ERROR: {e}")


def measure_inference(model_path: str,
                      prompt: str,
                      max_new_tokens: int = 128,
                      batch_size: int = 1):
    device = torch.device("cuda:0")

    print("=" * 80)
    print(f"MODEL: {model_path}")
    print(f"PROMPT (truncated): {prompt[:80]}...")
    print(f"max_new_tokens={max_new_tokens}, batch_size={batch_size}")
    print("=" * 80)

    # Cargar modelo y tokenizer (modelo YA cuantizado a 8bit en disco)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,   # para que todo lo que se pueda sea fp16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Construir batch
    texts = [prompt] * batch_size
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Resetear stats de memoria y medir
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    log_nvidia_smi(tag="[before_infer]")

    start = time.time()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # determinista para la medida
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gb = peak_bytes / (1024 ** 3)

    log_nvidia_smi(tag="[after_infer]")

    print(f"[METRICS] inference_time_sec={elapsed:.4f}")
    print(f"[METRICS] peak_vram_gb={peak_gb:.4f}")
    print("=" * 80)

    return elapsed, peak_gb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo (carpeta HF: base, LoRA mergeada, quantizada, etc.)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Kaixo, azaldu zer da LoRA euskaraz laburki.",
        help="Prompt de prueba.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    measure_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
