#!/usr/bin/env python
import argparse
import time
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    # Cargar modelo y tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,  #quitar solo en postquant
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
    torch.cuda.reset_peak_memory_stats(device)  # resetea el pico [web:25][web:67]

    log_nvidia_smi(tag="[before_infer]")

    start = time.time()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # para que sea más determinista en la medida
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    peak_bytes = torch.cuda.max_memory_allocated(device)  # pico desde el reset [web:25][web:64]
    peak_gb = peak_bytes / (1024 ** 3)

    log_nvidia_smi(tag="[after_infer]")

    print(f"[METRICS] inference_time_sec={elapsed:.4f}")
    print(f"[METRICS] peak_vram_gb={peak_gb:.4f}")
    print("=" * 80)

    return elapsed, peak_gb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Ruta al modelo (carpeta HF: base, LoRA mergeada, quantizada, etc.)")
    parser.add_argument("--prompt", type=str, default="Kaixo, azaldu zer da LoRA euskaraz laburki.",
                        help="Prompt de prueba.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    measure_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
