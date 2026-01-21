#!/usr/bin/env python3
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys

def medir_vram(model_id):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Para base: torch_dtype=torch.bfloat16
    # QLoRA: load_in_4bit=True (o 8bit), peft.from_pretrained si adapters
    # Post-quant: load_in_4bit=True si merged quant
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Ajusta: bfloat16 base, 4bit quant
        device_map="auto",  # O "cuda:0"
        low_cpu_mem_usage=True
    )
    
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"{model_id}: Peak VRAM {peak_gb:.2f} GB, Allocated {allocated_gb:.2f} GB")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return peak_gb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()
    medir_vram(args.model_id)
