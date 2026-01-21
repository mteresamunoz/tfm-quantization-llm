#!/usr/bin/env python3
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

def limpiar_gpu():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.synchronize()

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", required=True)  # Tu repo merged+quant
parser.add_argument("--quant_type", choices=["fp4", "int8", "nf4_dq"], required=True)
args = parser.parse_args()

limpiar_gpu()
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

if args.quant_type == "fp4":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )
elif args.quant_type == "int8":
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
elif args.quant_type == "nf4_dq":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
alloc_gb = torch.cuda.memory_allocated() / (1024**3)
print(f"**Postquant-{args.quant_type.upper()}** {args.model_id}: Peak VRAM {peak_gb:.2f} GB, Allocated {alloc_gb:.2f} GB")

del model, tokenizer
limpiar_gpu()
