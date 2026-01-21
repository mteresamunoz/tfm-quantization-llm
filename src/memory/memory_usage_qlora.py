#!/usr/bin/env python3
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse

def limpiar_gpu():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.synchronize()

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_id", required=True)
parser.add_argument("--base_id", default="HiTZ/Latxa-Llama-3.1-8B")
parser.add_argument("--quant_type", choices=["fp4", "int8", "nf4_dq"], required=True)
args = parser.parse_args()

limpiar_gpu()
tokenizer = AutoTokenizer.from_pretrained(args.base_id)

if args.quant_type == "fp4":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",  # FP4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )
elif args.quant_type == "int8":
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)  # INT8
elif args.quant_type == "nf4_dq":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # DQ
    )

base = AutoModelForCausalLM.from_pretrained(
    args.base_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base, args.adapter_id)

peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
alloc_gb = torch.cuda.memory_allocated() / (1024**3)
print(f"**QLoRA-{args.quant_type.upper()}** {args.adapter_id}: Peak VRAM {peak_gb:.2f} GB, Allocated {alloc_gb:.2f} GB")

del model, base, tokenizer
limpiar_gpu()
