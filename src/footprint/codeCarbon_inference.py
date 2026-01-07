#!/usr/bin/env python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="Kaixo, azaldu zer da LoRA euskaraz laburki.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device

    print("=" * 60)
    print("Inference footprint measurement")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    # Prepare input
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup (opcional, para estabilizar consumo)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8)

    # Footprint measurement
    tracker = EmissionsTracker(
        project_name="footprint_qloranf4",
        output_dir="/gaueko1/users/mmartin/tfm-quantization-llm/footprint_metricas",
        save_to_file=True,
        log_level="error",
        measure_power_secs=1,
    )

    tracker.start()
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    end = time.time()
    emissions_kg = tracker.stop()

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    elapsed = end - start

    print("\n=== RESULTS ===")
    print(f"[METRICS] inference_time_sec={elapsed:.3f}")
    print(f"[METRICS] emissions_kg={emissions_kg:.6f}")
    print(f"[METRICS] emissions_g_per_query={emissions_kg*1000:.3f}")
    print("\nGenerated output (truncated):")
    print(generated[:300])


if __name__ == "__main__":
    main()
