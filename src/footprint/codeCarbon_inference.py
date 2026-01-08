#!/usr/bin/env python3
import os
import time
import gc
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker
from peft import PeftModel
from huggingface_hub import snapshot_download
# -----------------------
# Config general
# -----------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATHS = {
    #"base": "/proiektuak/ikergaitu-data/azabala106/model_evaluation/trained_models/Latxa3.1_8b_lr1e-5",
    #"q8": "/gaueko1/users/mmartin/ENVIRONMENT/models/Latxa3.1_8b_lr1e-5_quantized8bit",
    #"q4": "/gaueko1/users/mmartin/ENVIRONMENT/models/Latxa3.1_8b_lr1e-5_quantized4bit",
    #"nf4": "/gaueko1/users/mmartin/tfm-quantization-llm/models/quantization/NF4",
    #"qlora8": "/gaueko1/users/mmartin/qloraTrain/qlora-latxa-8b-8bit/models/bs24maxSeq512/best_model",
    #"qlora4": "/gaueko1/users/mmartin/qloraTrain/qlora-latxa-8b-4bit/models/bs24maxSeq512/best_model",
    #"qlora_nf4": "/gaueko1/users/mmartin/tfm-quantization-llm/models/qLoRa/NF4/best_model_llama_latxa8b",
    "postq8": "maytemuma/lora_quant8_latxa8b",
    "postq4": "maytemuma/lora_quant4_latxa8b",
    #"post_nf4": "/gaueko1/users/mmartin/tfm-quantization-llm/models/PostQuant/NF4",
    #"fp8": "/gaueko1/users/mmartin/ptq_exp/models/Latxa3.1_8b_lr1e-5-FP8-calibration-dynamic-asym",
    #"post_fp8": "/gaueko1/users/mmartin/tfm-quantization-llm/models/PostQuant/FP8/Latxa3.1_8b_lr1e-5--LoRaMIO-FP8-calibration-dynamic-asym",
    #"lora": "/gaueko1/users/mmartin/qloraTrain/lora/models/latxa8b_instruct/best_model",
}

PROMPT = "Kaixo, azaldu zer da LoRA euskaraz laburki."
MAX_NEW_TOKENS = 128
WARMUP_TOKENS = 16
N_REPS = 50

OUTPUT_DIR = "/gaueko1/users/mmartin/tfm-quantization-llm/footprint_metricas"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "inference_emissions.csv")
HF_CACHE_DIR = os.path.join(OUTPUT_DIR, "hf_models_cache")

SKIP_MODELS = set(x.strip() for x in os.environ.get("SKIP_MODELS", "").split(",") if x.strip())



def _load_peft_model(peft_path: str):
    # Detecta base_model desde adapter_config.json
    config_path = Path(peft_path) / "adapter_config.json"
    if not config_path.exists():
        raise ValueError("No adapter_config.json found")
    
    import json
    with open(config_path) as f:
        config = json.load(f)
    base_model = config.get("base_model_name_or_path", "meta-llama/Llama-2-7b-hf")  # Fallback
    
    print(f"  Loading PEFT: base={base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, peft_path, device_map={"": 0})
    return model



def _safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _read_codecarbon_row(cc_out_dir: str) -> Optional[Dict[str, Any]]:
    """Lee el último registro de CodeCarbon emissions.csv"""
    cc_csv = os.path.join(cc_out_dir, "emissions.csv")
    if not os.path.exists(cc_csv):
        return None
    df = pd.read_csv(cc_csv)
    if len(df) == 0:
        return None
    return df.iloc[-1].to_dict()


#def _load_model_try_gpu_first(model_path: str):
    """Intenta GPU0, fallback auto"""
    if not torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            dtype=torch.float32,
            trust_remote_code=True,
        )
        return model

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model

#def _load_model_try_gpu_first(model_path: str):
    if not torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.float32, trust_remote_code=True)

    # Intentos ordenados: GPU0 explícita, PEFT fallback, auto
    attempts = [
        # GPU0 sin quant auto
        {"device_map": {"": 0}, "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16, "trust_remote_code": True, "load_in_4bit": False, "load_in_8bit": False},
        # PEFT para QLoRA (base + adapter)
        lambda: _load_peft_model(model_path),
        # Auto con offload fp32
        {"device_map": "auto", "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16, "trust_remote_code": True, "llm_int8_enable_fp32_cpu_offload": True}
    ]

    for i, attempt in enumerate(attempts):
        try:
            if callable(attempt):
                model = attempt()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **attempt)
            print(f"  Loaded with attempt {i+1}")
            return model
        except Exception as e:
            print(f"  Attempt {i+1} failed: {e}")
            continue
    
    raise RuntimeError("All load attempts failed")

def _is_hf_repo_id(s: str) -> bool:
    # Heurística simple: "org/repo"
    return isinstance(s, str) and ("/" in s) and (not os.path.exists(s))


def _resolve_model_path(model_name: str, model_path_or_repo: str) -> str:
    """
    Si `model_path_or_repo` es un repo_id HF (org/repo) lo descarga a disco y devuelve el path local.
    Si es path local existente, lo devuelve tal cual.
    """
    if os.path.exists(model_path_or_repo):
        return model_path_or_repo

    if not _is_hf_repo_id(model_path_or_repo):
        raise FileNotFoundError(f"Path not found and not a HF repo_id: {model_path_or_repo}")

    _safe_mkdir(HF_CACHE_DIR)
    local_dir = os.path.join(HF_CACHE_DIR, model_name)

    # Descarga (si ya está, reusa cache)
    # allow_patterns: lo típico para modelos + tokenizers (ajústalo si tu repo tiene nombres raros)
    snapshot_download(
        repo_id=model_path_or_repo,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "tokenizer.*",
            "tokenizer/*",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "generation_config.json",
            "*.model",
            "*.txt",
        ],
    )
    return local_dir

def _load_model_single_gpu0(local_model_path: str):
    """
    Carga en una sola GPU (cuda:0) para que CodeCarbon tenga sentido con gpu_ids=[0].
    """
    if not torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(
            local_model_path,
            low_cpu_mem_usage=True,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    # Intentos: bf16 -> auto dtype. (evita device_map="auto" para no repartirse por varias GPUs)
    attempts = [
        {"device_map": {"": 0}, "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True},
        {"device_map": {"": 0}, "torch_dtype": "auto",        "low_cpu_mem_usage": True, "trust_remote_code": True},
    ]

    last_err = None
    for i, kwargs in enumerate(attempts, start=1):
        try:
            model = AutoModelForCausalLM.from_pretrained(local_model_path, **kwargs)
            print(f"  Loaded model (attempt {i})")
            return model
        except Exception as e:
            last_err = e
            print(f"  Attempt {i} failed: {str(e)[:140]}")

    raise RuntimeError(f"All load attempts failed: {last_err}")
    
#def _load_model_try_gpu_first(model_path: str):
    if not torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.float32, trust_remote_code=True)

    # Intentos ordenados para postq/qlora: sin quantizer, GPU0 first
    attempts = [
        # 1. GPU0 full, no device_map/quant
        {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True, "device_map": None},
        # 2. GPU0 explícito sin quant
        {"device_map": {"": 0}, "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True, "quantization_config": None},
        # 3. PEFT para qlora
        lambda: _load_peft_model(model_path),
        # 4. Auto con CPU offload
        {"device_map": "auto", "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True}
    ]

    for i, attempt in enumerate(attempts):
        try:
            if callable(attempt):
                model = attempt()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **attempt)
            if "device_map" in attempt and attempt["device_map"] is None:
                model = model.to("cuda:0")
            print(f"  Loaded with attempt {i+1}")
            return model
        except Exception as e:
            print(f"  Attempt {i+1} failed: {str(e)[:100]}")
            continue
    
    raise RuntimeError("All load attempts failed")

#def benchmark_model(model_name: str, model_path: str) -> Optional[Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"[{model_name}] Inference benchmark")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    if model_name in SKIP_MODELS:
        print(f"Skipped by SKIP_MODELS: {model_name}")
        return None

    if not os.path.exists(model_path):
        print(f"Path not found: {model_path}")
        return None

    cc_out_dir = os.path.join(OUTPUT_DIR, "codecarbon", model_name)
    _safe_mkdir(cc_out_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = None
    tokenizer = None
    inputs = None
    outputs = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = _load_model_try_gpu_first(model_path)
        model.eval()

        vram_model_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        print(f"  VRAM model loaded: {vram_model_gb:.2f} GB")

        inputs = tokenizer(PROMPT, return_tensors="pt", padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("Warmup...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS)

        vram_warmup_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        print(f"  VRAM after warmup: {vram_warmup_gb:.2f} GB")

        tracker = EmissionsTracker(
            project_name=f"footprint_inference_{model_name}",
            output_dir=cc_out_dir,
            save_to_file=True,
            log_level="error",
            measure_power_secs=2,
            tracking_mode="machine",
            gpu_ids=[0] if torch.cuda.is_available() else None,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tracker.start()
        total_time = 0.0
        total_tokens = 0

        print(f"Running {N_REPS} inferences...")
        for i in range(N_REPS):
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            dt = time.time() - t0
            total_time += dt
            total_tokens += MAX_NEW_TOKENS

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_REPS} ({dt*1000:.0f}ms/query)")

        _ = tracker.stop()

        vram_peak_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )

        cc_row = _read_codecarbon_row(cc_out_dir)
        if cc_row is None:
            raise RuntimeError(f"CodeCarbon emissions.csv not found in {cc_out_dir}")

        cc_emissions_kg_total = float(cc_row.get("emissions", 0.0))
        cc_energy_kwh_total = float(cc_row.get("energy_consumed", 0.0))
        cc_duration_s_total = float(cc_row.get("duration", total_time))

        time_per_query_ms = (total_time / N_REPS) * 1000.0
        energy_per_query_kwh = cc_energy_kwh_total / N_REPS if N_REPS > 0 else 0.0
        emissions_per_query_kg = cc_emissions_kg_total / N_REPS if N_REPS > 0 else 0.0
        emissions_per_query_g = emissions_per_query_kg * 1000.0
        emissions_per_query_ug = emissions_per_query_kg * 1e6

        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        tok_per_kwh = (total_tokens / cc_energy_kwh_total) if cc_energy_kwh_total > 0 else 0.0

        params_millions = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"\n✅ [{model_name}] RESULTS:")
        print(f"  Time/query:       {time_per_query_ms:.1f} ms")
        print(f"  Emissions/query:  {emissions_per_query_ug:.2f} μg CO2e")
        print(f"  Energy/query:     {energy_per_query_kwh:.6f} kWh")
        print(f"  Tokens/sec:       {tokens_per_sec:.1f}")
        print(f"  VRAM peak:        {vram_peak_gb:.2f} GB")
        print(f"  Tok/kWh:          {tok_per_kwh:.0f}")

        return {
            "model": model_name,
            "path": model_path,
            "latency_ms": round(time_per_query_ms, 2),
            "energy_per_query_kwh": energy_per_query_kwh,
            "emissions_per_query_g": emissions_per_query_g,
            "emissions_per_query_ug": round(emissions_per_query_ug, 2),
            "vram_peak_gb": round(vram_peak_gb, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "tok_per_kwh": round(tok_per_kwh, 0),
            "vram_model_gb": round(vram_model_gb, 2),
            "vram_warmup_gb": round(vram_warmup_gb, 2),
            "params_millions": round(params_millions, 1),
            "n_reps": N_REPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "cc_energy_kwh_total": cc_energy_kwh_total,
            "cc_emissions_kg_total": cc_emissions_kg_total,
            "cc_duration_s_total": cc_duration_s_total,
            "cc_out_dir": cc_out_dir,
        }

    except Exception as e:
        print(f"[{model_name}] ERROR: {e}")
        print(traceback.format_exc())
        return None

    finally:
        try:
            del outputs
        except:
            pass
        try:
            del inputs
        except:
            pass
        try:
            del model
        except:
            pass
        try:
            del tokenizer
        except:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  Memory cleaned\n")

def benchmark_model(model_name: str, model_path_or_repo: str) -> Optional[Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"[{model_name}] Inference benchmark")
    print(f"Path/Repo: {model_path_or_repo}")
    print(f"{'='*60}")

    cc_out_dir = os.path.join(OUTPUT_DIR, "codecarbon", model_name)
    _safe_mkdir(cc_out_dir)

    model = None
    tokenizer = None
    inputs = None
    outputs = None

    try:
        local_path = _resolve_model_path(model_name, model_path_or_repo)
        print(f"  Local path: {local_path}")

        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = _load_model_single_gpu0(local_path)
        model.eval()

        # Usa el device real del modelo (evita cpu/cuda mismatch)
        device = next(model.parameters()).device

        vram_model_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
        print(f"  VRAM model loaded: {vram_model_gb:.2f} GB")

        inputs = tokenizer(PROMPT, return_tensors="pt", padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("Warmup...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)

        vram_warmup_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
        print(f"  VRAM after warmup: {vram_warmup_gb:.2f} GB")

        tracker = EmissionsTracker(
            project_name=f"footprint_inference_{model_name}",
            output_dir=cc_out_dir,
            save_to_file=True,
            log_level="error",
            measure_power_secs=2,
            tracking_mode="machine",
            gpu_ids=[0] if torch.cuda.is_available() else None,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tracker.start()
        total_time = 0.0
        total_tokens = 0

        print(f"Running {N_REPS} inferences...")
        for i in range(N_REPS):
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            dt = time.time() - t0
            total_time += dt
            total_tokens += MAX_NEW_TOKENS

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_REPS} ({dt*1000:.0f}ms/query)")

        _ = tracker.stop()

        vram_peak_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0

        cc_row = _read_codecarbon_row(cc_out_dir)
        if cc_row is None:
            raise RuntimeError(f"CodeCarbon emissions.csv not found in {cc_out_dir}")

        cc_emissions_kg_total = float(cc_row.get("emissions", 0.0))
        cc_energy_kwh_total = float(cc_row.get("energy_consumed", 0.0))
        cc_duration_s_total = float(cc_row.get("duration", total_time))

        time_per_query_ms = (total_time / N_REPS) * 1000.0
        energy_per_query_kwh = cc_energy_kwh_total / N_REPS if N_REPS > 0 else 0.0
        emissions_per_query_kg = cc_emissions_kg_total / N_REPS if N_REPS > 0 else 0.0

        emissions_per_query_g = emissions_per_query_kg * 1000.0
        emissions_per_query_ug = emissions_per_query_kg * 1e6

        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        tok_per_kwh = (total_tokens / cc_energy_kwh_total) if cc_energy_kwh_total > 0 else 0.0

        params_millions = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"\n✅ [{model_name}] RESULTS:")
        print(f"  Time/query:       {time_per_query_ms:.1f} ms")
        print(f"  Emissions/query:  {emissions_per_query_ug:.2f} μg CO2e")
        print(f"  Energy/query:     {energy_per_query_kwh:.6f} kWh")
        print(f"  Tokens/sec:       {tokens_per_sec:.1f}")
        print(f"  VRAM peak:        {vram_peak_gb:.2f} GB")
        print(f"  Tok/kWh:          {tok_per_kwh:.0f}")

        return {
            "model": model_name,
            "path": model_path_or_repo,
            "latency_ms": round(time_per_query_ms, 2),
            "energy_per_query_kwh": energy_per_query_kwh,
            "emissions_per_query_g": emissions_per_query_g,
            "emissions_per_query_ug": round(emissions_per_query_ug, 2),
            "vram_peak_gb": round(vram_peak_gb, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "tok_per_kwh": round(tok_per_kwh, 0),
            "vram_model_gb": round(vram_model_gb, 2),
            "vram_warmup_gb": round(vram_warmup_gb, 2),
            "params_millions": round(params_millions, 1),
            "n_reps": N_REPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "cc_energy_kwh_total": cc_energy_kwh_total,
            "cc_emissions_kg_total": cc_emissions_kg_total,
            "cc_duration_s_total": cc_duration_s_total,
            "cc_out_dir": cc_out_dir,
        }

    except Exception as e:
        print(f"[{model_name}] ERROR: {e}")
        print(traceback.format_exc())
        return None

    finally:
        for obj in [outputs, inputs, model, tokenizer]:
            try:
                del obj
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("  Memory cleaned\n")

def main():
    _safe_mkdir(OUTPUT_DIR)

    print("INFERENCE BENCHMARKING START")
    print(f"Models: {len(MODEL_PATHS)} | Reps/model: {N_REPS}")
    print(f"TFM table: {RESULTS_CSV}")
    print(f"CodeCarbon logs: {os.path.join(OUTPUT_DIR, 'codecarbon')}\n")

    results = []

    for model_name, model_path in MODEL_PATHS.items():
        r = benchmark_model(model_name, model_path)
        if r is not None:
            results.append(r)
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"Partial save ({len(results)} models): {RESULTS_CSV}\n")

    if len(results) == 0:
        print("No results (all failed/skipped).")
        return

    df_final = pd.DataFrame(results)
    df_final.to_csv(RESULTS_CSV, index=False)

    print(f"\nFINAL TABLE: {RESULTS_CSV}")
    print(df_final[["model", "latency_ms", "emissions_per_query_g", "vram_peak_gb"]].round(4))


if __name__ == "__main__":
    main()
