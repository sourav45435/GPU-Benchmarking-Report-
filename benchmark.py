#!/usr/bin/env python3
"""
LLM Performance Benchmark Harness
Measures: tokens/s, peak RAM, peak VRAM, GPU utilization
Compatible with Gemma-2-9B / Gemma-2-27B / Gemma-3 variants and similar models
"""

import argparse
import csv
import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    print("[WARN] pynvml not available – GPU util will be approximate")


@dataclass
class BenchmarkResult:
    model_name: str
    prompt_tokens: int
    generated_tokens: int
    total_time_s: float
    tokens_per_second: float
    peak_ram_mb: float
    peak_vram_mb: float
    avg_gpu_util_pct: float
    max_gpu_util_pct: float
    device: str
    dtype: str
    quantization: str
    timestamp: str


class GPUMonitor:
    """Background thread that samples GPU utilization and VRAM."""
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.samples_util = []
        self.samples_vram = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not NVML_AVAILABLE:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples_util.append(util)
                self.samples_vram.append(mem.used / (1024 ** 2))  # MB
            except Exception:
                pass
            time.sleep(self.interval)

    def stats(self) -> Dict[str, float]:
        if not self.samples_util:
            return {"avg_util": 0.0, "max_util": 0.0, "peak_vram_mb": 0.0}
        return {
            "avg_util": sum(self.samples_util) / len(self.samples_util),
            "max_util": max(self.samples_util),
            "peak_vram_mb": max(self.samples_vram) if self.samples_vram else 0.0,
        }


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def load_model(model_id: str, quant: str = "none", dtype: str = "bfloat16"):
    """Load model with optional 4-bit / 8-bit quantization."""
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    quant_config = None
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"[INFO] Loading {model_id} (quant={quant}, dtype={dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch_dtype if quant == "none" else None,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Single generation pass with timing and resource tracking."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Warm-up (important for accurate timing)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    monitor = GPUMonitor(interval=0.15)
    monitor.start()
    ram_before = get_ram_mb()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    monitor.stop()
    gpu_stats = monitor.stats()

    generated_len = outputs.shape[1] - prompt_len
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Prefer NVML peak if available (more accurate under continuous load)
    if gpu_stats["peak_vram_mb"] > peak_vram:
        peak_vram = gpu_stats["peak_vram_mb"]

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": generated_len,
        "total_time_s": elapsed,
        "tokens_per_second": generated_len / elapsed if elapsed > 0 else 0.0,
        "peak_ram_mb": max(get_ram_mb(), ram_before),
        "peak_vram_mb": peak_vram,
        "avg_gpu_util_pct": gpu_stats["avg_util"],
        "max_gpu_util_pct": gpu_stats["max_util"],
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Performance Benchmark")
    parser.add_argument("--model", default="google/gemma-2-9b-it", help="HF model ID")
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--prompt", default="Explain the difference between supervised and unsupervised learning in two paragraphs.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, quant=args.quant, dtype=args.dtype)

    results: List[BenchmarkResult] = []
    for i in range(args.runs):
        print(f"\n=== Run {i+1}/{args.runs} ===")
        metrics = run_inference(
            model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens
        )
        result = BenchmarkResult(
            model_name=args.model,
            **metrics,
            device=str(model.device),
            dtype=args.dtype,
            quantization=args.quant,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        results.append(result)
        print(f"  Tokens/s : {result.tokens_per_second:.2f}")
        print(f"  Peak VRAM: {result.peak_vram_mb:.0f} MB")
        print(f"  Peak RAM : {result.peak_ram_mb:.0f} MB")
        print(f"  GPU util : avg {result.avg_gpu_util_pct:.1f}%  max {result.max_gpu_util_pct:.1f}%")

    # Save JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Human-readable summary
    avg_tps = sum(r.tokens_per_second for r in results) / len(results)
    avg_vram = sum(r.peak_vram_mb for r in results) / len(results)
    avg_ram = sum(r.peak_ram_mb for r in results) / len(results)
    avg_util = sum(r.avg_gpu_util_pct for r in results) / len(results)

    summary = f"""
# Benchmark Summary
Model          : {args.model}
Quantization   : {args.quant}
Dtype          : {args.dtype}
Runs           : {args.runs}
Prompt tokens  : ~{results[0].prompt_tokens}
Max new tokens : {args.max_new_tokens}

Average tokens/s     : {avg_tps:.2f}
Average peak VRAM    : {avg_vram:.0f} MB
Average peak RAM     : {avg_ram:.0f} MB
Average GPU util     : {avg_util:.1f}%

Raw data: {json_path}
CSV     : {csv_path}
"""
    print(summary)
    (out_dir / "SUMMARY.md").write_text(summary)

    print(f"\n[DONE] Results written to {out_dir}/")


if __name__ == "__main__":
    main()
