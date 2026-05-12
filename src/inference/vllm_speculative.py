"""
vLLM-based speculative decoding pipeline with acceptance rate logging.

Why vLLM alongside the custom loop?
  - The proposal explicitly requires a vLLM-based pipeline.
  - vLLM handles batching, KV-cache management, and continuous batching
    automatically — giving more realistic throughput numbers than our
    single-sequence custom loop.
  - vLLM internally tracks SpecDecodeWorkerMetrics (acceptance rate,
    system efficiency) which we extract after each generation.

Two modes:
  1. vllm_generate()     — simple batch generation, returns texts + timing
  2. vllm_benchmark()    — runs on a list of prompts, aggregates all metrics

Acceptance rate is extracted from vLLM's engine stats via
`_run_engine()` internals or the AsyncLLMEngine stats callback.

Usage:
    python src/inference/vllm_speculative.py
    python src/inference/vllm_speculative.py --condition domain_tuned --adapter ./path/to/adapter
"""

import argparse
import json
import os
import time
import sys

sys.path.append(os.path.dirname(__file__))
from metrics import DecodingMetrics, save_metrics, print_summary

try:
    from vllm import LLM, SamplingParams
    from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("WARNING: vLLM not installed. Run: pip install vllm")


TARGET_MODEL = "codellama/CodeLlama-7b-hf"
DRAFT_MODEL  = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


# ── Engine setup ──────────────────────────────────────────────────────────────

def build_vllm_engine(
    target_model: str = TARGET_MODEL,
    draft_model: str = DRAFT_MODEL,
    lora_adapter: str = None,
    gamma: int = 5,
    gpu_memory_utilization: float = 0.90,
) -> "LLM":
    """
    Build a vLLM LLM engine with speculative decoding enabled.

    Args:
        target_model:           HF model ID for the target (verifier) model
        draft_model:            HF model ID for the draft model
        lora_adapter:           path to LoRA adapter for domain-tuned condition
                                (merged into draft model before loading)
        gamma:                  number of speculative tokens (lookahead)
        gpu_memory_utilization: fraction of GPU memory for vLLM's KV-cache
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is not installed. Run: pip install vllm>=0.4.0")

    # If a LoRA adapter is provided, merge it into the base draft model first
    # and save to a temp path — vLLM loads full models, not PEFT adapters
    actual_draft_model = draft_model
    if lora_adapter:
        actual_draft_model = _merge_lora_for_vllm(draft_model, lora_adapter)

    print(f"Building vLLM engine...")
    print(f"  Target : {target_model}")
    print(f"  Draft  : {actual_draft_model} (gamma={gamma})")

    llm = LLM(
        model=target_model,
        speculative_model=actual_draft_model,
        num_speculative_tokens=gamma,
        gpu_memory_utilization=gpu_memory_utilization,
        # quantization="bitsandbytes",  # uncomment if GPU memory is tight
        dtype="float16",
        trust_remote_code=True,
        disable_log_stats=False,    # keep stats enabled to extract acceptance rate
    )
    return llm


def _merge_lora_for_vllm(base_model_id: str, lora_adapter: str,
                           save_dir: str = "/tmp/vllm_merged_draft") -> str:
    """
    Merge a LoRA adapter into the base model and save as a full model.
    vLLM requires a full model path for the speculative draft, not a PEFT adapter.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if os.path.exists(save_dir):
        print(f"  Merged draft already exists at {save_dir}, skipping merge.")
        return save_dir

    print(f"  Merging LoRA adapter into base draft model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, lora_adapter)
    merged = model.merge_and_unload()   # merge LoRA weights into base

    os.makedirs(save_dir, exist_ok=True)
    merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Merged model saved to {save_dir}")
    return save_dir


# ── Metrics extraction ────────────────────────────────────────────────────────

def extract_spec_decode_metrics(llm: "LLM") -> dict:
    """
    Extract speculative decoding metrics from vLLM's internal stats.

    vLLM tracks these in SpecDecodeWorkerMetrics:
      - draft_acceptance_rate  : fraction of draft tokens accepted
      - system_efficiency      : fraction of time not wasted on rejections
      - accepted_tokens        : total accepted draft tokens
      - draft_tokens           : total draft tokens proposed
      - emitted_tokens         : total tokens in output
    """
    try:
        # Access the spec decode worker stats via the llm_engine
        engine = llm.llm_engine
        if hasattr(engine, "stat_logger") and engine.stat_logger is not None:
            stats = engine.stat_logger.spec_decode_metrics
            if stats:
                return {
                    "draft_acceptance_rate": stats.draft_acceptance_rate,
                    "system_efficiency":     stats.system_efficiency,
                    "accepted_tokens":       stats.accepted_tokens,
                    "draft_tokens":          stats.draft_tokens,
                    "emitted_tokens":        stats.emitted_tokens,
                }
    except Exception:
        pass

    # Fallback: return empty dict if stats not accessible
    return {}


# ── Generation ────────────────────────────────────────────────────────────────

def vllm_generate(
    prompts: list,
    llm: "LLM",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> tuple[list, dict]:
    """
    Run vLLM speculative decoding on a list of prompts.

    Returns:
        completions : list of generated strings
        timing      : dict with elapsed_sec, tokens_per_sec, ttft_ms
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t_start

    completions   = [out.outputs[0].text for out in outputs]
    total_tokens  = sum(len(out.outputs[0].token_ids) for out in outputs)
    peak_gpu_gb   = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # TTFT: vLLM doesn't expose per-request TTFT in batch mode directly
    # We approximate as elapsed / num_prompts for the first token
    ttft_ms = (elapsed / len(prompts)) * 1000

    timing = {
        "elapsed_sec":   elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "ttft_ms":        ttft_ms,
        "peak_gpu_gb":    peak_gpu_gb,
        "total_tokens":   total_tokens,
    }
    return completions, timing


# ── Full benchmark runner ─────────────────────────────────────────────────────

def vllm_benchmark(
    prompts: list,
    condition: str,
    target_model: str = TARGET_MODEL,
    draft_model: str = DRAFT_MODEL,
    lora_adapter: str = None,
    gamma: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    output_dir: str = "./results",
) -> DecodingMetrics:
    """
    Full vLLM benchmark: build engine → generate → extract metrics → save.
    """
    llm = build_vllm_engine(
        target_model=target_model,
        draft_model=draft_model,
        lora_adapter=lora_adapter,
        gamma=gamma,
    )

    print(f"\nRunning vLLM generation on {len(prompts)} prompts...")
    completions, timing = vllm_generate(
        prompts=prompts,
        llm=llm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Extract speculative decoding metrics from vLLM internals
    spec_metrics = extract_spec_decode_metrics(llm)
    acceptance_rate = spec_metrics.get("draft_acceptance_rate", 0.0)

    metrics = DecodingMetrics(
        condition=f"vllm_{condition}",
        total_tokens_generated=timing["total_tokens"],
        total_draft_tokens=spec_metrics.get("draft_tokens", 0),
        total_accepted_tokens=spec_metrics.get("accepted_tokens", 0),
        acceptance_rate=acceptance_rate,
        elapsed_sec=timing["elapsed_sec"],
        tokens_per_sec=timing["tokens_per_sec"],
        time_to_first_token_ms=timing["ttft_ms"],
        peak_gpu_memory_gb=timing["peak_gpu_gb"],
    )

    print_summary(metrics)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out = {
        "condition":    condition,
        "metrics":      {
            "acceptance_rate": acceptance_rate,
            "tokens_per_sec":  timing["tokens_per_sec"],
            "ttft_ms":         timing["ttft_ms"],
            "peak_gpu_gb":     timing["peak_gpu_gb"],
            **spec_metrics,
        },
        "completions": completions,
    }
    out_path = os.path.join(output_dir, f"vllm_{condition}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="baseline_generic",
                        choices=["baseline_generic", "domain_tuned"])
    parser.add_argument("--adapter",        default=None,
                        help="LoRA adapter path (for domain_tuned)")
    parser.add_argument("--gamma",          type=int,   default=5)
    parser.add_argument("--max_new_tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--output_dir",     default="./results")
    args = parser.parse_args()

    TEST_PROMPTS = [
        "def fibonacci(n):\n    ",
        "def binary_search(arr, target):\n    ",
        "class Stack:\n    def __init__(self):\n        ",
        "def merge_sort(arr):\n    ",
        "def is_palindrome(s):\n    ",
    ]

    vllm_benchmark(
        prompts=TEST_PROMPTS,
        condition=args.condition,
        lora_adapter=args.adapter,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
