"""
Metrics tracker for speculative decoding experiments.
Logs: acceptance rate, tokens/sec, TTFT, GPU memory, KV-cache efficiency.

KV-cache efficiency definition
-------------------------------
In standard autoregressive decoding, every target forward pass produces exactly
1 new token.  In speculative decoding the target runs once to *verify* a batch of
gamma draft tokens, and may run a second time to resample a corrected token at the
rejection point.  Tracking how many output tokens each target forward pass produces
captures how well we amortise that fixed overhead:

    kv_cache_efficiency = total_tokens_generated / total_target_forward_passes

  - Pure AR baseline:          1.0  (1 token per pass)
  - Spec decoding (α=0.8, γ=5): ≈ 4–5  (several tokens per pass when acceptance is high)

A value > 1 means speculative decoding is making the target model more efficient;
the higher, the better.  The metric is logged per step (tokens_out / passes_this_step)
so you can plot efficiency vs. decoding progress.
"""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List

import torch


@dataclass
class DecodingMetrics:
    condition: str                    # e.g. "baseline_generic", "domain_tuned"
    prompt: str = ""
    total_tokens_generated: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    elapsed_sec: float = 0.0
    tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    step_acceptance_rates: List[float] = field(default_factory=list)
    # KV-cache efficiency fields
    total_target_forward_passes: int = 0   # verification + rejection resamples
    kv_cache_efficiency: float = 0.0       # tokens_generated / target_forward_passes
    step_kv_efficiencies: List[float] = field(default_factory=list)  # per-step


class MetricsTracker:
    def __init__(self, condition: str):
        self.condition = condition
        self._start_time = None
        self._first_token_time = None
        self._draft_tokens = 0
        self._accepted_tokens = 0
        self._total_tokens = 0
        self._step_rates = []
        # KV-cache efficiency tracking
        self._target_forward_passes = 0    # 1 per verification + 1 per rejection resample
        self._step_kv_efficiencies = []

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.perf_counter()
        self._first_token_time = None

    def record_first_token(self):
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()

    def record_step(self, n_draft: int, n_accepted: int, n_target_passes: int = 1):
        """
        Call after each speculative decoding step.

        Args:
            n_draft:          number of draft tokens proposed this step (gamma)
            n_accepted:       number of draft tokens accepted
            n_target_passes:  target model forward passes this step.
                              Typically 1 (verification only) or 2 (verification +
                              rejection resample).  Pass the exact count from your
                              decode loop for accurate KV-cache efficiency.
        """
        self.record_first_token()
        self._draft_tokens += n_draft
        self._accepted_tokens += n_accepted
        # tokens produced = accepted + 1 (the resampled/bonus token on any exit)
        tokens_out = n_accepted + 1
        self._total_tokens += tokens_out
        rate = n_accepted / n_draft if n_draft > 0 else 0.0
        self._step_rates.append(rate)
        self._target_forward_passes += n_target_passes
        kv_eff = tokens_out / n_target_passes if n_target_passes > 0 else 0.0
        self._step_kv_efficiencies.append(kv_eff)

    def finalize(self, prompt: str = "") -> DecodingMetrics:
        elapsed = time.perf_counter() - self._start_time
        ttft_ms = (
            (self._first_token_time - self._start_time) * 1000
            if self._first_token_time else 0.0
        )
        peak_mem = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )
        acceptance_rate = (
            self._accepted_tokens / self._draft_tokens
            if self._draft_tokens > 0 else 0.0
        )
        kv_cache_efficiency = (
            self._total_tokens / self._target_forward_passes
            if self._target_forward_passes > 0 else 0.0
        )
        return DecodingMetrics(
            condition=self.condition,
            prompt=prompt[:100],
            total_tokens_generated=self._total_tokens,
            total_draft_tokens=self._draft_tokens,
            total_accepted_tokens=self._accepted_tokens,
            acceptance_rate=acceptance_rate,
            elapsed_sec=elapsed,
            tokens_per_sec=self._total_tokens / elapsed if elapsed > 0 else 0.0,
            time_to_first_token_ms=ttft_ms,
            peak_gpu_memory_gb=peak_mem,
            step_acceptance_rates=self._step_rates,
            total_target_forward_passes=self._target_forward_passes,
            kv_cache_efficiency=kv_cache_efficiency,
            step_kv_efficiencies=self._step_kv_efficiencies,
        )


def save_metrics(metrics: List[DecodingMetrics], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(m) for m in metrics], f, indent=2)
    print(f"Metrics saved to {output_path}")


def print_summary(metrics: DecodingMetrics):
    print(f"\n{'='*55}")
    print(f"Condition            : {metrics.condition}")
    print(f"Acceptance rate      : {metrics.acceptance_rate:.3f} ({metrics.acceptance_rate*100:.1f}%)")
    print(f"Tokens/sec           : {metrics.tokens_per_sec:.1f}")
    print(f"TTFT                 : {metrics.time_to_first_token_ms:.1f} ms")
    print(f"Peak GPU memory      : {metrics.peak_gpu_memory_gb:.2f} GB")
    print(f"Total tokens         : {metrics.total_tokens_generated}")
    print(f"Target fwd passes    : {metrics.total_target_forward_passes}")
    print(f"KV-cache efficiency  : {metrics.kv_cache_efficiency:.2f}x  "
          f"(tokens/target pass; AR baseline = 1.0)")
    print(f"{'='*55}\n")
