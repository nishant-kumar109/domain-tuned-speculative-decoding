# Domain-Tuned Draft Models for Efficient Speculative Decoding in Code LLMs

**Course:** LLMs: A Hands-on Approach, CCE IISc  
**Author:** Nishant Kumar  
**Date:** May 2026

---

## Overview

Speculative decoding speeds up LLM inference by using a small draft model to propose tokens that a large target model verifies in parallel. The speedup depends on the **acceptance rate** — how often the target agrees with the draft.

This project tests the hypothesis: *fine-tuning the draft model on code-domain data improves acceptance rate and unlocks real latency gains.*

## Experimental Conditions

| # | Condition | Draft | Target | Method |
|---|---|---|---|---|
| C1 | Baseline (generic) | TinyLlama-1.1B (no tuning) | CodeLlama-7B-4bit | Speculative Decoding |
| C2 | **Domain-tuned (ours)** | TinyLlama-1.1B + QLoRA on code | CodeLlama-7B-4bit | Speculative Decoding |
| C3 | Medusa | — | CodeLlama-7B + Medusa heads | Multi-head Decoding |
| C4 | EAGLE-2 | — | CodeLlama-7B + EAGLE-2 MLP | Draft Tree Decoding |

## Notebook Execution Order

All experiments run on Google Colab with an **NVIDIA A100 (40GB)**. Run in this order:

| Step | Notebook | What it does | Runtime |
|---|---|---|---|
| 1 | `01_baseline_speculative_decoding.ipynb` | Baseline SD pipeline, acceptance rate logging | ~30 min |
| 2 | `02_qlora_finetuning.ipynb` | QLoRA fine-tune TinyLlama on CodeSearchNet (10%, 50%, 100%) | ~3 hrs |
| 3 | `03_domain_tuned_vs_baseline.ipynb` | Compare generic vs domain-tuned — acceptance rate + throughput | ~30 min |
| 4 | `06_medusa_eagle_baselines.ipynb` | Train Medusa heads + run Medusa & EAGLE-2 inference | ~2 hrs 25 min |
| 5 | `04_benchmarking_humaneval_mbpp.ipynb` | HumanEval + MBPP benchmark across all 4 conditions | ~11.8 hrs |
| 6 | `05_ablation_studies.ipynb` | Ablation studies, failure analysis, statistical tests | ~45 min |

> **Total compute: ~45 hours on A100** (~29 hrs training + ~12 hrs benchmarking + ~4 hrs inference/ablations)

### Notebook Dependencies

```
NB01 → NB02 → NB03 → NB06
                         ↓
                       NB04 → NB05
```

- NB04 requires: NB01 (baseline pipeline), NB02 (trained adapters), NB06 (Medusa heads)
- NB05 requires: NB04 results saved to Google Drive (`benchmark_*.json`)

### Artifacts

| Notebook | Saved to |
|---|---|
| NB02 | HF Hub: `nishant-k/tinyllama-code-specdraft-100pct` (and 10%, 50% variants) |
| NB06 | HF Hub: `nishant-k/medusa-heads-codellama` |
| NB04 | HF Hub: `nishant-k/speculative-decoding-benchmark-results` + Google Drive |
| NB05 | Google Drive: ablation plots + `ablations_and_failure_modes.json` |
| All | WandB: `nishantkr109-na/speculative-decoding-code-llm` |

### Actual Training Times (A100 40GB)

| Corpus | Samples | Time | Notes |
|---|---|---|---|
| 10% | 41K | 3h 15min | Single session |
| 50% | 206K | ~8.5 hrs | Resumed from checkpoint after session crash |
| 100% | 412K | ~17.5 hrs | Two sessions with overnight reset |

---

## Technical Notes

**VRAM requirements:** CodeLlama-7B (bf16) ~14 GB + TinyLlama (bf16) ~2.2 GB = ~16 GB minimum. A100 40GB recommended for training + inference simultaneously.

**PEFT inference overhead:** Loading the QLoRA adapter via `PeftModel` adds per-forward-pass latency (~24–33% throughput reduction vs generic draft). Use `merge_and_unload()` before inference to fold adapter weights into the base model and eliminate this overhead.

**vLLM incompatibility:** vLLM validates exact draft/target vocab size match. CodeLlama (32,016 tokens) and TinyLlama (32,000 tokens) mismatch blocks vLLM initialisation. All throughput numbers are from a custom HF inference loop (single sequence, no KV-cache reuse) and underestimate production performance.

**Vocab masking:** Target model logits beyond index 32,000 are masked to `-inf` before softmax to prevent rejected draft tokens from generating invalid token IDs in the TinyLlama embedding layer.

---

## Project Structure

```
domain-tuned-speculative-decoding/
├── notebooks/                         # Colab notebooks (run in order NB01–NB06)
├── src/
│   ├── training/
│   │   ├── data_utils.py              # CodeSearchNet loading & formatting
│   │   └── finetune_qlora.py          # QLoRA fine-tuning script
│   ├── inference/
│   │   ├── speculative_decoding.py    # SD loop + acceptance rate logging
│   │   └── metrics.py                 # latency, throughput, GPU memory
│   └── evaluation/
│       ├── eval_utils.py              # Pass@k calculation
│       └── run_humaneval.py           # HumanEval benchmark runner
├── configs/
│   ├── qlora_config.yaml              # QLoRA hyperparameters
│   ├── speculative_decoding.yaml      # inference pipeline config
│   └── benchmark_config.yaml         # evaluation settings
├── results/
│   ├── benchmarking/                  # HumanEval/MBPP JSON outputs + plots
│   ├── domain_tuning/                 # Per-run training metrics (10%/50%/100%)
│   ├── ablations/                     # A1–A3, F1–F4, S1–S3 plots + JSON
│   ├── efficiency/                    # Throughput & memory results
│   └── training-weights-and-biases/   # WandB training graphs
├── scripts/
├── report.md                          # Full project report (Markdown)
├── project-report-nishant.pdf         # Compiled PDF report
├── project-proposal-nishant.pdf       # Original project proposal
├── generate_report_pdf.py             # PDF generator (WeasyPrint)
└── requirements.txt
```

## Setup

```bash
git clone <repo-url>
cd domain-tuned-speculative-decoding
pip install -r requirements.txt
# Set HF_TOKEN and WANDB_API_KEY in your environment
```

## Key Metrics

- **Acceptance rate** — fraction of draft tokens accepted by the target model
- **Tokens/sec** — throughput (wall-clock)
- **Speedup** — tokens/sec relative to autoregressive baseline
- **Pass@1** — unbiased code correctness estimator (Chen et al., 2021) on HumanEval + MBPP
