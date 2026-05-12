#!/bin/bash
# Run HumanEval + MBPP benchmark across all four conditions

set -e
cd "$(dirname "$0")/.."

ADAPTER="./checkpoints/tinyllama-qlora-code-100pct/final_adapter"
MEDUSA_HEADS="./checkpoints/medusa-heads/medusa_heads.pt"

echo "=== Condition 1: Baseline (generic draft) ==="
python src/evaluation/run_humaneval.py --condition baseline_generic --n_samples 10
python src/evaluation/run_mbpp.py      --condition baseline_generic --n_samples 10

echo "=== Condition 2: Domain-tuned draft (QLoRA) ==="
python src/evaluation/run_humaneval.py --condition domain_tuned --adapter $ADAPTER --n_samples 10
python src/evaluation/run_mbpp.py      --condition domain_tuned --adapter $ADAPTER --n_samples 10

echo "=== Condition 3: Medusa ==="
python src/evaluation/run_humaneval.py --condition medusa --medusa_heads $MEDUSA_HEADS --n_samples 10
python src/evaluation/run_mbpp.py      --condition medusa --medusa_heads $MEDUSA_HEADS --n_samples 10

echo "=== Condition 4: EAGLE-2 ==="
python src/evaluation/run_humaneval.py --condition eagle2 --n_samples 10
python src/evaluation/run_mbpp.py      --condition eagle2 --n_samples 10

echo "All benchmark runs complete. Results in ./results/"
