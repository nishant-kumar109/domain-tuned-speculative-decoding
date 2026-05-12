#!/bin/bash
# Run QLoRA fine-tuning
# Usage:
#   bash scripts/run_finetune.sh                  → CodeSearchNet only (ablation sizes)
#   bash scripts/run_finetune.sh --source combined → CodeSearchNet + The Stack

set -e
cd "$(dirname "$0")/.."

SOURCE=${1:-"codesearchnet"}   # default: codesearchnet; pass "combined" for both datasets

echo "=== Fine-tuning: 10% of corpus (source=$SOURCE) ==="
python src/training/finetune_qlora.py --config configs/qlora_config.yaml \
    --split_fraction 0.1 --source $SOURCE

echo "=== Fine-tuning: 50% of corpus (source=$SOURCE) ==="
python src/training/finetune_qlora.py --config configs/qlora_config.yaml \
    --split_fraction 0.5 --source $SOURCE

echo "=== Fine-tuning: 100% of corpus (source=$SOURCE) ==="
python src/training/finetune_qlora.py --config configs/qlora_config.yaml \
    --split_fraction 1.0 --source $SOURCE

echo "All fine-tuning runs complete."
