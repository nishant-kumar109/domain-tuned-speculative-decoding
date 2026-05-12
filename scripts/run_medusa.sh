#!/bin/bash
# Train Medusa heads + run benchmark (Condition 3)

set -e
cd "$(dirname "$0")/.."

echo "=== Training Medusa Heads (20% CodeSearchNet, ~1hr on A100) ==="
python src/training/train_medusa_heads.py \
    --base_model codellama/CodeLlama-7b-hf \
    --num_heads 4 \
    --output_dir ./checkpoints/medusa-heads \
    --epochs 1 \
    --split_fraction 0.2

echo "=== Medusa heads saved. Run Notebook 06 for full comparison. ==="
