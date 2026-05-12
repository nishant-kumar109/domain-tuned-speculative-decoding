"""
Evaluation utilities: Pass@k calculation and result aggregation.
"""

import json
import os
from collections import defaultdict
from typing import List, Dict

import numpy as np
from scipy.special import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased Pass@k estimator (Chen et al., 2021).

    Args:
        n: total number of samples generated per problem
        c: number of correct samples
        k: k in Pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(comb(n - c, k, exact=True)) / float(comb(n, k, exact=True))


def compute_pass_at_k(results: List[Dict], k_values: List[int] = [1, 10]) -> Dict:
    """
    Compute Pass@k from a list of result dicts.

    Each result dict must have:
        - task_id: str
        - passed: bool
    """
    task_results = defaultdict(list)
    for r in results:
        task_results[r["task_id"]].append(r["passed"])

    metrics = {}
    for k in k_values:
        scores = []
        for task_id, outcomes in task_results.items():
            n = len(outcomes)
            c = sum(outcomes)
            if n >= k:
                scores.append(pass_at_k(n, c, k))
        metrics[f"pass@{k}"] = np.mean(scores) if scores else 0.0

    return metrics


def save_results(results: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def print_eval_summary(condition: str, metrics: Dict):
    print(f"\nCondition: {condition}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
