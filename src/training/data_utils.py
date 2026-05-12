"""
Data loading and preprocessing utilities for QLoRA fine-tuning.

Supported datasets:
  - CodeSearchNet (Python subset, ~400K functions) — primary
  - The Stack (Python, filtered)                   — optional extra data
  - Combined (CodeSearchNet + The Stack)            — maximum coverage
"""

from datasets import load_dataset, concatenate_datasets


# ── CodeSearchNet ─────────────────────────────────────────────────────────────

def load_codesearchnet(split_fraction: float = 1.0, seed: int = 42):
    """
    Load CodeSearchNet Python subset.

    Args:
        split_fraction: fraction of training data to use (0.1 / 0.5 / 1.0 for ablations)
        seed: random seed for reproducibility
    """
    dataset   = load_dataset("code_search_net", "python", trust_remote_code=True)
    train_data = dataset["train"]

    if split_fraction < 1.0:
        n = int(len(train_data) * split_fraction)
        train_data = train_data.shuffle(seed=seed).select(range(n))

    print(f"[CodeSearchNet] {len(train_data):,} samples ({split_fraction*100:.0f}%)")
    return train_data


def format_codesearchnet(example: dict) -> dict:
    """Plain next-token prediction — no instruction template."""
    return {"text": example["whole_func_string"]}


# ── The Stack ─────────────────────────────────────────────────────────────────

def load_the_stack(max_samples: int = 200_000, seed: int = 42):
    """
    Load The Stack (Python subset) from HuggingFace.

    NOTE: Requires accepting the BigCode license at huggingface.co/datasets/bigcode/the-stack
    before running. Run `huggingface-cli login` and accept the license on the dataset page.

    Args:
        max_samples: cap on number of samples to load (full set is very large)
        seed: random seed
    """
    print("[The Stack] Loading Python subset (this may take a few minutes)...")
    dataset = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        split="train",
        streaming=False,        # set True to stream if disk space is limited
        trust_remote_code=True,
    )

    # Filter: keep only files that look like Python functions/scripts
    # (skip very short or very long files)
    dataset = dataset.filter(
        lambda ex: 100 < len(ex["content"]) < 8000,
        desc="Filtering by content length",
    )

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    print(f"[The Stack] {len(dataset):,} samples loaded")
    return dataset


def format_the_stack(example: dict) -> dict:
    """Extract raw file content as plain text."""
    return {"text": example["content"]}


# ── Combined dataset ──────────────────────────────────────────────────────────

def load_combined(
    csn_fraction: float = 1.0,
    stack_samples: int = 200_000,
    seed: int = 42,
):
    """
    Combine CodeSearchNet + The Stack for maximum code coverage.

    Args:
        csn_fraction:  fraction of CodeSearchNet to use
        stack_samples: number of samples from The Stack
        seed:          random seed
    """
    csn   = load_codesearchnet(split_fraction=csn_fraction, seed=seed)
    csn   = csn.map(format_codesearchnet, remove_columns=csn.column_names)

    stack = load_the_stack(max_samples=stack_samples, seed=seed)
    stack = stack.map(format_the_stack, remove_columns=stack.column_names)

    combined = concatenate_datasets([csn, stack]).shuffle(seed=seed)
    print(f"[Combined] {len(combined):,} total samples")
    return combined


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare_dataset(
    source: str = "codesearchnet",
    split_fraction: float = 1.0,
    stack_samples: int = 200_000,
    max_seq_length: int = 512,
    seed: int = 42,
):
    """
    Load and format dataset for QLoRA fine-tuning.

    Args:
        source:         "codesearchnet" | "the_stack" | "combined"
        split_fraction: fraction of CodeSearchNet to use (for ablations)
        stack_samples:  max samples from The Stack (if source includes it)
        max_seq_length: for reference; actual truncation happens in SFTTrainer
        seed:           random seed

    Returns:
        HuggingFace Dataset with a single "text" column
    """
    if source == "codesearchnet":
        raw = load_codesearchnet(split_fraction=split_fraction, seed=seed)
        return raw.map(format_codesearchnet, remove_columns=raw.column_names)

    elif source == "the_stack":
        raw = load_the_stack(max_samples=stack_samples, seed=seed)
        return raw.map(format_the_stack, remove_columns=raw.column_names)

    elif source == "combined":
        return load_combined(
            csn_fraction=split_fraction,
            stack_samples=stack_samples,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown source: {source!r}. Choose: codesearchnet | the_stack | combined")
