"""
Train Medusa heads on top of a frozen CodeLlama-7B.

Medusa adds K lightweight linear heads to the base model.
Each head i predicts the token at position t+i+1, given the hidden state at t.
Training is fast (~1-2 hrs on A100) since the base model is fully frozen.

Usage:
    python src/training/train_medusa_heads.py --config configs/qlora_config.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import wandb


# ── Medusa Heads ──────────────────────────────────────────────────────────────

class MedusaHeads(nn.Module):
    """
    K independent linear heads, each projecting hidden_size → vocab_size.
    Head i predicts the token at position t+i+1.
    """
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        # Each head: two-layer MLP with residual (matches original Medusa paper)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            list of K tensors, each (batch, seq_len, vocab_size)
        """
        return [head(hidden_states) for head in self.heads]


# ── Dataset ───────────────────────────────────────────────────────────────────

def collate_fn(batch, tokenizer, max_length=512):
    texts = [ex["whole_func_string"] for ex in batch]
    enc   = tokenizer(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
    return enc


# ── Training ──────────────────────────────────────────────────────────────────

def train_medusa_heads(
    base_model_id: str = "codellama/CodeLlama-7b-hf",
    num_heads: int = 4,
    output_dir: str = "./checkpoints/medusa-heads",
    num_epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-3,
    warmup_ratio: float = 0.05,
    max_seq_length: int = 512,
    split_fraction: float = 0.2,    # 20% of CodeSearchNet is enough for heads
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load frozen base model (4-bit to save VRAM) ───────────────────
    print("Loading frozen base model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map="auto",
        output_hidden_states=True,
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    hidden_size = base.config.hidden_size
    vocab_size  = base.config.vocab_size
    print(f"Base model: hidden_size={hidden_size}, vocab_size={vocab_size}")

    # ── Medusa heads (only these are trained) ─────────────────────────
    heads = MedusaHeads(hidden_size, vocab_size, num_heads=num_heads).to(device)
    total_params = sum(p.numel() for p in heads.parameters())
    print(f"Medusa heads params: {total_params:,} ({num_heads} heads)")

    # ── Dataset ───────────────────────────────────────────────────────
    raw  = load_dataset("code_search_net", "python", trust_remote_code=True)["train"]
    n    = int(len(raw) * split_fraction)
    data = raw.shuffle(seed=42).select(range(n))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_length))
    print(f"Training on {n:,} samples")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    total_steps = (len(loader) // grad_accum) * num_epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps
    )

    wandb.init(project="speculative-decoding-code-llm", name="medusa-heads-training")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ── Training loop ─────────────────────────────────────────────────
    heads.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get hidden states from frozen base (no grad)
            with torch.no_grad():
                out         = base(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = out.hidden_states[-1]   # (B, L, H) — last layer

            # Predict next K tokens from each position
            head_logits = heads(hidden_states.float())  # list of K × (B, L, V)

            loss = torch.tensor(0.0, device=device)
            for k, logits in enumerate(head_logits):
                # Head k predicts token at position t+k+1
                # Shift: logits[:, :-k-1, :] predicts input_ids[:, k+1:]
                offset = k + 1
                if input_ids.shape[1] <= offset:
                    continue
                pred    = logits[:, :-offset, :].reshape(-1, vocab_size)
                targets = input_ids[:, offset:].reshape(-1)
                loss    = loss + criterion(pred, targets)

            loss = loss / (len(head_logits) * grad_accum)
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    wandb.log({"loss": loss.item() * grad_accum, "step": global_step})

    # ── Save heads ────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, "medusa_heads.pt")
    torch.save({
        "state_dict":  heads.state_dict(),
        "num_heads":   num_heads,
        "hidden_size": hidden_size,
        "vocab_size":  vocab_size,
    }, save_path)
    print(f"Medusa heads saved to {save_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",    default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--num_heads",     type=int, default=4)
    parser.add_argument("--output_dir",    default="./checkpoints/medusa-heads")
    parser.add_argument("--epochs",        type=int, default=1)
    parser.add_argument("--split_fraction", type=float, default=0.2)
    args = parser.parse_args()

    train_medusa_heads(
        base_model_id=args.base_model,
        num_heads=args.num_heads,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        split_fraction=args.split_fraction,
    )
