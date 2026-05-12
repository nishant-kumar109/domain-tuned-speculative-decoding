"""
Medusa Inference — Condition 3.

How Medusa works:
  - K extra heads sit on top of the frozen base model.
  - Each head i predicts token at position t+i+1 directly from the hidden state at t.
  - At each step, the base model does ONE forward pass.
  - All K head predictions (draft tree) are verified in the same pass.
  - Accepted tokens extend the sequence; first mismatch truncates the draft.

Key difference from speculative decoding:
  - No separate draft model loaded — heads share the base model's compute.
  - Draft tokens come from parallel heads, not sequential autoregressive steps.
  - Lower memory overhead; faster setup; but head quality depends on training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from metrics import MetricsTracker, DecodingMetrics, print_summary


# ── Medusa Heads (must match training definition) ─────────────────────────────

class MedusaHeads(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor):
        return [head(hidden_states) for head in self.heads]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_medusa_model(base_model_id: str, heads_path: str):
    """
    Load CodeLlama-7B (4-bit) + trained Medusa heads.

    Args:
        base_model_id: HuggingFace model ID for the base model
        heads_path:    Path to medusa_heads.pt saved by train_medusa_heads.py
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        output_hidden_states=True,
    )
    base.eval()

    # Load heads
    ckpt  = torch.load(heads_path, map_location="cpu")
    heads = MedusaHeads(
        hidden_size=ckpt["hidden_size"],
        vocab_size=ckpt["vocab_size"],
        num_heads=ckpt["num_heads"],
    )
    heads.load_state_dict(ckpt["state_dict"])
    heads = heads.to(next(base.parameters()).device).half()
    heads.eval()

    print(f"Medusa model loaded: {ckpt['num_heads']} heads")
    return tokenizer, base, heads


# ── Medusa decoding ────────────────────────────────────────────────────────────

@torch.no_grad()
def medusa_decode(
    prompt: str,
    base_model,
    heads: MedusaHeads,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    condition: str = "medusa",
) -> tuple[str, DecodingMetrics]:
    """
    Medusa decoding loop.

    Per step:
      1. Forward pass through base → hidden states + base logits
      2. Each Medusa head predicts a draft token from the last hidden state
      3. Accept the longest prefix where each draft token == argmax of base logits
         at the shifted position (greedy verification)
      4. Append all accepted tokens + one resampled token at the rejection point
    """
    device    = next(base_model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    tracker = MetricsTracker(condition=condition)
    tracker.start()
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # ── Single forward pass ───────────────────────────────────────
        out           = base_model(generated, output_hidden_states=True)
        hidden_states = out.hidden_states[-1]          # (1, L, H)
        base_logits   = out.logits                     # (1, L, vocab)

        # Sample next token from base (position t)
        last_logits = base_logits[:, -1, :] / temperature
        base_token  = torch.multinomial(F.softmax(last_logits, dim=-1), 1)
        tracker.record_first_token()

        # ── Medusa heads predict tokens at t+1, t+2, ..., t+K ────────
        last_hidden  = hidden_states[:, -1:, :]        # (1, 1, H)
        head_logits  = heads(last_hidden.float())      # list of K × (1, 1, vocab)
        draft_tokens = [
            torch.argmax(F.softmax(hl[:, 0, :] / temperature, dim=-1), dim=-1, keepdim=True)
            for hl in head_logits
        ]  # list of K tensors (1, 1)

        # ── Verify draft tokens ────────────────────────────────────────
        # Build candidate sequence: [base_token] + draft_tokens
        candidate  = torch.cat([base_token] + draft_tokens, dim=-1)  # (1, K+1)
        verify_ids = torch.cat([generated, candidate], dim=-1)

        verify_out   = base_model(verify_ids, output_hidden_states=False)
        verify_logits = verify_out.logits  # (1, L+K+1, vocab)

        # Position of base_token in sequence = len(generated)
        # Verification logits for positions t+1, t+2, ..., t+K+1
        verify_start = generated.shape[1]

        n_accepted = 1  # base_token always accepted
        generated  = torch.cat([generated, base_token], dim=-1)
        tokens_generated += 1

        for k in range(len(draft_tokens)):
            if tokens_generated >= max_new_tokens:
                break
            verify_logit = verify_logits[:, verify_start + k, :] / temperature
            verify_token = torch.argmax(verify_logit, dim=-1)
            if verify_token.item() == draft_tokens[k].item():
                generated = torch.cat([generated, draft_tokens[k]], dim=-1)
                n_accepted += 1
                tokens_generated += 1
            else:
                # Mismatch: append the verified token and stop this draft
                resampled = torch.multinomial(F.softmax(verify_logit, dim=-1), 1)
                generated = torch.cat([generated, resampled.unsqueeze(0)], dim=-1)
                tokens_generated += 1
                break

        tracker.record_step(n_draft=len(draft_tokens) + 1, n_accepted=n_accepted)

    metrics    = tracker.finalize(prompt=prompt)
    output_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text, metrics


if __name__ == "__main__":
    BASE_MODEL  = "codellama/CodeLlama-7b-hf"
    HEADS_PATH  = "./checkpoints/medusa-heads/medusa_heads.pt"

    print("Loading Medusa model...")
    tokenizer, base, heads = load_medusa_model(BASE_MODEL, HEADS_PATH)

    prompt = "def fibonacci(n):\n    "
    text, metrics = medusa_decode(
        prompt=prompt,
        base_model=base,
        heads=heads,
        tokenizer=tokenizer,
        max_new_tokens=128,
        condition="medusa",
    )
    print(f"\nGenerated:\n{text}")
    print_summary(metrics)
