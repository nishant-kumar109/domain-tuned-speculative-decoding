"""
EAGLE-2 Inference — Condition 4.

How EAGLE-2 works:
  - A small draft model takes (hidden_states[t], token[t]) as input
    and predicts the next token's hidden state, then decodes from that.
  - This is "feature-level" speculation — the draft model sees the target's
    internal representations, giving it much richer context than token IDs alone.
  - EAGLE-2 adds dynamic draft trees: it only expands tree nodes above a
    confidence threshold, making the tree adaptive per token.

Two usage modes:
  1. Official EAGLE repo (recommended for full tree attention):
       pip install eagle-llm  OR  git clone SafeAILab/EAGLE
  2. Standalone fallback implemented here (simplified, no tree attention).
     Useful when the EAGLE repo isn't installed.

We auto-detect which is available and use the best one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from metrics import MetricsTracker, DecodingMetrics, print_summary


# ── Try importing official EAGLE-2 ────────────────────────────────────────────

try:
    from eagle.model.ea_model import EaModel            # official EAGLE repo
    EAGLE_OFFICIAL = True
    print("Official EAGLE library detected.")
except ImportError:
    EAGLE_OFFICIAL = False
    print("EAGLE library not found — using standalone fallback implementation.")


# ── Standalone EAGLE-2 Draft Model ────────────────────────────────────────────

class EAGLEDraftModel(nn.Module):
    """
    Simplified EAGLE draft model.
    Input : concatenation of hidden_state[t] and token_embedding[t]  →  (hidden_size * 2)
    Output: predicted hidden_state[t+1], decoded to a token via the base model's LM head.

    In the real EAGLE-2 this is a full transformer layer; here we use a 2-layer MLP
    as a reasonable approximation for experimentation.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_state: torch.Tensor, token_embed: torch.Tensor):
        """
        Args:
            hidden_state : (B, hidden_size) — last hidden state from base model
            token_embed  : (B, hidden_size) — embedding of current token
        Returns:
            predicted_hidden : (B, hidden_size)
        """
        x = torch.cat([hidden_state, token_embed], dim=-1)
        return self.fc(x)


def load_eagle_model_official(base_model_id: str, eagle_model_path: str):
    """
    Load using the official EAGLE library.
    eagle_model_path: path or HuggingFace repo with the trained EAGLE draft model.
    """
    model = EaModel.from_pretrained(
        base_model_path=base_model_id,
        ea_model_path=eagle_model_path,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model.eval()
    return tokenizer, model


def load_eagle_model_standalone(base_model_id: str, draft_weights_path: str = None):
    """
    Load base model + standalone draft model.
    If draft_weights_path is None, initializes an untrained draft (for smoke testing).
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

    device      = next(base.parameters()).device
    hidden_size = base.config.hidden_size
    draft       = EAGLEDraftModel(hidden_size).half().to(device)

    if draft_weights_path:
        ckpt = torch.load(draft_weights_path, map_location=device)
        draft.load_state_dict(ckpt)
        print(f"Loaded EAGLE draft weights from {draft_weights_path}")
    else:
        print("EAGLE draft model initialized with random weights (smoke test only).")

    draft.eval()
    return tokenizer, base, draft


# ── Official EAGLE-2 decoding ─────────────────────────────────────────────────

@torch.no_grad()
def eagle_decode_official(
    prompt: str,
    model,          # EaModel instance
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    condition: str = "eagle2",
) -> tuple[str, DecodingMetrics]:
    """Decode using the official EAGLE-2 EaModel interface."""
    tracker = MetricsTracker(condition=condition)
    tracker.start()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)

    # EaModel.eagenerate returns (output_ids, accept_length_list)
    output_ids, accept_lengths = model.eagenerate(
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        is_llama3=False,
    )

    total_tokens   = output_ids.shape[1] - input_ids.shape[1]
    total_draft    = sum(accept_lengths) + len(accept_lengths)  # approx
    total_accepted = sum(accept_lengths)

    tracker.record_first_token()
    # Record aggregate as a single step for metrics compatibility
    tracker.record_step(n_draft=total_draft, n_accepted=total_accepted)

    metrics     = tracker.finalize(prompt=prompt)
    # Override token count with actual value
    metrics.total_tokens_generated = total_tokens
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text, metrics


# ── Standalone EAGLE-2 decoding ───────────────────────────────────────────────

@torch.no_grad()
def eagle_decode_standalone(
    prompt: str,
    base_model,
    draft_model: EAGLEDraftModel,
    tokenizer,
    gamma: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    condition: str = "eagle2",
) -> tuple[str, DecodingMetrics]:
    """
    Standalone EAGLE-2 decoding loop (no tree attention, simplified).

    Per step:
      1. Run base model → hidden_state[t], logits[t]
      2. Draft model predicts hidden_state[t+1] from (hidden[t], embed[t])
      3. Decode t+1, t+2, ... up to gamma tokens using the draft model
      4. Verify all draft tokens with one target forward pass (same as spec decoding)
    """
    device     = next(base_model.parameters()).device
    embed_layer = base_model.get_input_embeddings()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    tracker = MetricsTracker(condition=condition)
    tracker.start()
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # ── Base model pass to get current hidden state ───────────────
        out           = base_model(generated, output_hidden_states=True)
        hidden_state  = out.hidden_states[-1][:, -1, :]   # (1, H) last position
        base_probs    = F.softmax(out.logits[:, -1, :] / temperature, dim=-1)
        base_token    = torch.multinomial(base_probs, 1)   # (1, 1)
        tracker.record_first_token()

        # ── Draft with EAGLE: roll forward using draft model ──────────
        draft_tokens, draft_probs_list = [], []
        cur_hidden = hidden_state          # (1, H)
        cur_token  = base_token.squeeze(0) # (1,)

        for _ in range(gamma):
            token_embed    = embed_layer(cur_token.unsqueeze(0)).squeeze(1)  # (1, H)
            pred_hidden    = draft_model(cur_hidden.float(), token_embed.float())  # (1, H)
            # Decode predicted hidden to logits via LM head
            draft_logits   = base_model.lm_head(pred_hidden.half())           # (1, V)
            draft_probs    = F.softmax(draft_logits / temperature, dim=-1)
            draft_tok      = torch.multinomial(draft_probs, 1).squeeze(0)     # (1,)
            draft_tokens.append(draft_tok.unsqueeze(0))
            draft_probs_list.append(draft_probs[0, draft_tok.item()].item())
            cur_hidden = pred_hidden
            cur_token  = draft_tok

        # ── Verify draft tokens with target ───────────────────────────
        draft_seq    = torch.cat(draft_tokens, dim=-1)                         # (1, gamma)
        candidate    = torch.cat([base_token, draft_seq], dim=-1)              # (1, gamma+1)
        full_ids     = torch.cat([generated, candidate], dim=-1)
        tgt_out      = base_model(full_ids, output_hidden_states=False)
        tgt_logits   = tgt_out.logits[:, generated.shape[1]-1:-1, :] / temperature
        tgt_probs_all = F.softmax(tgt_logits, dim=-1)

        # Accept base token unconditionally
        generated = torch.cat([generated, base_token], dim=-1)
        n_accepted = 1
        tokens_generated += 1

        for i in range(gamma):
            if tokens_generated >= max_new_tokens:
                break
            tok = draft_seq[0, i].item()
            p   = tgt_probs_all[0, i + 1, tok].item()
            q   = draft_probs_list[i]
            if torch.rand(1).item() <= min(1.0, p / (q + 1e-8)):
                generated = torch.cat([generated, draft_seq[:, i:i+1]], dim=-1)
                n_accepted += 1
                tokens_generated += 1
            else:
                # Resample at rejection point
                corrected = F.relu(tgt_probs_all[0, i + 1] -
                                   F.softmax(base_model(generated).logits[:, -1, :] / temperature, dim=-1)[0])
                corrected = corrected / (corrected.sum() + 1e-8)
                resampled = torch.multinomial(corrected, 1).unsqueeze(0)
                generated = torch.cat([generated, resampled], dim=-1)
                tokens_generated += 1
                break

        tracker.record_step(n_draft=gamma + 1, n_accepted=n_accepted)

    metrics     = tracker.finalize(prompt=prompt)
    output_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text, metrics


# ── Unified entry point ────────────────────────────────────────────────────────

def eagle_decode(prompt: str, *args, condition="eagle2", **kwargs):
    """
    Auto-dispatch to official or standalone implementation based on what's available.
    """
    if EAGLE_OFFICIAL:
        return eagle_decode_official(prompt, *args, condition=condition, **kwargs)
    return eagle_decode_standalone(prompt, *args, condition=condition, **kwargs)


if __name__ == "__main__":
    BASE_MODEL         = "codellama/CodeLlama-7b-hf"
    EAGLE_DRAFT_WEIGHTS = None   # set to path if trained; None = smoke test

    print("Loading EAGLE model (standalone)...")
    tokenizer, base, draft = load_eagle_model_standalone(BASE_MODEL, EAGLE_DRAFT_WEIGHTS)

    prompt = "def fibonacci(n):\n    "
    text, metrics = eagle_decode_standalone(
        prompt=prompt,
        base_model=base,
        draft_model=draft,
        tokenizer=tokenizer,
        gamma=5,
        max_new_tokens=128,
        condition="eagle2",
    )
    print(f"\nGenerated:\n{text}")
    print_summary(metrics)
