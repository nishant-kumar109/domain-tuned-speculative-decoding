"""
Custom speculative decoding loop with acceptance rate logging.

Flow per step:
  1. Draft model generates `gamma` tokens autoregressively
  2. Target model scores all gamma tokens in one forward pass
  3. Accept/reject each token using the speculative sampling criterion
  4. Append accepted tokens; rewind to first rejection point
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from metrics import MetricsTracker, DecodingMetrics, print_summary


def load_target_model(model_id: str, load_in_4bit: bool = True):
    bnb = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def load_draft_model(model_id: str, lora_adapter: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_adapter:
        model = PeftModel.from_pretrained(model, lora_adapter)
        print(f"Loaded LoRA adapter from {lora_adapter}")
    model.eval()
    return tokenizer, model


@torch.no_grad()
def speculative_decode(
    prompt: str,
    target_model,
    draft_model,
    tokenizer,
    gamma: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    condition: str = "unknown",
) -> tuple[str, DecodingMetrics]:
    """
    Run speculative decoding and return generated text + metrics.

    Args:
        gamma: number of draft tokens to speculatively generate per step
        condition: label for metrics (e.g. "baseline_generic", "domain_tuned")
    """
    device = next(target_model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    tracker = MetricsTracker(condition=condition)
    tracker.start()

    generated = input_ids.clone()
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # ── Step 1: Draft model generates gamma tokens ─────────────────
        draft_tokens = []
        draft_probs = []
        draft_input = generated.clone()

        for _ in range(gamma):
            out = draft_model(draft_input)
            logits = out.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            draft_probs.append(probs[0, token.item()].item())
            draft_input = torch.cat([draft_input, token], dim=-1)

        draft_sequence = torch.cat(draft_tokens, dim=-1)  # (1, gamma)

        # ── Step 2: Target model scores the full draft in one pass ─────
        full_input = torch.cat([generated, draft_sequence], dim=-1)
        target_out = target_model(full_input)
        target_logits = target_out.logits[:, generated.shape[1] - 1 : -1, :] / temperature
        target_probs_all = F.softmax(target_logits, dim=-1)  # (1, gamma, vocab)

        # ── Step 3: Accept / reject each draft token ───────────────────
        n_accepted = 0
        n_target_passes = 1   # 1 for the verification pass above
        for i in range(gamma):
            token_id = draft_sequence[0, i].item()
            q = draft_probs[i]                                  # draft prob
            p = target_probs_all[0, i, token_id].item()        # target prob

            accept_prob = min(1.0, p / (q + 1e-8))
            u = torch.rand(1).item()

            if u <= accept_prob:
                generated = torch.cat([generated, draft_sequence[:, i : i + 1]], dim=-1)
                n_accepted += 1
                tokens_generated += 1
                tracker.record_first_token()

                if tokens_generated >= max_new_tokens:
                    break
            else:
                # Resample from corrected distribution at rejection point
                # This is an extra target forward pass — counts against KV-cache efficiency
                corrected = F.relu(target_probs_all[0, i] - F.softmax(
                    target_model(generated).logits[:, -1, :] / temperature, dim=-1
                )[0])
                n_target_passes += 1   # rejection resample pass
                corrected = corrected / (corrected.sum() + 1e-8)
                resampled = torch.multinomial(corrected, num_samples=1).unsqueeze(0)
                generated = torch.cat([generated, resampled], dim=-1)
                tokens_generated += 1
                break

        tracker.record_step(n_draft=gamma, n_accepted=n_accepted,
                            n_target_passes=n_target_passes)

        if tokens_generated >= max_new_tokens:
            break

    metrics = tracker.finalize(prompt=prompt)
    output_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text, metrics


if __name__ == "__main__":
    # Quick smoke test — swap model IDs for your actual models
    TARGET_MODEL = "codellama/CodeLlama-7b-hf"
    DRAFT_MODEL  = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    LORA_ADAPTER = None  # set to adapter path for domain-tuned condition

    print("Loading models...")
    tokenizer, target = load_target_model(TARGET_MODEL)
    _, draft = load_draft_model(DRAFT_MODEL, lora_adapter=LORA_ADAPTER)

    prompt = "def fibonacci(n):\n    "
    condition = "baseline_generic" if LORA_ADAPTER is None else "domain_tuned"

    text, metrics = speculative_decode(
        prompt=prompt,
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        gamma=5,
        max_new_tokens=128,
        condition=condition,
    )

    print(f"\nGenerated:\n{text}")
    print_summary(metrics)
