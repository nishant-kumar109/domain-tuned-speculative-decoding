"""
MBPP benchmark runner across all four conditions.

Usage:
    # Conditions 1 & 2 (speculative decoding)
    python src/evaluation/run_mbpp.py --condition baseline_generic
    python src/evaluation/run_mbpp.py --condition domain_tuned --adapter ./checkpoints/tinyllama-qlora-code/final_adapter

    # Condition 3 (Medusa)
    python src/evaluation/run_mbpp.py --condition medusa --medusa_heads ./checkpoints/medusa-heads/medusa_heads.pt

    # Condition 4 (EAGLE-2)
    python src/evaluation/run_mbpp.py --condition eagle2 --eagle_weights ./checkpoints/eagle-draft.pt
"""

import argparse
import os
import sys
import json
import signal

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../inference"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../training"))

from eval_utils import compute_pass_at_k, save_results, print_eval_summary

TARGET_MODEL = "codellama/CodeLlama-7b-hf"
DRAFT_MODEL  = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


# ── MBPP execution ────────────────────────────────────────────────────────────

def run_mbpp_test(completion: str, test_list: list, timeout: float = 10.0) -> bool:
    """Execute completion + MBPP test cases. Returns True if all pass."""
    def handler(signum, frame):
        raise TimeoutError()

    code = completion + "\n" + "\n".join(test_list)
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        exec(compile(code, "<string>", "exec"), {})
        signal.alarm(0)
        return True
    except Exception:
        return False


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_spec_models(condition: str, adapter: str = None):
    from speculative_decoding import load_target_model, load_draft_model
    tokenizer, target = load_target_model(TARGET_MODEL)
    _, draft = load_draft_model(
        DRAFT_MODEL,
        lora_adapter=adapter if condition == "domain_tuned" else None,
    )
    return tokenizer, target, draft


def load_medusa(heads_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from medusa_inference import MedusaHeads
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, quantization_config=bnb, device_map="auto", output_hidden_states=True
    )
    model.eval()
    ckpt  = torch.load(heads_path, map_location="cpu")
    heads = MedusaHeads(ckpt["hidden_size"], ckpt["vocab_size"], ckpt["num_heads"])
    heads.load_state_dict(ckpt["state_dict"])
    heads = heads.to(next(model.parameters()).device).half().eval()
    return tokenizer, model, heads


def load_eagle(eagle_weights: str = None):
    from eagle_inference import load_eagle_model_standalone
    return load_eagle_model_standalone(TARGET_MODEL, eagle_weights)


# ── Generation functions ──────────────────────────────────────────────────────

def make_spec_generate_fn(target, draft, tokenizer, gamma, max_new_tokens, temperature):
    @torch.no_grad()
    def generate(prompt: str) -> str:
        device    = next(target.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()
        tokens_gen = 0
        while tokens_gen < max_new_tokens:
            draft_ids, draft_probs = [], []
            ctx = generated.clone()
            for _ in range(gamma):
                logits = draft(ctx).logits[:, -1, :] / temperature
                probs  = F.softmax(logits, dim=-1)
                token  = torch.multinomial(probs, 1)
                draft_ids.append(token)
                draft_probs.append(probs[0, token.item()].item())
                ctx = torch.cat([ctx, token], dim=-1)
            draft_seq  = torch.cat(draft_ids, dim=-1)
            full_ctx   = torch.cat([generated, draft_seq], dim=-1)
            tgt_logits = target(full_ctx).logits[:, generated.shape[1]-1:-1, :] / temperature
            tgt_probs  = F.softmax(tgt_logits, dim=-1)
            for i in range(gamma):
                tok = draft_seq[0, i].item()
                p, q = tgt_probs[0, i, tok].item(), draft_probs[i]
                if torch.rand(1).item() <= min(1.0, p / (q + 1e-8)):
                    generated = torch.cat([generated, draft_seq[:, i:i+1]], dim=-1)
                    tokens_gen += 1
                    if tok == tokenizer.eos_token_id or tokens_gen >= max_new_tokens:
                        break
                else:
                    tgt_last  = F.softmax(target(generated).logits[:, -1, :] / temperature, dim=-1)[0]
                    corrected = F.relu(tgt_probs[0, i] - tgt_last)
                    corrected = corrected / (corrected.sum() + 1e-8)
                    generated = torch.cat([generated, torch.multinomial(corrected, 1).unsqueeze(0)], dim=-1)
                    tokens_gen += 1
                    break
            if tokens_gen >= max_new_tokens:
                break
        return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generate


def make_medusa_generate_fn(model, heads, tokenizer, max_new_tokens, temperature):
    @torch.no_grad()
    def generate(prompt: str) -> str:
        device    = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()
        tokens_gen = 0
        while tokens_gen < max_new_tokens:
            out          = model(generated, output_hidden_states=True)
            hidden       = out.hidden_states[-1][:, -1:, :]
            base_token   = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            head_logits  = heads(hidden.float())
            draft_tokens = [torch.argmax(hl[:, 0, :], dim=-1, keepdim=True) for hl in head_logits]
            candidate    = torch.cat([base_token] + draft_tokens, dim=-1)
            verify_ids   = torch.cat([generated, candidate], dim=-1)
            verify_logits = model(verify_ids).logits
            verify_start  = generated.shape[1]
            generated  = torch.cat([generated, base_token], dim=-1)
            tokens_gen += 1
            for k in range(len(draft_tokens)):
                if tokens_gen >= max_new_tokens:
                    break
                v_tok = torch.argmax(verify_logits[:, verify_start + k, :], dim=-1)
                if v_tok.item() == draft_tokens[k].item():
                    generated = torch.cat([generated, draft_tokens[k]], dim=-1)
                    tokens_gen += 1
                else:
                    resampled = torch.multinomial(
                        F.softmax(verify_logits[:, verify_start + k, :] / temperature, dim=-1), 1
                    )
                    generated = torch.cat([generated, resampled.unsqueeze(0)], dim=-1)
                    tokens_gen += 1
                    break
        return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generate


def make_eagle_generate_fn(base, draft, tokenizer, gamma, max_new_tokens, temperature):
    from eagle_inference import eagle_decode_standalone
    def generate(prompt: str) -> str:
        text, _ = eagle_decode_standalone(
            prompt=prompt, base_model=base, draft_model=draft,
            tokenizer=tokenizer, gamma=gamma,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        return text
    return generate


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["baseline_generic", "domain_tuned", "medusa", "eagle2"])
    parser.add_argument("--adapter",       default=None, help="LoRA adapter path (domain_tuned)")
    parser.add_argument("--medusa_heads",  default="./checkpoints/medusa-heads/medusa_heads.pt")
    parser.add_argument("--eagle_weights", default=None, help="EAGLE draft weights path")
    parser.add_argument("--n_samples",     type=int,   default=10)
    parser.add_argument("--max_new_tokens",type=int,   default=256)
    parser.add_argument("--temperature",   type=float, default=0.8)
    parser.add_argument("--gamma",         type=int,   default=5)
    parser.add_argument("--output_dir",    default="./results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nRunning MBPP — condition: {args.condition}")

    # ── Load model + build generate_fn ───────────────────────────────
    if args.condition in ("baseline_generic", "domain_tuned"):
        tokenizer, target, draft = load_spec_models(args.condition, args.adapter)
        generate_fn = make_spec_generate_fn(
            target, draft, tokenizer, args.gamma, args.max_new_tokens, args.temperature
        )
    elif args.condition == "medusa":
        tokenizer, model, heads = load_medusa(args.medusa_heads)
        generate_fn = make_medusa_generate_fn(
            model, heads, tokenizer, args.max_new_tokens, args.temperature
        )
    else:  # eagle2
        tokenizer, base, draft = load_eagle(args.eagle_weights)
        generate_fn = make_eagle_generate_fn(
            base, draft, tokenizer, args.gamma, args.max_new_tokens, args.temperature
        )

    # ── Load MBPP problems ────────────────────────────────────────────
    problems = list(load_dataset("google-research-datasets/mbpp", split="test"))
    print(f"Loaded {len(problems)} MBPP problems, {args.n_samples} samples each")

    # ── Generate + evaluate ───────────────────────────────────────────
    results = []
    for problem in tqdm(problems, desc=args.condition):
        prompt = f"# {problem['text']}\n"
        for _ in range(args.n_samples):
            completion = generate_fn(prompt)
            passed     = run_mbpp_test(completion, problem["test_list"])
            results.append({
                "task_id":    str(problem["task_id"]),
                "completion": completion,
                "passed":     passed,
                "condition":  args.condition,
            })

    # ── Compute + save ────────────────────────────────────────────────
    metrics = compute_pass_at_k(results, k_values=[1, 10])
    print_eval_summary(args.condition, metrics)

    out_path = os.path.join(args.output_dir, f"mbpp_{args.condition}.json")
    save_results({"metrics": metrics, "results": results}, out_path)


if __name__ == "__main__":
    main()
