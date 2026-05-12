"""
QLoRA fine-tuning of TinyLlama-1.1B on CodeSearchNet (Python).

Usage:
    python src/training/finetune_qlora.py --config configs/qlora_config.yaml
    python src/training/finetune_qlora.py --config configs/qlora_config.yaml --split_fraction 0.1
"""

import argparse
import os
import yaml
import torch
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from data_utils import prepare_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["model"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        target_modules=lora["target_modules"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--split_fraction", type=float, default=None,
                        help="Override dataset fraction (for ablations: 0.1, 0.5, 1.0)")
    parser.add_argument("--source", default=None,
                        choices=["codesearchnet", "the_stack", "combined"],
                        help="Override dataset source from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_fraction = args.split_fraction or cfg["data"]["train_split_fraction"]
    source         = args.source or cfg["data"].get("source", "codesearchnet")
    stack_samples  = cfg["data"].get("stack_samples", 200_000)

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = prepare_dataset(
        source=source,
        split_fraction=split_fraction,
        stack_samples=stack_samples,
        max_seq_length=cfg["data"]["max_seq_length"],
    )

    # ── Tokenizer ─────────────────────────────────────────────────────
    model_id = cfg["model"]["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (4-bit quantized base) ──────────────────────────────────
    bnb_config = build_bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(cfg))
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────
    t = cfg["training"]
    run_name = t.get("run_name", "tinyllama-qlora-code") + f"-{source}-{split_fraction*100:.0f}pct"
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        run_name=run_name,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["data"]["max_seq_length"],
        args=training_args,
    )

    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────
    adapter_path = os.path.join(t["output_dir"], "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
