#!/usr/bin/env python3
"""
Script 06 — Fine-Tuning: Track B (Llama-3.2-3B-Instruct)

QLoRA fine-tuning of Llama-3.2-3B-Instruct using Unsloth + SFTTrainer.
Speed-optimized: smaller LoRA rank (r=8), shorter sequence length (1024),
larger batch size (8) compared to Track A.

Trains on data/train.jsonl for 3 epochs, evaluates on data/val.jsonl,
and saves the merged adapter weights to output/llama-3.2-3b-ecommerce-gk/.

Tracked in Weights & Biases (project=ecommerce-gatekeeper, group=llama-3b).

After training, run script 03 with --adapter-path to evaluate:
    python scripts/03_baseline_eval.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --adapter-path output/llama-3.2-3b-ecommerce-gk \\
        --load-in-4bit

NOTE: Run this on RunPod A10G (24GB VRAM). ~45 min, ~$0.35.
      Requires HF_TOKEN in .env (Llama is a gated model).

Usage:
    python scripts/06_train_llama.py
    python scripts/06_train_llama.py --output-dir /custom/output --epochs 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_jsonl(path: str) -> list[dict]:
    """Load a .jsonl file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _format_prompt(row: dict, tokenizer) -> str:
    """Format a training row into a chat-templated string.

    The model is trained to map:
        system: LABELING_SYSTEM_PROMPT
        user:   synthetic_message
        assistant: full JSON arbitration decision

    This mirrors the exact zero-shot format used in script 03, so the
    model learns to respond to the same prompt structure it will be
    evaluated on.
    """
    from scripts.lib.prompts import LABELING_SYSTEM_PROMPT

    # Build the ground-truth assistant response (structured JSON)
    assistant_response = {
        "chain_of_thought": row.get("chain_of_thought", ""),
        "intent_action": row.get("intent_action", ""),
        "extracted_slots": row.get("extracted_slots", {}),
        "policy_evaluation": row.get("policy_evaluation", {}),
        "gatekeeper_status": row.get("gatekeeper_status", ""),
        "confidence_score": row.get("confidence_score", 0.0),
        "fallback_escalation": row.get("fallback_escalation", False),
        "user_facing_response": row.get("user_facing_response", ""),
    }

    messages = [
        {"role": "system", "content": LABELING_SYSTEM_PROMPT},
        {"role": "user", "content": row["synthetic_message"]},
        {"role": "assistant", "content": json.dumps(assistant_response, ensure_ascii=False)},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    # Llama is a gated model — HF token required
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "ERROR: HF_TOKEN not set.\n"
            "Llama-3.2-3B-Instruct is a gated model. Add HF_TOKEN to your .env file.\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )

    # Validate W&B API key
    if not os.environ.get("WANDB_API_KEY"):
        print("[WARN] WANDB_API_KEY not set — W&B logging will be disabled.")

    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        sys.exit(f"ERROR: {train_path} not found. Run scripts 01 and 02 first.")
    if not val_path.exists():
        sys.exit(f"ERROR: {val_path} not found. Run scripts 01 and 02 first.")

    print(f"{'='*56}")
    print(f"FINE-TUNING: Track B — Llama-3.2-3B-Instruct")
    print(f"{'='*56}")
    print(f"Model      : {args.model}")
    print(f"Output dir : {output_dir}")
    print(f"Epochs     : {args.epochs}")
    print(f"{'='*56}\n")

    # --- Unsloth model load ---
    from unsloth import FastLanguageModel

    print("Loading base model via Unsloth…")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect (bfloat16 on A10G)
        token=hf_token,
    )

    # --- Attach LoRA adapters ---
    # Smaller rank (r=8) vs Track A (r=16) — speed/cost optimized
    print("Attaching LoRA adapters (r=8, Wq + Wv)…")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # --- Prepare datasets ---
    print("Preparing training and validation datasets…")
    train_rows = _load_jsonl(str(train_path))
    val_rows = _load_jsonl(str(val_path))
    print(f"  Train rows : {len(train_rows)}")
    print(f"  Val rows   : {len(val_rows)}\n")

    from datasets import Dataset

    def rows_to_hf_dataset(rows: list[dict]) -> Dataset:
        texts = [_format_prompt(row, tokenizer) for row in rows]
        return Dataset.from_dict({"text": texts})

    train_dataset = rows_to_hf_dataset(train_rows)
    val_dataset = rows_to_hf_dataset(val_rows)

    # --- W&B init ---
    import wandb
    wandb.init(
        project="ecommerce-gatekeeper",
        name="llama-3b-run-001",
        group="llama-3b",
        config={
            "model": args.model,
            "lora_r": 8,
            "lora_alpha": 16,
            "epochs": args.epochs,
            "learning_rate": 2e-4,
            "batch_size": 8,
            "grad_accum": 2,
            "max_seq_length": args.max_seq_length,
        },
    )

    # --- SFTTrainer ---
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,   # larger batch — smaller model
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=10,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name="llama-3b-run-001",
        ),
    )

    print("Starting training…")
    trainer.train()

    # --- Save adapter weights ---
    print(f"\nSaving adapter weights to {output_dir}…")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    wandb.finish()

    print(f"\n{'='*56}")
    print(f"Training complete — Track B (Llama-3.2-3B)")
    print(f"  Adapter saved : {output_dir}")
    print(f"  Next step     : run script 03 with --adapter-path to evaluate")
    print(f"{'='*56}")
    print(
        f"\n  python scripts/03_baseline_eval.py \\\n"
        f"      --model {args.model} \\\n"
        f"      --adapter-path {output_dir} \\\n"
        f"      --load-in-4bit"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.2-3B-Instruct via QLoRA (Track B)."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model name (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/llama-3.2-3b-ecommerce-gk",
        help="Directory to save adapter weights (default: output/llama-3.2-3b-ecommerce-gk)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing train.jsonl and val.jsonl (default: data/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024 — shorter than Track A for speed)",
    )
    main(parser.parse_args())
