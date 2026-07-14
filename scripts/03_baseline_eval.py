#!/usr/bin/env python3
"""
Script 03 — Baseline & Fine-tuned Model Evaluation

Loads a HuggingFace model (optionally with a PEFT/LoRA adapter), runs zero-shot
inference on the 150-row held-out test set, and computes five metrics:
JSON validity rate, intent accuracy, gatekeeper accuracy, slot F1, and
hallucination rate.

Run BEFORE fine-tuning to capture the baseline (untrained) numbers.
Run AFTER fine-tuning (with --adapter-path) to capture the fine-tuned numbers.
The identical output schema enables direct before/after comparison via
scripts/04_compare_results.py.

NOTE: This script requires a GPU. Run it on RunPod A10G, not on a local Mac.

Usage:
    # Baseline (before fine-tuning):
    python scripts/03_baseline_eval.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
    python scripts/03_baseline_eval.py --model meta-llama/Llama-3.2-3B-Instruct --load-in-4bit

    # Fine-tuned (after training):
    python scripts/03_baseline_eval.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --adapter-path output/qwen-2.5-7b-ecommerce-gk \\
        --load-in-4bit
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.checkpoint import load_checkpoint
from scripts.lib.prompts import LABELING_SYSTEM_PROMPT
from scripts.lib.validator import validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if the model wraps its JSON in them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if len(lines) > 2 else lines[1:]
        text = "\n".join(inner).strip()
    return text


def _derive_slug(model: str) -> str:
    """Derive a short identifier from a HF model name or local path.

    Examples:
        'Qwen/Qwen2.5-7B-Instruct'         → 'qwen'
        'meta-llama/Llama-3.2-3B-Instruct' → 'llama'
        '/output/qwen-2.5-7b-gk'           → 'qwen'
    """
    lower = model.lower()
    if "qwen" in lower:
        return "qwen"
    if "llama" in lower:
        return "llama"
    return Path(model).name.lower().replace("-", "_")[:12]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_slot_f1(
    gt_order_id: str | None,
    gt_invoice_id: str | None,
    pred_order_id: str | None,
    pred_invoice_id: str | None,
) -> float:
    """Micro-averaged F1 over order_id and invoice_id slots.

    Per slot:
      TP = both gt and pred are non-null and match (case-insensitive strip)
      FP = pred is non-null but gt is null, OR pred non-null but doesn't match gt
      FN = gt is non-null but pred is null
      TN = both null — ignored

    Returns micro F1 in [0.0, 1.0].
    """
    tp = fp = fn = 0

    for gt, pred in [(gt_order_id, pred_order_id), (gt_invoice_id, pred_invoice_id)]:
        if gt is not None and pred is not None:
            if str(gt).strip().upper() == str(pred).strip().upper():
                tp += 1
            else:
                fp += 1
                fn += 1
        elif gt is None and pred is not None:
            fp += 1
        elif gt is not None and pred is None:
            fn += 1
        # gt None, pred None → TN, not counted

    if (2 * tp + fp + fn) == 0:
        return 1.0  # both slots null in gt and pred — no extraction needed, perfect
    return (2 * tp) / (2 * tp + fp + fn)


def check_hallucination(
    message: str,
    pred_order_id: str | None,
    pred_invoice_id: str | None,
) -> bool:
    """Return True if any predicted non-null slot value is NOT a substring
    of the input message (case-insensitive).

    The model's only input is the customer message, so any order/invoice ID
    it extracts must have been present in the message. If not, it fabricated it.
    """
    message_lower = message.lower()
    if pred_order_id is not None:
        if str(pred_order_id).strip().lower() not in message_lower:
            return True
    if pred_invoice_id is not None:
        if str(pred_invoice_id).strip().lower() not in message_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(args: argparse.Namespace):
    """Load tokenizer + model, optionally in 4-bit and with a PEFT adapter.

    Returns a HuggingFace text-generation pipeline.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    load_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("Using 4-bit quantization (BitsAndBytes)")

    print(f"Loading model weights…")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.adapter_path:
        from peft import PeftModel
        print(f"Loading PEFT adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    print("Model ready.\n")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )
    return pipe


def run_inference(pipe, message: str) -> str:
    """Run one inference call using the chat template.

    Returns the raw string output from the model.
    Raises on any exception (caller catches and records error).
    """
    messages = [
        {"role": "system", "content": LABELING_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    prompt_text = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    result = pipe(prompt_text)
    return result[0]["generated_text"]


# ---------------------------------------------------------------------------
# Per-row evaluation
# ---------------------------------------------------------------------------

def eval_row(pipe, row: dict) -> dict:
    """Run inference + compute all metrics for a single test row.

    Returns a per-row result dict matching the output schema.
    """
    seed_id = row["seed_id"]
    synthetic_message = row["synthetic_message"]
    gt_intent = row["intent_action"]
    gt_status = row["gatekeeper_status"]
    gt_slots = row.get("extracted_slots", {}) or {}
    gt_order_id = gt_slots.get("order_id")
    gt_invoice_id = gt_slots.get("invoice_id")

    base_result: dict = {
        "seed_id": seed_id,
        "synthetic_message": synthetic_message,
        "gt_intent_action": gt_intent,
        "gt_gatekeeper_status": gt_status,
        "gt_order_id": gt_order_id,
        "gt_invoice_id": gt_invoice_id,
        "raw_output": None,
        "json_valid": False,
        "pred_intent_action": None,
        "pred_gatekeeper_status": None,
        "pred_order_id": None,
        "pred_invoice_id": None,
        "intent_correct": False,
        "gatekeeper_correct": False,
        "slot_f1": 0.0,
        "hallucinated_slots": False,
        "error": None,
    }

    # Run inference
    try:
        raw_output = run_inference(pipe, synthetic_message)
    except Exception as e:
        base_result["error"] = f"generation_error: {e}"
        return base_result

    base_result["raw_output"] = raw_output

    # Parse and validate
    cleaned = _strip_code_fences(raw_output)
    json_valid, parsed = validate(cleaned)
    base_result["json_valid"] = json_valid

    if not json_valid:
        base_result["error"] = "json_parse_error"
        return base_result

    # Extract predictions
    pred_intent = parsed.get("intent_action")
    pred_status = parsed.get("gatekeeper_status")
    pred_slots = parsed.get("extracted_slots") or {}
    pred_order_id = pred_slots.get("order_id")
    pred_invoice_id = pred_slots.get("invoice_id")

    # Compute metrics
    intent_correct = pred_intent == gt_intent
    gatekeeper_correct = pred_status == gt_status
    row_slot_f1 = compute_slot_f1(gt_order_id, gt_invoice_id, pred_order_id, pred_invoice_id)
    hallucinated = check_hallucination(synthetic_message, pred_order_id, pred_invoice_id)

    base_result.update({
        "pred_intent_action": pred_intent,
        "pred_gatekeeper_status": pred_status,
        "pred_order_id": pred_order_id,
        "pred_invoice_id": pred_invoice_id,
        "intent_correct": intent_correct,
        "gatekeeper_correct": gatekeeper_correct,
        "slot_f1": round(row_slot_f1, 4),
        "hallucinated_slots": hallucinated,
    })
    return base_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    # Resolve paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = data_dir / "test.jsonl"

    # Derive slug and run type
    slug = _derive_slug(args.model)
    run_type = "finetuned" if args.adapter_path else "baseline"
    output_file = output_dir / f"{run_type}_results_{slug}.json"

    print(f"{'='*56}")
    print(f"MODEL EVAL — {run_type.upper()} ({slug})")
    print(f"{'='*56}")
    print(f"Model      : {args.model}")
    print(f"Adapter    : {args.adapter_path or 'none'}")
    print(f"Output     : {output_file}")
    print(f"{'='*56}\n")

    # Load test rows
    rows = load_checkpoint(str(test_path))
    if not rows:
        sys.exit(f"ERROR: {test_path} is empty or missing.\nRun scripts 01 and 02 first.")
    print(f"Loaded {len(rows)} test rows from {test_path}\n")

    # Load model
    pipe = load_model(args)

    # Evaluation loop
    results = []
    for i, row in enumerate(rows):
        row_result = eval_row(pipe, row)
        results.append(row_result)

        status_parts = [
            f"json={'OK' if row_result['json_valid'] else 'FAIL'}",
            f"intent={'OK' if row_result['intent_correct'] else 'X'}",
            f"gk={'OK' if row_result['gatekeeper_correct'] else 'X'}",
            f"f1={row_result['slot_f1']:.2f}",
            f"halluc={'Y' if row_result['hallucinated_slots'] else 'N'}",
        ]
        print(f"[{i+1:03d}/{len(rows)}] {row['seed_id']}  {' | '.join(status_parts)}")

    # Compute summary
    n = len(rows)
    json_valid_count = sum(1 for r in results if r["json_valid"])
    intent_correct_count = sum(1 for r in results if r["intent_correct"])
    gatekeeper_correct_count = sum(1 for r in results if r["gatekeeper_correct"])
    hallucinated_count = sum(1 for r in results if r["hallucinated_slots"])

    summary = {
        "json_validity_rate": round(json_valid_count / n, 4),
        "intent_accuracy": round(intent_correct_count / n, 4),
        "gatekeeper_accuracy": round(gatekeeper_correct_count / n, 4),
        "slot_f1": round(sum(r["slot_f1"] for r in results) / n, 4),
        "hallucination_rate": round(hallucinated_count / n, 4),
        "json_valid_count": json_valid_count,
        "intent_correct_count": intent_correct_count,
        "gatekeeper_correct_count": gatekeeper_correct_count,
        "total_evaluated": n,
    }

    # Write output JSON
    output = {
        "metadata": {
            "model": args.model,
            "adapter_path": args.adapter_path,
            "run_type": run_type,
            "slug": slug,
            "eval_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_file": str(test_path),
            "total_rows": n,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
        },
        "summary": summary,
        "rows": results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Final summary table
    print(f"\n{'='*56}")
    print(f"EVALUATION COMPLETE — {run_type.upper()} ({slug})")
    print(f"{'='*56}")
    print(f"Model          : {args.model}")
    print(f"Adapter        : {args.adapter_path or 'none'}")
    print(f"Output file    : {output_file}")
    print(f"{'─'*56}")
    print(f"JSON validity  : {summary['json_validity_rate']:.1%}  ({json_valid_count}/{n})")
    print(f"Intent acc.    : {summary['intent_accuracy']:.1%}  ({intent_correct_count}/{n})")
    print(f"Gatekeeper acc.: {summary['gatekeeper_accuracy']:.1%}  ({gatekeeper_correct_count}/{n})")
    print(f"Slot F1        : {summary['slot_f1']:.3f}")
    print(f"Halluc. rate   : {summary['hallucination_rate']:.1%}  ({hallucinated_count}/{n})")
    print(f"{'='*56}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model (baseline or fine-tuned) on the 150-row test set."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or local path (e.g. Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to PEFT/LoRA adapter directory (omit for baseline eval)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write evaluation JSON (default: results/)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing test.jsonl (default: data/)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per row (default: 512)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization via BitsAndBytes (recommended on A10G)",
    )
    main(parser.parse_args())
