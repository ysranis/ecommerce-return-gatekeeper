#!/usr/bin/env python3
"""
Script 09 — Targeted REQUEST_EVIDENCE Supplement Generation (Run 2)

Run 1 diagnosis identified REQUEST_EVIDENCE as severely underrepresented:
  - Only 12% (144/1200) of training rows were REQUEST_EVIDENCE
  - FT Qwen achieved only 23.8% per-class accuracy on REQUEST_EVIDENCE
  - 0 of 416 cancel_order training examples had REQUEST_EVIDENCE (blind spot)
  - FT Qwen collapses 15/16 REQUEST_EVIDENCE errors to ESCALATE_TO_HUMAN

This script fixes those issues:
  - Generates ~n-rows REQUEST_EVIDENCE-labelled rows across all 5 intents
  - Includes cancel_order + REQUEST_EVIDENCE scenarios (closes blind spot)
  - Chain-of-thought in generated labels explicitly states the trigger rule:
    "item damaged → photo evidence required → REQUEST_EVIDENCE, not ESCALATE"
  - Validates schema (via shared validator) AND enforces gatekeeper_status check
  - Supports checkpointing — safe to interrupt and re-run
  - Merges supplement + original train.jsonl → data/train_v2.jsonl (~25% RE)

Expected API cost: ~$0.03–0.05 for 200 rows (DeepSeek-V3 @ ~$0.27/M tokens)

Usage:
    python scripts/09_generate_request_evidence.py
    python scripts/09_generate_request_evidence.py --n-rows 200
    python scripts/09_generate_request_evidence.py --dry-run
    python scripts/09_generate_request_evidence.py --output-dir /custom/data/path
"""
import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from scripts.lib.checkpoint import append_row, load_checkpoint
from scripts.lib.deepseek_client import DeepSeekClient
from scripts.lib.prompts import (
    POLICY_HANDBOOK,
    LABELING_SYSTEM_PROMPT,
    LABELING_SYSTEM_PROMPT_STRICT,
)
from scripts.lib.validator import validate


# ---------------------------------------------------------------------------
# Targeted scenario seeds for REQUEST_EVIDENCE generation
#
# Each seed describes a customer situation where evidence is required before a
# decision can be made. Seeds span all 5 intents to close structural blind spots.
# ---------------------------------------------------------------------------
RE_SCENARIOS = [
    # --- get_refund: damaged item scenarios (primary gap) ---
    {
        "intent": "get_refund",
        "tone": "frustrated",
        "scenario": (
            "customer received a visibly cracked or shattered item and wants a refund; "
            "photo evidence of damage required per policy before refund can be issued"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "polite",
        "scenario": (
            "customer says item arrived with parts clearly broken off or missing; "
            "photo evidence of the broken/missing parts required before processing"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "confused",
        "scenario": (
            "customer received an item that stopped working after just 2–3 days of normal use; "
            "photo or video evidence of defect required to confirm it was not due to misuse"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "angry",
        "scenario": (
            "customer received an electronics item with a cracked or smashed screen; "
            "high-value item requires photo documentation before refund can be authorised"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "disappointed",
        "scenario": (
            "clothing item arrived with visible holes, tears or stitching defects; "
            "photo needed to confirm damage is manufacturing defect, not customer use"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "upset",
        "scenario": (
            "customer says item has a manufacturing defect visible straight out of the box; "
            "photo evidence required before return shipping is authorised or refund issued"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "frustrated",
        "scenario": (
            "package arrived severely damaged externally; customer not sure if contents "
            "are also damaged; photo of packaging and contents required before deciding"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "annoyed",
        "scenario": (
            "customer received item in wrong size (too small/too large) despite ordering "
            "correct size; photo of size label vs. order confirmation required to verify"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "disgusted",
        "scenario": (
            "customer received a food/perishable item that arrived visibly spoiled or mouldy; "
            "photo of spoiled item required before refund claim can be filed"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "polite",
        "scenario": (
            "customer received only part of a bundle order — some items missing from the box; "
            "photo of what arrived vs. order summary needed to confirm shortage"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "suspicious",
        "scenario": (
            "customer claims item serial number does not match what is listed on their invoice; "
            "photo of serial number label and invoice required to investigate discrepancy"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "upset",
        "scenario": (
            "toy or gift item arrived with pieces snapped off or broken inside the box; "
            "photo required before a replacement or refund can be issued"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "frustrated",
        "scenario": (
            "customer received what appears to be a counterfeit or knockoff version of the item; "
            "photo evidence required so team can verify authenticity before refund"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "polite",
        "scenario": (
            "multi-item order: one item arrived damaged, others are fine; "
            "photo of the specific damaged item required to process partial refund"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "angry",
        "scenario": (
            "customer claims the package was opened/tampered with before delivery; "
            "photo evidence of tampered packaging required to file a courier claim"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "confused",
        "scenario": (
            "fabric or material of item is visibly different from what was shown on product page; "
            "photo comparison required before a refund or exchange can be approved"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "annoyed",
        "scenario": (
            "item dimensions appear significantly smaller than advertised; "
            "photo with a ruler or measurement reference required to verify discrepancy"
        ),
    },
    {
        "intent": "get_refund",
        "tone": "polite",
        "scenario": (
            "customer received an obviously used or refurbished item instead of a new one; "
            "photo evidence of wear/use marks required before refund can be processed"
        ),
    },

    # --- cancel_order: closes the blind spot (0 cancel+REQUEST_EVIDENCE in train) ---
    {
        "intent": "cancel_order",
        "tone": "frustrated",
        "scenario": (
            "customer wants to cancel an in-transit order claiming the outer packaging "
            "was already visibly destroyed by the courier; photo of damaged packaging needed "
            "before cancellation-as-damaged-in-transit can be approved"
        ),
    },
    {
        "intent": "cancel_order",
        "tone": "polite",
        "scenario": (
            "customer wants to cancel and claims they already returned the item but "
            "cancellation was never processed; proof of return (tracking number or receipt) "
            "required before cancellation can be confirmed"
        ),
    },
    {
        "intent": "cancel_order",
        "tone": "angry",
        "scenario": (
            "customer received wrong item and wants order cancelled and refunded; "
            "photo of the wrong item next to their order confirmation required "
            "to verify mismatch before cancellation is approved"
        ),
    },
    {
        "intent": "cancel_order",
        "tone": "shocked",
        "scenario": (
            "customer claims the package arrived empty — no product inside — and wants "
            "the order cancelled; photo evidence of the empty package and sealed label required"
        ),
    },
    {
        "intent": "cancel_order",
        "tone": "upset",
        "scenario": (
            "customer wants to cancel because item arrived clearly broken; "
            "photo of the broken item required before cancellation can be filed "
            "as damaged-in-transit with the courier"
        ),
    },

    # --- check_refund_policy: evidence required before policy determination ---
    {
        "intent": "check_refund_policy",
        "tone": "polite",
        "scenario": (
            "customer asking whether a damaged-on-arrival item qualifies for full refund; "
            "photo of damage required so agent can apply correct policy tier"
        ),
    },
    {
        "intent": "check_refund_policy",
        "tone": "uncertain",
        "scenario": (
            "customer asking if a malfunctioning item qualifies for return; "
            "evidence of malfunction required to determine whether it falls under "
            "defect policy (full refund) or misuse policy (no refund)"
        ),
    },
    {
        "intent": "check_refund_policy",
        "tone": "polite",
        "scenario": (
            "customer asking about return policy for item that arrived with cosmetic damage "
            "but still functions; photo of damage required to determine applicable policy"
        ),
    },

    # --- complaint: evidence required to assess complaint ---
    {
        "intent": "complaint",
        "tone": "angry",
        "scenario": (
            "customer complaining item quality is far below what was advertised; "
            "photo comparison between received item and product listing image required"
        ),
    },
    {
        "intent": "complaint",
        "tone": "outraged",
        "scenario": (
            "customer complaining they received an obviously used or refurbished item "
            "sold as brand new; photo evidence of wear or missing new-item indicators required"
        ),
    },
    {
        "intent": "complaint",
        "tone": "frustrated",
        "scenario": (
            "customer complaining that quantity delivered was less than what was ordered; "
            "photo of received items laid out next to the order summary required"
        ),
    },
]

# Prompt used to generate a realistic customer message from a scenario seed.
# Specifically instructs the model to include evidence-triggering details.
_RE_GENERATION_SYSTEM_PROMPT = f"""\
You are a synthetic data generator for an e-commerce dispute AI training dataset.

{POLICY_HANDBOOK}

You will be given a scenario description and an emotional tone. Write ONE realistic
customer support message based on that scenario. The message must:
- Be written in first person as a real customer
- Match the specified emotional tone (frustrated, polite, angry, confused, etc.)
- Be conversational — include natural typos, abbreviations, run-on sentences as appropriate
- Include realistic slot data: order IDs like AX-XXXX or ORD-XXXXX; invoice IDs like INV-XXXXXX
- Clearly describe a situation where physical evidence (photos, documents, return tracking)
  would be needed before a refund or cancellation decision can be made
- Be 80–180 words — detailed enough to require evidence, concise enough to be realistic

Output ONLY the customer message. No labels, no JSON, no explanation.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if len(lines) > 2 else lines[1:]
        text = "\n".join(inner).strip()
    return text


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Core generation: message → label → validate → filter
# ---------------------------------------------------------------------------

async def _generate_message(
    client: DeepSeekClient,
    scenario: dict,
    seed_id: str,
) -> str | None:
    """Generate one synthetic customer message from a scenario seed."""
    user_prompt = (
        f"Intent: {scenario['intent']}\n"
        f"Tone: {scenario['tone']}\n"
        f"Scenario: {scenario['scenario']}"
    )
    try:
        return await client.generate(
            user_prompt,
            _RE_GENERATION_SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.85,
        )
    except Exception as e:
        print(f"[WARN] {seed_id} generation failed: {e}")
        return None


async def _label_message(
    client: DeepSeekClient,
    message: str,
    seed_id: str,
) -> dict | None:
    """Label a customer message with a structured JSON arbitration decision.

    Tries up to 3 times (1 normal + 2 strict retries). Returns parsed dict
    only if schema validates AND gatekeeper_status == REQUEST_EVIDENCE.
    """
    for attempt in range(3):
        system = LABELING_SYSTEM_PROMPT if attempt == 0 else LABELING_SYSTEM_PROMPT_STRICT
        try:
            raw = await client.generate(
                message,
                system,
                max_tokens=1024,
                temperature=0.2,
            )
            text = _strip_code_fences(raw)
            valid, parsed = validate(text)
            if not valid:
                continue
            if parsed.get("gatekeeper_status") != "REQUEST_EVIDENCE":
                # Labelled as something else — discard, not worth retrying
                return None
            return parsed
        except Exception as e:
            print(f"[WARN] {seed_id} labeling attempt {attempt} failed: {e}")
            continue
    return None


async def generate_one(
    client: DeepSeekClient,
    scenario: dict,
    seed_id: str,
    output_path: Path,
) -> dict | None:
    """Run the full pipeline for one attempt: generate message → label → filter → save."""
    message = await _generate_message(client, scenario, seed_id)
    if not message:
        return None

    parsed = await _label_message(client, message.strip(), seed_id)
    if not parsed:
        return None

    row = {
        "seed_id": seed_id,
        "synthetic_message": message.strip(),
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        **parsed,
    }
    append_row(str(output_path), row)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supplement_path = output_dir / "request_evidence_supplement.jsonl"
    train_v1_path = output_dir / "train.jsonl"
    train_v2_path = output_dir / "train_v2.jsonl"

    # --- Dry run: just report current state ---
    if args.dry_run:
        existing = _load_jsonl(supplement_path)
        train_v1 = _load_jsonl(train_v1_path)
        print(f"[DRY RUN] Supplement rows so far: {len(existing)} / {args.n_rows} target")
        print(f"[DRY RUN] train.jsonl rows: {len(train_v1)}")
        if train_v1:
            from collections import Counter
            re_existing = Counter(r["gatekeeper_status"] for r in train_v1)
            re_total = len(train_v1) + len(existing)
            re_count = re_existing.get("REQUEST_EVIDENCE", 0) + len(existing)
            print(
                f"[DRY RUN] Projected train_v2: {re_total} rows, "
                f"REQUEST_EVIDENCE = {re_count} ({re_count/re_total*100:.1f}%)"
            )
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: DEEPSEEK_API_KEY not set.\n"
            "Create a .env file with: DEEPSEEK_API_KEY=your_key_here"
        )

    # --- Resume from checkpoint ---
    existing = _load_jsonl(supplement_path)
    already_done = len(existing)
    remaining = args.n_rows - already_done

    if remaining <= 0:
        print(f"Target already reached: {already_done} rows in {supplement_path}")
    else:
        print(
            f"Supplement target: {args.n_rows} rows | "
            f"Already done: {already_done} | Remaining: {remaining}"
        )
        print(f"Generating with concurrency={args.concurrency}…")
        print(
            "Note: each attempt may be discarded if DeepSeek labels it as "
            "APPROVE/ESCALATE instead of REQUEST_EVIDENCE. Expect ~1.5–2× API "
            "calls per successful row."
        )

        client = DeepSeekClient(api_key=api_key, concurrency=args.concurrency)

        # Use a sequential counter for seed_ids, starting after existing rows
        seed_counter = already_done
        attempts = 0
        successes = 0
        discards = 0
        max_attempts = remaining * 3  # safety cap — avoids infinite loop

        # Run in batches equal to concurrency to stay efficient while tracking progress
        batch_size = args.concurrency
        scenarios = RE_SCENARIOS

        while successes < remaining and attempts < max_attempts:
            # Sample a batch of scenarios (random with replacement for variety)
            batch_scenarios = [
                random.choice(scenarios) for _ in range(batch_size)
            ]
            batch_ids = [
                f"re_supp_{seed_counter + i:04d}" for i in range(batch_size)
            ]

            tasks = [
                generate_one(client, sc, sid, supplement_path)
                for sc, sid in zip(batch_scenarios, batch_ids)
            ]
            results = await asyncio.gather(*tasks)

            batch_successes = sum(1 for r in results if r is not None)
            batch_discards = sum(1 for r in results if r is None)

            successes += batch_successes
            discards += batch_discards
            attempts += batch_size
            seed_counter += batch_size

            total_now = already_done + successes
            print(
                f"  Progress: {total_now}/{args.n_rows} rows "
                f"| This batch: +{batch_successes} kept, {batch_discards} discarded "
                f"| Overall hit rate: {successes/(successes+discards)*100:.0f}%"
                if (successes + discards) > 0
                else f"  Progress: {total_now}/{args.n_rows} rows"
            )

            if successes >= remaining:
                break

        final_supplement = _load_jsonl(supplement_path)
        print(
            f"\nGeneration complete: {len(final_supplement)} REQUEST_EVIDENCE rows "
            f"in {supplement_path}"
        )
        if attempts >= max_attempts and successes < remaining:
            print(
                f"[WARN] Hit max attempts ({max_attempts}) before reaching target. "
                f"Re-run to continue from checkpoint."
            )

    # --- Merge: train.jsonl + supplement → train_v2.jsonl ---
    print(f"\nMerging {train_v1_path} + {supplement_path} → {train_v2_path}…")
    train_v1 = _load_jsonl(train_v1_path)
    supplement = _load_jsonl(supplement_path)

    if not train_v1:
        sys.exit(f"ERROR: {train_v1_path} not found or empty. Run scripts 01 + 02 first.")

    combined = train_v1 + supplement
    random.seed(42)
    random.shuffle(combined)
    _write_jsonl(train_v2_path, combined)

    # Report final class distribution
    from collections import Counter
    dist = Counter(r["gatekeeper_status"] for r in combined)
    total = len(combined)
    print(f"\ntrain_v2.jsonl written: {total} rows")
    print("Class distribution:")
    for label in ["APPROVE_AUTOMATED", "ESCALATE_TO_HUMAN", "REQUEST_EVIDENCE"]:
        count = dist.get(label, 0)
        print(f"  {label:<25}: {count:>4} ({count/total*100:.1f}%)")

    re_pct = dist.get("REQUEST_EVIDENCE", 0) / total * 100
    target_pct = 25.0
    if re_pct >= target_pct:
        print(f"\nREQUEST_EVIDENCE at {re_pct:.1f}% — target {target_pct:.0f}% met ✓")
    else:
        shortfall = int((target_pct / 100 * total) - dist.get("REQUEST_EVIDENCE", 0))
        print(
            f"\n[WARN] REQUEST_EVIDENCE at {re_pct:.1f}% — below target {target_pct:.0f}%. "
            f"Generate ~{shortfall} more rows and re-run."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate targeted REQUEST_EVIDENCE supplement rows for Run 2."
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=200,
        help="Target number of REQUEST_EVIDENCE rows to generate (default: 200)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max simultaneous DeepSeek API calls (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Directory for output files (default: data/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report current state without making API calls",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
