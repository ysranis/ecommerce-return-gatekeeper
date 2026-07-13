#!/usr/bin/env python3
"""
Script 01 — Bitext Ingestion + DeepSeek-V3 Synthetic Generation

Downloads the Bitext customer support dataset from HuggingFace, filters to 5
target intents, samples 400 rows per intent (2,000 total), then calls
DeepSeek-V3 to rewrite each raw message as a richer, more realistic dispute.

Supports checkpointing: if interrupted, re-running will skip already-completed
rows and continue from where it left off.

Usage:
    python scripts/01_generate_dataset.py
    python scripts/01_generate_dataset.py --concurrency 20
    python scripts/01_generate_dataset.py --output-dir /custom/data/path
    python scripts/01_generate_dataset.py --dry-run   # filter + sample only, no API calls
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from datasets import load_dataset

from scripts.lib.bitext import TARGET_INTENTS, filter_bitext, sample_balanced
from scripts.lib.checkpoint import append_row, load_checkpoint
from scripts.lib.deepseek_client import DeepSeekClient
from scripts.lib.prompts import GENERATION_SYSTEM_PROMPT


async def generate_one(
    client: DeepSeekClient,
    seed: dict,
    output_path: str,
) -> dict | None:
    """Generate one synthetic dispute message for a single seed.

    On success: appends to output_path and returns the completed row.
    On failure: prints a warning and returns None (row is skipped, not retried).
    """
    user_prompt = (
        f"Seed message (intent: {seed['intent']}):\n{seed['original_message']}"
    )
    try:
        synthetic = await client.generate(
            user_prompt,
            GENERATION_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.9,
        )
        row = {
            "seed_id": seed["seed_id"],
            "original_message": seed["original_message"],
            "intent": seed["intent"],
            "synthetic_message": synthetic.strip(),
        }
        append_row(output_path, row)
        return row
    except Exception as e:
        print(f"[WARN] Failed to generate for {seed['seed_id']}: {e}")
        return None


async def main(args: argparse.Namespace) -> None:
    load_dotenv()

    if not args.dry_run:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            sys.exit(
                "ERROR: DEEPSEEK_API_KEY not set.\n"
                "Create a .env file with: DEEPSEEK_API_KEY=your_key_here"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "bitext_seeds.jsonl")

    # 1. Load and filter Bitext dataset
    print("Downloading Bitext dataset from HuggingFace…")
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        token=hf_token,
    )
    raw_rows = [
        {"original_message": r["instruction"], "intent": r["intent"]}
        for r in ds["train"]
    ]
    filtered = filter_bitext(raw_rows, TARGET_INTENTS)
    print(f"Filtered to {len(filtered)} rows across {len(TARGET_INTENTS)} intents.")

    seeds = sample_balanced(filtered, TARGET_INTENTS, per_intent=400, seed=42)
    for i, seed in enumerate(seeds):
        seed["seed_id"] = f"bitext_{i:04d}"

    print(f"Sampled {len(seeds)} seeds ({len(TARGET_INTENTS)} intents × 400 rows).")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before API calls. Seeds prepared successfully.")
        return

    # 2. Checkpoint resume
    done = load_checkpoint(output_path)
    done_ids = {r["seed_id"] for r in done}
    todo = [s for s in seeds if s["seed_id"] not in done_ids]
    print(f"Already done: {len(done_ids)} | Remaining: {len(todo)}")

    if not todo:
        print("All seeds already generated. Nothing to do.")
        return

    # 3. Async generation
    print(f"Generating synthetic messages (concurrency={args.concurrency})…")
    client = DeepSeekClient(api_key=api_key, concurrency=args.concurrency)
    tasks = [generate_one(client, seed, output_path) for seed in todo]
    results = await asyncio.gather(*tasks)

    successes = sum(1 for r in results if r is not None)
    failures = sum(1 for r in results if r is None)

    print(f"\n{'='*50}")
    print(f"Generation complete.")
    print(f"  Written this run : {successes}")
    print(f"  Skipped (cached) : {len(done_ids)}")
    print(f"  Total in file    : {len(done_ids) + successes}")
    print(f"  Failed           : {failures}")
    print(f"  Output           : {output_path}")
    if failures:
        print(f"\n[NOTE] {failures} rows failed. Re-run the script to retry them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic dispute messages from Bitext seeds."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Max parallel DeepSeek API calls (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output files (default: data/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and filter Bitext only — no API calls (for testing)",
    )
    asyncio.run(main(parser.parse_args()))
