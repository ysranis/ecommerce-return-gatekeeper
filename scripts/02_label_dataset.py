#!/usr/bin/env python3
"""
Script 02 — Golden Labeling + Validation Gate + Train/Val/Test Split

Reads bitext_seeds.jsonl produced by script 01. For each synthetic dispute
message, calls DeepSeek-V3 to produce a structured JSON arbitration decision,
validates every response, retries up to 2× with a stricter prompt on failure,
discards rows that still fail, enforces a ≥1,500 row quality gate, then
splits into train/val/test sets.

Supports checkpointing: if interrupted, re-running skips already-labeled rows.

Usage:
    python scripts/02_label_dataset.py
    python scripts/02_label_dataset.py --concurrency 20
    python scripts/02_label_dataset.py --output-dir /custom/data/path
    python scripts/02_label_dataset.py --dry-run   # count rows only, no API calls
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from scripts.lib.checkpoint import append_row, load_checkpoint
from scripts.lib.deepseek_client import DeepSeekClient
from scripts.lib.prompts import LABELING_SYSTEM_PROMPT, LABELING_SYSTEM_PROMPT_STRICT
from scripts.lib.splitter import check_quality_gate, split_dataset
from scripts.lib.validator import validate


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if DeepSeek wraps its JSON in them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if len(lines) > 2 else lines[1:]
        text = "\n".join(inner).strip()
    return text


async def label_one(
    client: DeepSeekClient,
    seed: dict,
    output_path: str,
) -> dict | None:
    """Label one seed with a structured JSON arbitration decision.

    Attempts up to 3 times (1 normal prompt + 2 strict-prompt retries).
    On success: appends merged row to output_path and returns it.
    On failure: prints a discard notice and returns None.
    """
    user_prompt = seed["synthetic_message"]

    for attempt in range(3):
        system = (
            LABELING_SYSTEM_PROMPT if attempt == 0 else LABELING_SYSTEM_PROMPT_STRICT
        )
        try:
            raw = await client.generate(
                user_prompt,
                system,
                max_tokens=1024,
                temperature=0.3,
            )
            text = _strip_code_fences(raw)
            valid, parsed = validate(text)
            if valid:
                parsed["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
                row = {
                    "seed_id": seed["seed_id"],
                    "synthetic_message": seed["synthetic_message"],
                    **parsed,
                }
                append_row(output_path, row)
                return row
            # Invalid JSON or missing keys — retry with strict prompt
        except Exception as e:
            print(f"[WARN] {seed['seed_id']} attempt {attempt} API error: {e}")
            return None  # Don't retry on API errors

    print(f"[DISCARD] {seed['seed_id']} — failed validation after 3 attempts")
    return None


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


async def main(args: argparse.Namespace) -> None:
    load_dotenv()

    output_dir = Path(args.output_dir)
    seeds_path = str(output_dir / "bitext_seeds.jsonl")
    distilled_path = str(output_dir / "distilled_dataset.jsonl")

    # 1. Load seeds — handle missing file gracefully in dry-run mode
    seeds = load_checkpoint(seeds_path)
    if not seeds:
        if args.dry_run:
            print(
                f"[DRY RUN] Seeds file not found or empty: {seeds_path}\n"
                f"Run script 01 first: python scripts/01_generate_dataset.py"
            )
            return
        sys.exit(
            f"ERROR: {seeds_path} is empty or missing.\n"
            f"Run script 01 first: python scripts/01_generate_dataset.py"
        )
    print(f"Loaded {len(seeds)} seeds from {seeds_path}")

    if args.dry_run:
        done = load_checkpoint(distilled_path)
        print(
            f"\n[DRY RUN] Seeds: {len(seeds)} | Already labeled: {len(done)} | "
            f"Remaining: {len(seeds) - len(done)}"
        )
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: DEEPSEEK_API_KEY not set.\n"
            "Create a .env file with: DEEPSEEK_API_KEY=your_key_here"
        )

    # 2. Checkpoint resume
    done = load_checkpoint(distilled_path)
    done_ids = {r["seed_id"] for r in done}
    todo = [s for s in seeds if s["seed_id"] not in done_ids]
    print(f"Already labeled: {len(done_ids)} | Remaining: {len(todo)}")

    # 3. Async labeling
    if todo:
        print(f"Labeling {len(todo)} seeds (concurrency={args.concurrency})…")
        client = DeepSeekClient(api_key=api_key, concurrency=args.concurrency)
        tasks = [label_one(client, seed, distilled_path) for seed in todo]
        results = await asyncio.gather(*tasks)

        successes = sum(1 for r in results if r is not None)
        failures = sum(1 for r in results if r is None)
        print(f"\nThis run — labeled: {successes} | discarded: {failures}")

    # 4. Quality gate
    all_valid = load_checkpoint(distilled_path)
    check_quality_gate(all_valid, min_rows=1500)
    print(f"Quality gate passed: {len(all_valid)} valid rows ✓")

    # 5. Train/val/test split
    train, val, test = split_dataset(all_valid, seed=42)
    _write_jsonl(str(output_dir / "train.jsonl"), train)
    _write_jsonl(str(output_dir / "val.jsonl"), val)
    _write_jsonl(str(output_dir / "test.jsonl"), test)

    print(f"\n{'='*50}")
    print(f"Split complete.")
    print(f"  train.jsonl : {len(train)} rows")
    print(f"  val.jsonl   : {len(val)} rows")
    print(f"  test.jsonl  : {len(test)} rows  ← HELD OUT — never use during training")
    print(f"\nWeek 1 complete. Run Week 2 training on the GPU instance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label synthetic disputes and split into train/val/test."
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
        help="Count remaining rows only — no API calls (for testing)",
    )
    asyncio.run(main(parser.parse_args()))
