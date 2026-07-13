import random
import sys


def split_dataset(
    rows: list[dict],
    seed: int = 42,
    train_n: int = 1200,
    val_n: int = 150,
    test_n: int = 150,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle rows with a fixed seed and split into train / val / test.

    Args:
        rows:    Validated golden rows from distilled_dataset.jsonl.
        seed:    Random seed (default 42 — always use the same value).
        train_n: Number of training rows (default 1200).
        val_n:   Number of validation rows (default 150).
        test_n:  Number of held-out test rows (default 150).

    Returns:
        (train, val, test) tuple of row lists.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:train_n]
    val = shuffled[train_n : train_n + val_n]
    test = shuffled[train_n + val_n : train_n + val_n + test_n]
    return train, val, test


def check_quality_gate(rows: list[dict], min_rows: int = 1500) -> None:
    """Exit with a clear error message if fewer than min_rows valid rows exist.

    Args:
        rows:     List of validated golden rows.
        min_rows: Minimum required (default 1500).
    """
    if len(rows) < min_rows:
        print(
            f"\nERROR: Quality gate FAILED\n"
            f"  Valid rows   : {len(rows)}\n"
            f"  Required     : ≥{min_rows}\n"
            f"  Discarded    : {min_rows - len(rows)} rows failed validation after retries\n"
            f"\nInvestigate the discard rate before proceeding to Week 2."
        )
        sys.exit(1)
