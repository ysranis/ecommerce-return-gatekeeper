import random
from collections import defaultdict

TARGET_INTENTS: list[str] = [
    "get_refund",
    "cancel_order",
    "track_refund",
    "complaint",
    "check_refund_policy",
]


def filter_bitext(rows: list[dict], target_intents: list[str]) -> list[dict]:
    """Keep only rows whose `intent` field is in target_intents."""
    return [r for r in rows if r["intent"] in target_intents]


def sample_balanced(
    rows: list[dict],
    target_intents: list[str],
    per_intent: int = 400,
    seed: int = 42,
) -> list[dict]:
    """Sample up to per_intent rows per intent, then shuffle the combined result.

    If an intent has fewer than per_intent rows, all rows for that intent are
    taken and a warning is printed.

    Args:
        rows:           Filtered rows from filter_bitext().
        target_intents: List of intent strings to include.
        per_intent:     Max rows per intent (default 400 → 2,000 total).
        seed:           Random seed for reproducibility (default 42).

    Returns:
        Flat shuffled list of sampled rows.
    """
    rng = random.Random(seed)
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)

    sampled: list[dict] = []
    for intent in target_intents:
        pool = by_intent[intent]
        if len(pool) < per_intent:
            print(
                f"[WARN] Intent '{intent}' has only {len(pool)} rows "
                f"(wanted {per_intent}) — taking all available."
            )
        taken = rng.sample(pool, min(per_intent, len(pool)))
        sampled.extend(taken)

    rng.shuffle(sampled)
    return sampled
