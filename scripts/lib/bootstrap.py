"""Bootstrap confidence interval computation (pure stdlib — no numpy)."""
import random
import statistics


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Compute bootstrap confidence interval for a list of per-row metric values.

    Args:
        values:      Per-row metric values (e.g. [1.0, 0.0, 1.0, ...] for accuracy).
        n_resamples: Number of bootstrap resamples (default 1000).
        ci:          Confidence level (default 0.95 → 95% CI).

    Returns:
        {"mean": float, "ci_lower": float, "ci_upper": float} — all rounded to 4 dp.
    """
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()

    alpha = (1 - ci) / 2
    lower_idx = max(0, int(alpha * n_resamples))
    upper_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples) - 1)

    return {
        "mean": round(statistics.mean(values), 4),
        "ci_lower": round(means[lower_idx], 4),
        "ci_upper": round(means[upper_idx], 4),
    }
