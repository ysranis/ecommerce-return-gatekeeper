import random
import pytest
from scripts.lib.bootstrap import bootstrap_ci


def test_all_same_values_returns_zero_width_ci():
    """All 1.0 → mean 1.0, CI collapses to [1.0, 1.0]."""
    result = bootstrap_ci([1.0] * 150, n_resamples=500)
    assert result["mean"] == 1.0
    assert result["ci_lower"] == 1.0
    assert result["ci_upper"] == 1.0


def test_returns_required_keys():
    result = bootstrap_ci([0.0, 0.5, 1.0] * 50, n_resamples=100)
    assert set(result.keys()) == {"mean", "ci_lower", "ci_upper"}


def test_ci_ordering():
    """ci_lower <= mean <= ci_upper for any input."""
    random.seed(99)
    values = [random.random() for _ in range(150)]
    result = bootstrap_ci(values, n_resamples=500)
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_binary_accuracy_ci_is_plausible():
    """60% accuracy (90 correct of 150): CI should bracket 0.6."""
    values = [1.0] * 90 + [0.0] * 60
    result = bootstrap_ci(values, n_resamples=1000)
    assert abs(result["mean"] - 0.6) < 0.01
    assert result["ci_lower"] < 0.6
    assert result["ci_upper"] > 0.6


def test_values_are_rounded_to_4_decimal_places():
    result = bootstrap_ci([1 / 3] * 150, n_resamples=100)
    # Check that the mean has at most 4 decimal places
    assert result["mean"] == round(result["mean"], 4)
