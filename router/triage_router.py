"""
Dynamic Triage Router

Rule-based classifier that routes incoming disputes to the appropriate
model track:
  - Track A (Qwen-2.5-7B): complex or emotionally charged disputes
  - Track B (Llama-3.2-3B): simple procedural tasks (fast + cheap)

This is intentionally rule-based — no additional model training required.
The PRD explicitly defers ML-based routing to a future version.
"""

HIGH_COMPLEXITY_INTENTS = {"get_refund", "complaint", "check_refund_policy"}
HIGH_EMOTION_MARKERS = {"HIGH_EMOTION", "OFFENSIVE", "FRUSTRATED"}


def route_request(
    detected_intent: str,
    emotion_markers: list[str],
) -> str:
    """Return 'track_a' or 'track_b' for the given intent and emotion markers.

    Track A (Qwen-2.5-7B) handles:
    - High-complexity intents: get_refund, complaint, check_refund_policy
    - High-emotion signals: frustrated, offensive, or highly emotional messages

    Track B (Llama-3.2-3B) handles:
    - Simple procedural intents: cancel_order, track_refund

    Args:
        detected_intent:  One of the 5 target intents (e.g. "get_refund").
        emotion_markers:  List of emotion signal strings detected upstream.

    Returns:
        "track_a" or "track_b"
    """
    if detected_intent in HIGH_COMPLEXITY_INTENTS:
        return "track_a"
    if any(m in HIGH_EMOTION_MARKERS for m in emotion_markers):
        return "track_a"
    return "track_b"


def describe_route(track: str) -> dict:
    """Return metadata about the selected track for logging / dashboard use."""
    tracks = {
        "track_a": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "adapter": "output/qwen-2.5-7b-ecommerce-gk",
            "description": "Accuracy-optimized (LoRA r=16)",
        },
        "track_b": {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "adapter": "output/llama-3.2-3b-ecommerce-gk",
            "description": "Speed-optimized (LoRA r=8)",
        },
    }
    return tracks[track]
