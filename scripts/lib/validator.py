import json

REQUIRED_KEYS: list[str] = [
    "chain_of_thought",
    "intent_action",
    "extracted_slots",
    "gatekeeper_status",
    "confidence_score",
    "user_facing_response",
]


def validate(response_str: str) -> tuple[bool, dict | None]:
    """Parse response_str as JSON and check all REQUIRED_KEYS are present.

    Returns:
        (True, parsed_dict)  if valid
        (False, None)        if invalid JSON or missing required keys
    """
    try:
        data = json.loads(response_str)
    except (json.JSONDecodeError, ValueError):
        return False, None
    if not all(k in data for k in REQUIRED_KEYS):
        return False, None
    return True, data
