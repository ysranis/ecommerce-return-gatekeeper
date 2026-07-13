import json
from pathlib import Path


def load_checkpoint(path: str) -> list[dict]:
    """Read a .jsonl file and return its rows as a list of dicts.

    Returns an empty list if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_row(path: str, row: dict) -> None:
    """Append a single dict as a JSON line to a .jsonl file.

    Creates the file (and parent directories) if they do not exist.
    Safe to call concurrently from asyncio — each write is a single
    atomic append (one open/write/close cycle).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
