# Week 1 — Dataset Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Week 1 data pipeline — Bitext ingestion → DeepSeek-V3 synthetic generation → golden labeling → validation gate → train/val/test split.

**Architecture:** Two async Python scripts share a `scripts/lib/` module. Script 01 downloads Bitext, samples 2,000 seeds, and calls DeepSeek-V3 to enrich each message. Script 02 reads those seeds and calls DeepSeek-V3 again to produce structured JSON arbitration decisions, validates every response, and splits the result into train/val/test files. Both scripts checkpoint progress row-by-row so crashes are recoverable.

**Tech Stack:** Python 3.11+, `openai` (DeepSeek-compatible), `datasets` (HuggingFace), `asyncio` semaphore for concurrency, `python-dotenv`, `pytest`, `pytest-asyncio`

## Global Constraints

- Python 3.11+ required (uses `tuple[bool, dict | None]` union syntax)
- All imports from `scripts.lib.*` — consistent path from both scripts and tests
- Scripts always run from project root: `python scripts/01_generate_dataset.py`
- Output directory default: `data/` relative to project root
- Default concurrency: 15 parallel DeepSeek API calls (override with `--concurrency N`)
- Fixed random seed: `42` for all sampling and splitting operations
- Train/val/test split: 1200 / 150 / 150 rows
- Quality gate: ≥ 1,500 valid golden rows required before split
- Checkpointing: append each row immediately after completion; resume skips already-done `seed_id`s
- DeepSeek model: `deepseek-chat` via `https://api.deepseek.com/v1`
- Never commit `.env` — only `.env.example`
- `data/*.jsonl` files are gitignored (large generated data)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Create | All Python dependencies |
| `.env.example` | Create | Template for API keys |
| `pytest.ini` | Create | Test discovery + asyncio mode |
| `data/.gitkeep` | Create | Ensures data/ dir exists in git |
| `scripts/__init__.py` | Create | Makes scripts/ a package |
| `scripts/lib/__init__.py` | Create | Makes lib/ a subpackage |
| `scripts/lib/checkpoint.py` | Create | `load_checkpoint()`, `append_row()` |
| `scripts/lib/validator.py` | Create | `REQUIRED_KEYS`, `validate()` |
| `scripts/lib/prompts.py` | Create | All system prompts — strings only |
| `scripts/lib/deepseek_client.py` | Create | `DeepSeekClient` — async, semaphore, retries |
| `scripts/lib/bitext.py` | Create | `filter_bitext()`, `sample_balanced()` |
| `scripts/lib/splitter.py` | Create | `split_dataset()`, `check_quality_gate()` |
| `scripts/01_generate_dataset.py` | Create | Orchestration: ingestion + generation |
| `scripts/02_label_dataset.py` | Create | Orchestration: labeling + gate + split |
| `tests/__init__.py` | Create | Test package root |
| `tests/lib/__init__.py` | Create | Test subpackage |
| `tests/lib/test_checkpoint.py` | Create | Tests for checkpoint.py |
| `tests/lib/test_validator.py` | Create | Tests for validator.py |
| `tests/lib/test_deepseek_client.py` | Create | Tests for deepseek_client.py |
| `tests/lib/test_bitext.py` | Create | Tests for bitext.py |
| `tests/lib/test_splitter.py` | Create | Tests for splitter.py |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `pytest.ini`
- Create: `data/.gitkeep`
- Create: `scripts/__init__.py`
- Create: `scripts/lib/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/lib/__init__.py`

**Interfaces:**
- Produces: importable `scripts.lib.*` namespace; working `pytest` runner

- [ ] **Step 1: Create `requirements.txt`**

```
datasets>=2.19.0
openai>=1.30.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Create `.env.example`**

```
# Copy this file to .env and fill in your API keys.
# Never commit .env to git.
DEEPSEEK_API_KEY=your_deepseek_api_key_here
HF_TOKEN=your_huggingface_token_here
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 4: Create empty init files and data placeholder**

Create the following empty files (content: empty):
- `data/.gitkeep`
- `scripts/__init__.py`
- `scripts/lib/__init__.py`
- `tests/__init__.py`
- `tests/lib/__init__.py`

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. Verify with:
```bash
python -c "import openai, datasets, dotenv; print('OK')"
```
Expected output: `OK`

- [ ] **Step 6: Commit**

```bash
git checkout -b feature/week1-dataset-engineering
git add requirements.txt .env.example pytest.ini data/.gitkeep scripts/__init__.py scripts/lib/__init__.py tests/__init__.py tests/lib/__init__.py
git commit -m "feat: scaffold Week 1 project structure and dependencies

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `lib/checkpoint.py` — Checkpoint Helpers

**Files:**
- Create: `scripts/lib/checkpoint.py`
- Create: `tests/lib/test_checkpoint.py`

**Interfaces:**
- Produces:
  - `load_checkpoint(path: str) -> list[dict]`
  - `append_row(path: str, row: dict) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_checkpoint.py`:

```python
import json
import pytest
from scripts.lib.checkpoint import append_row, load_checkpoint


def test_load_checkpoint_returns_empty_list_when_file_missing(tmp_path):
    result = load_checkpoint(str(tmp_path / "missing.jsonl"))
    assert result == []


def test_load_checkpoint_returns_rows_from_existing_file(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n')
    result = load_checkpoint(str(path))
    assert result == [{"id": 1}, {"id": 2}]


def test_load_checkpoint_skips_blank_lines(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n')
    result = load_checkpoint(str(path))
    assert result == [{"id": 1}, {"id": 2}]


def test_append_row_creates_file_and_writes_row(tmp_path):
    path = tmp_path / "out.jsonl"
    append_row(str(path), {"seed_id": "bitext_0001", "value": "test"})
    lines = [l for l in path.read_text().strip().split("\n") if l]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"seed_id": "bitext_0001", "value": "test"}


def test_append_row_appends_multiple_rows(tmp_path):
    path = tmp_path / "out.jsonl"
    append_row(str(path), {"seed_id": "a"})
    append_row(str(path), {"seed_id": "b"})
    lines = [l for l in path.read_text().strip().split("\n") if l]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"seed_id": "a"}
    assert json.loads(lines[1]) == {"seed_id": "b"}


def test_load_then_append_roundtrip(tmp_path):
    path = str(tmp_path / "data.jsonl")
    append_row(path, {"seed_id": "x", "value": 42})
    rows = load_checkpoint(path)
    assert rows == [{"seed_id": "x", "value": 42}]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/lib/test_checkpoint.py -v
```

Expected: all 6 tests fail with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `scripts/lib/checkpoint.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/lib/test_checkpoint.py -v
```

Expected output:
```
tests/lib/test_checkpoint.py::test_load_checkpoint_returns_empty_list_when_file_missing PASSED
tests/lib/test_checkpoint.py::test_load_checkpoint_returns_rows_from_existing_file PASSED
tests/lib/test_checkpoint.py::test_load_checkpoint_skips_blank_lines PASSED
tests/lib/test_checkpoint.py::test_append_row_creates_file_and_writes_row PASSED
tests/lib/test_checkpoint.py::test_append_row_appends_multiple_rows PASSED
tests/lib/test_checkpoint.py::test_load_then_append_roundtrip PASSED
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/checkpoint.py tests/lib/test_checkpoint.py
git commit -m "feat: add checkpoint helpers (load/append .jsonl)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: `lib/validator.py` — JSON Response Validator

**Files:**
- Create: `scripts/lib/validator.py`
- Create: `tests/lib/test_validator.py`

**Interfaces:**
- Produces:
  - `REQUIRED_KEYS: list[str]`
  - `validate(response_str: str) -> tuple[bool, dict | None]`

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_validator.py`:

```python
import json
import pytest
from scripts.lib.validator import REQUIRED_KEYS, validate


def _valid_row() -> dict:
    """Return a minimal valid golden row with all required keys."""
    return {
        "chain_of_thought": "Customer ordered 15 days ago; within 30-day window.",
        "intent_action": "get_refund",
        "extracted_slots": {"order_id": "AX-1234", "invoice_id": None,
                            "return_window_days": 15, "item_condition": None},
        "gatekeeper_status": "APPROVE_AUTOMATED",
        "confidence_score": 0.95,
        "user_facing_response": "Your refund has been approved.",
    }


def test_validate_returns_true_with_all_required_keys():
    row = _valid_row()
    valid, parsed = validate(json.dumps(row))
    assert valid is True
    assert parsed == row


def test_validate_returns_false_for_invalid_json():
    valid, parsed = validate("not json {{{")
    assert valid is False
    assert parsed is None


def test_validate_returns_false_for_empty_string():
    valid, parsed = validate("")
    assert valid is False
    assert parsed is None


def test_validate_returns_false_when_one_key_missing():
    row = _valid_row()
    del row["gatekeeper_status"]
    valid, parsed = validate(json.dumps(row))
    assert valid is False
    assert parsed is None


def test_validate_returns_false_when_multiple_keys_missing():
    valid, parsed = validate(json.dumps({"chain_of_thought": "only one key"}))
    assert valid is False
    assert parsed is None


def test_required_keys_contains_six_entries():
    assert len(REQUIRED_KEYS) == 6


def test_validate_preserves_extra_keys():
    row = _valid_row()
    row["extra_key"] = "extra_value"
    valid, parsed = validate(json.dumps(row))
    assert valid is True
    assert parsed["extra_key"] == "extra_value"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/lib/test_validator.py -v
```

Expected: all 7 tests fail with `ImportError`.

- [ ] **Step 3: Implement `scripts/lib/validator.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/lib/test_validator.py -v
```

Expected output:
```
tests/lib/test_validator.py::test_validate_returns_true_with_all_required_keys PASSED
tests/lib/test_validator.py::test_validate_returns_false_for_invalid_json PASSED
tests/lib/test_validator.py::test_validate_returns_false_for_empty_string PASSED
tests/lib/test_validator.py::test_validate_returns_false_when_one_key_missing PASSED
tests/lib/test_validator.py::test_validate_returns_false_when_multiple_keys_missing PASSED
tests/lib/test_validator.py::test_required_keys_contains_six_entries PASSED
tests/lib/test_validator.py::test_validate_preserves_extra_keys PASSED
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/validator.py tests/lib/test_validator.py
git commit -m "feat: add JSON response validator

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: `lib/prompts.py` — System Prompts

**Files:**
- Create: `scripts/lib/prompts.py`

**Interfaces:**
- Produces:
  - `POLICY_HANDBOOK: str`
  - `GENERATION_SYSTEM_PROMPT: str`
  - `LABELING_SYSTEM_PROMPT: str`
  - `LABELING_SYSTEM_PROMPT_STRICT: str`

> No unit tests — these are string constants. Correctness is validated end-to-end when scripts run.

- [ ] **Step 1: Create `scripts/lib/prompts.py`**

```python
"""All system prompts used by the dataset engineering scripts.

Edit this file to tune prompt quality. Both scripts import from here —
changes take effect immediately on the next script run.
"""

POLICY_HANDBOOK = """\
ACME CORP RETURNS & DISPUTE POLICY HANDBOOK (v2.1):
- Orders within 30 days: Full refund eligible
- Orders 31-60 days: Store credit only
- Damaged items: Photo evidence required before any refund
- Orders in transit >14 business days: Eligible for immediate automated cancellation
- Fraudulent chargeback flags: Auto-escalate to human agent
- Orders already delivered and opened: Requires human review\
"""

GENERATION_SYSTEM_PROMPT = f"""\
You are a synthetic data generator for an e-commerce dispute AI training dataset.

{POLICY_HANDBOOK}

Given a seed customer support message, generate a NEW, more complex, realistic customer dispute scenario.
The new message should:
- Be messier and more conversational than the seed
- Include realistic slot data (order IDs like AX-XXXX or ORD-XXXXX, invoice IDs like INV-XXXXXX)
- Vary emotional register (frustrated, polite, aggressive, confused)
- Contain natural typos and abbreviations where appropriate

Output ONLY the new customer message. No labels, no JSON, no explanation.\
"""

LABELING_SYSTEM_PROMPT = f"""\
You are the expert arbitration engine for Acme Corp's return gatekeeper system.

{POLICY_HANDBOOK}

Analyze the customer dispute message and output a structured JSON arbitration decision.
Your output must be valid JSON matching this exact schema — no markdown, no code fences, no prose:

{{
  "chain_of_thought": "<your step-by-step reasoning>",
  "intent_action": "<one of: get_refund | cancel_order | track_refund | complaint | check_refund_policy>",
  "extracted_slots": {{
    "order_id": "<string or null>",
    "invoice_id": "<string or null>",
    "return_window_days": <integer or null>,
    "item_condition": "<string or null>"
  }},
  "policy_evaluation": {{
    "within_return_window": <true or false>,
    "item_opened": <true or false>,
    "evidence_required": <true or false>
  }},
  "gatekeeper_status": "<one of: APPROVE_AUTOMATED | REQUEST_EVIDENCE | ESCALATE_TO_HUMAN>",
  "confidence_score": <float between 0.0 and 1.0>,
  "fallback_escalation": <true or false>,
  "user_facing_response": "<the message to send back to the customer>"
}}

Start your response with {{ and end with }}. Output nothing else.\
"""

LABELING_SYSTEM_PROMPT_STRICT = f"""\
You are the expert arbitration engine for Acme Corp's return gatekeeper system.

{POLICY_HANDBOOK}

CRITICAL INSTRUCTION: Your entire response must be a single valid JSON object.
- Start with {{
- End with }}
- No markdown code fences (no ```)
- No prose before or after the JSON
- No comments inside the JSON

Required keys (ALL must be present):
  chain_of_thought       — string
  intent_action          — exactly one of: get_refund, cancel_order, track_refund, complaint, check_refund_policy
  extracted_slots        — object with: order_id, invoice_id, return_window_days, item_condition
  policy_evaluation      — object with: within_return_window, item_opened, evidence_required
  gatekeeper_status      — exactly one of: APPROVE_AUTOMATED, REQUEST_EVIDENCE, ESCALATE_TO_HUMAN
  confidence_score       — float between 0.0 and 1.0
  fallback_escalation    — boolean
  user_facing_response   — string

Analyze the customer dispute message and return ONLY the JSON object.\
"""
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from scripts.lib.prompts import GENERATION_SYSTEM_PROMPT, LABELING_SYSTEM_PROMPT, LABELING_SYSTEM_PROMPT_STRICT; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/lib/prompts.py
git commit -m "feat: add system prompts for generation and labeling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: `lib/deepseek_client.py` — Async API Client

**Files:**
- Create: `scripts/lib/deepseek_client.py`
- Create: `tests/lib/test_deepseek_client.py`

**Interfaces:**
- Produces:
  - `class DeepSeekClient`
    - `__init__(self, api_key: str, concurrency: int = 15, retry_delay: float = 1.0)`
    - `async generate(self, user_prompt: str, system_prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_deepseek_client.py`:

```python
import asyncio
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from openai import RateLimitError

from scripts.lib.deepseek_client import DeepSeekClient


def _rate_limit_error() -> RateLimitError:
    """Construct a real RateLimitError (openai SDK requires httpx.Response)."""
    return RateLimitError(
        message="rate limit exceeded",
        response=httpx.Response(
            429,
            request=httpx.Request(
                "POST", "https://api.deepseek.com/v1/chat/completions"
            ),
        ),
        body={"error": {"type": "rate_limit_error", "message": "rate limit exceeded"}},
    )


def _mock_openai_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


async def test_generate_returns_content():
    with patch("scripts.lib.deepseek_client.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response("test response")
        )
        mock_cls.return_value = mock_client

        client = DeepSeekClient(api_key="test-key", concurrency=5)
        result = await client.generate("user prompt", "system prompt")

    assert result == "test response"


async def test_generate_retries_on_rate_limit_then_succeeds():
    with patch("scripts.lib.deepseek_client.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _rate_limit_error(),
                _rate_limit_error(),
                _mock_openai_response("ok after retries"),
            ]
        )
        mock_cls.return_value = mock_client

        client = DeepSeekClient(api_key="test-key", concurrency=5, retry_delay=0.0)
        result = await client.generate("prompt", "system")

    assert result == "ok after retries"
    assert mock_client.chat.completions.create.call_count == 3


async def test_generate_raises_after_max_retries():
    with patch("scripts.lib.deepseek_client.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[_rate_limit_error()] * 4
        )
        mock_cls.return_value = mock_client

        client = DeepSeekClient(api_key="test-key", concurrency=5, retry_delay=0.0)
        with pytest.raises(RateLimitError):
            await client.generate("prompt", "system")

    assert mock_client.chat.completions.create.call_count == 4


async def test_semaphore_limits_concurrency():
    """Verify that the semaphore prevents more than `concurrency` simultaneous calls."""
    active = 0
    peak = 0

    async def slow_call(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _mock_openai_response("ok")

    with patch("scripts.lib.deepseek_client.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = slow_call
        mock_cls.return_value = mock_client

        client = DeepSeekClient(api_key="test-key", concurrency=3, retry_delay=0.0)
        tasks = [client.generate(f"prompt {i}", "system") for i in range(10)]
        await asyncio.gather(*tasks)

    assert peak <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/lib/test_deepseek_client.py -v
```

Expected: all 4 tests fail with `ImportError`.

- [ ] **Step 3: Implement `scripts/lib/deepseek_client.py`**

```python
import asyncio
from openai import AsyncOpenAI, RateLimitError

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"


class DeepSeekClient:
    """Async DeepSeek API client with concurrency control and rate-limit retries.

    Args:
        api_key:     DeepSeek API key (from DEEPSEEK_API_KEY env var).
        concurrency: Max simultaneous in-flight API calls (default 15).
        retry_delay: Base delay in seconds for exponential backoff (default 1.0).
    """

    def __init__(
        self,
        api_key: str,
        concurrency: int = 15,
        retry_delay: float = 1.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._retry_delay = retry_delay

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send a prompt to DeepSeek and return the response text.

        Retries up to 3 times with exponential backoff on rate-limit errors.
        Raises RateLimitError if all retries are exhausted.
        """
        async with self._semaphore:
            for attempt in range(4):  # 1 initial + 3 retries
                try:
                    response = await self._client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                except RateLimitError:
                    if attempt == 3:
                        raise
                    wait = self._retry_delay * (2**attempt)
                    print(
                        f"[WARN] Rate limit hit — waiting {wait:.1f}s "
                        f"(retry {attempt + 1}/3)"
                    )
                    await asyncio.sleep(wait)
        # Unreachable — semaphore always releases — but satisfies type checker
        raise RuntimeError("Unexpected exit from generate()")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/lib/test_deepseek_client.py -v
```

Expected output:
```
tests/lib/test_deepseek_client.py::test_generate_returns_content PASSED
tests/lib/test_deepseek_client.py::test_generate_retries_on_rate_limit_then_succeeds PASSED
tests/lib/test_deepseek_client.py::test_generate_raises_after_max_retries PASSED
tests/lib/test_deepseek_client.py::test_semaphore_limits_concurrency PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/deepseek_client.py tests/lib/test_deepseek_client.py
git commit -m "feat: add async DeepSeek client with semaphore and retry backoff

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: `lib/bitext.py` — Bitext Filtering and Sampling

**Files:**
- Create: `scripts/lib/bitext.py`
- Create: `tests/lib/test_bitext.py`

**Interfaces:**
- Produces:
  - `TARGET_INTENTS: list[str]`
  - `filter_bitext(rows: list[dict], target_intents: list[str]) -> list[dict]`
  - `sample_balanced(rows: list[dict], target_intents: list[str], per_intent: int = 400, seed: int = 42) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_bitext.py`:

```python
import pytest
from scripts.lib.bitext import TARGET_INTENTS, filter_bitext, sample_balanced


def _make_rows(intents: list[str], per_intent: int) -> list[dict]:
    rows = []
    for intent in intents:
        for i in range(per_intent):
            rows.append({"original_message": f"{intent} msg {i}", "intent": intent})
    return rows


def test_filter_bitext_keeps_target_intents():
    rows = [
        {"original_message": "msg1", "intent": "get_refund"},
        {"original_message": "msg2", "intent": "other_intent"},
        {"original_message": "msg3", "intent": "cancel_order"},
    ]
    result = filter_bitext(rows, TARGET_INTENTS)
    assert len(result) == 2
    assert all(r["intent"] in TARGET_INTENTS for r in result)


def test_filter_bitext_removes_non_target_intents():
    rows = [{"original_message": "msg", "intent": "unknown"}]
    result = filter_bitext(rows, TARGET_INTENTS)
    assert result == []


def test_filter_bitext_empty_input():
    assert filter_bitext([], TARGET_INTENTS) == []


def test_sample_balanced_returns_correct_total():
    rows = _make_rows(TARGET_INTENTS, 500)
    result = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    assert len(result) == len(TARGET_INTENTS) * 400


def test_sample_balanced_equal_distribution():
    rows = _make_rows(TARGET_INTENTS, 500)
    result = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    from collections import Counter
    counts = Counter(r["intent"] for r in result)
    for intent in TARGET_INTENTS:
        assert counts[intent] == 400


def test_sample_balanced_is_reproducible():
    rows = _make_rows(TARGET_INTENTS, 500)
    r1 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    r2 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    assert [r["original_message"] for r in r1] == [r["original_message"] for r in r2]


def test_sample_balanced_different_seeds_differ():
    rows = _make_rows(TARGET_INTENTS, 500)
    r1 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    r2 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=99)
    # With 2000 rows sampled randomly, these will differ with overwhelming probability
    assert [r["original_message"] for r in r1] != [r["original_message"] for r in r2]


def test_sample_balanced_takes_all_when_fewer_than_per_intent():
    # Only 200 rows per intent — should take all 200
    rows = _make_rows(TARGET_INTENTS, 200)
    result = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    assert len(result) == len(TARGET_INTENTS) * 200


def test_target_intents_contains_five_entries():
    assert len(TARGET_INTENTS) == 5
    assert "get_refund" in TARGET_INTENTS
    assert "cancel_order" in TARGET_INTENTS
    assert "track_refund" in TARGET_INTENTS
    assert "complaint" in TARGET_INTENTS
    assert "check_refund_policy" in TARGET_INTENTS
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/lib/test_bitext.py -v
```

Expected: all 9 tests fail with `ImportError`.

- [ ] **Step 3: Implement `scripts/lib/bitext.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/lib/test_bitext.py -v
```

Expected output:
```
tests/lib/test_bitext.py::test_filter_bitext_keeps_target_intents PASSED
tests/lib/test_bitext.py::test_filter_bitext_removes_non_target_intents PASSED
tests/lib/test_bitext.py::test_filter_bitext_empty_input PASSED
tests/lib/test_bitext.py::test_sample_balanced_returns_correct_total PASSED
tests/lib/test_bitext.py::test_sample_balanced_equal_distribution PASSED
tests/lib/test_bitext.py::test_sample_balanced_is_reproducible PASSED
tests/lib/test_bitext.py::test_sample_balanced_different_seeds_differ PASSED
tests/lib/test_bitext.py::test_sample_balanced_takes_all_when_fewer_than_per_intent PASSED
tests/lib/test_bitext.py::test_target_intents_contains_five_entries PASSED
9 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/bitext.py tests/lib/test_bitext.py
git commit -m "feat: add Bitext filtering and balanced sampling helpers

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 7: `lib/splitter.py` — Dataset Split and Quality Gate

**Files:**
- Create: `scripts/lib/splitter.py`
- Create: `tests/lib/test_splitter.py`

**Interfaces:**
- Produces:
  - `split_dataset(rows: list[dict], seed: int = 42, train_n: int = 1200, val_n: int = 150, test_n: int = 150) -> tuple[list[dict], list[dict], list[dict]]`
  - `check_quality_gate(rows: list[dict], min_rows: int = 1500) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_splitter.py`:

```python
import pytest
from scripts.lib.splitter import check_quality_gate, split_dataset


def _rows(n: int) -> list[dict]:
    return [{"seed_id": str(i)} for i in range(n)]


def test_split_produces_correct_counts():
    train, val, test = split_dataset(_rows(1500))
    assert len(train) == 1200
    assert len(val) == 150
    assert len(test) == 150


def test_split_covers_all_rows():
    rows = _rows(1500)
    train, val, test = split_dataset(rows)
    assert len(train) + len(val) + len(test) == 1500


def test_split_no_overlap():
    rows = _rows(1500)
    train, val, test = split_dataset(rows)
    train_ids = {r["seed_id"] for r in train}
    val_ids = {r["seed_id"] for r in val}
    test_ids = {r["seed_id"] for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_is_reproducible():
    rows = _rows(1500)
    t1, v1, te1 = split_dataset(rows, seed=42)
    t2, v2, te2 = split_dataset(rows, seed=42)
    assert t1 == t2
    assert v1 == v2
    assert te1 == te2


def test_split_differs_with_different_seed():
    rows = _rows(1500)
    t1, _, _ = split_dataset(rows, seed=42)
    t2, _, _ = split_dataset(rows, seed=99)
    assert t1 != t2


def test_quality_gate_passes_at_exactly_1500():
    check_quality_gate(_rows(1500), min_rows=1500)  # must not raise


def test_quality_gate_passes_above_threshold():
    check_quality_gate(_rows(2000), min_rows=1500)  # must not raise


def test_quality_gate_fails_below_threshold():
    with pytest.raises(SystemExit):
        check_quality_gate(_rows(1499), min_rows=1500)


def test_quality_gate_fails_at_zero():
    with pytest.raises(SystemExit):
        check_quality_gate([], min_rows=1500)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/lib/test_splitter.py -v
```

Expected: all 9 tests fail with `ImportError`.

- [ ] **Step 3: Implement `scripts/lib/splitter.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/lib/test_splitter.py -v
```

Expected output:
```
tests/lib/test_splitter.py::test_split_produces_correct_counts PASSED
tests/lib/test_splitter.py::test_split_covers_all_rows PASSED
tests/lib/test_splitter.py::test_split_no_overlap PASSED
tests/lib/test_splitter.py::test_split_is_reproducible PASSED
tests/lib/test_splitter.py::test_split_differs_with_different_seed PASSED
tests/lib/test_splitter.py::test_quality_gate_passes_at_exactly_1500 PASSED
tests/lib/test_splitter.py::test_quality_gate_passes_above_threshold PASSED
tests/lib/test_splitter.py::test_quality_gate_fails_below_threshold PASSED
tests/lib/test_splitter.py::test_quality_gate_fails_at_zero PASSED
9 passed
```

- [ ] **Step 5: Run all tests to confirm nothing is broken**

```bash
pytest -v
```

Expected: all 35 tests pass (6 + 7 + 4 + 9 + 9).

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/splitter.py tests/lib/test_splitter.py
git commit -m "feat: add dataset splitter and quality gate

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: `scripts/01_generate_dataset.py` — Generation Orchestrator

**Files:**
- Create: `scripts/01_generate_dataset.py`

**Interfaces:**
- Consumes:
  - `filter_bitext(rows, target_intents)` from `scripts.lib.bitext`
  - `sample_balanced(rows, target_intents, per_intent, seed)` from `scripts.lib.bitext`
  - `TARGET_INTENTS` from `scripts.lib.bitext`
  - `load_checkpoint(path)` from `scripts.lib.checkpoint`
  - `append_row(path, row)` from `scripts.lib.checkpoint`
  - `DeepSeekClient(api_key, concurrency)` from `scripts.lib.deepseek_client`
  - `GENERATION_SYSTEM_PROMPT` from `scripts.lib.prompts`
- Produces: `data/bitext_seeds.jsonl`

> This script is an orchestrator — no unit tests. Verify by dry-run (Step 3) before spending API credits.

- [ ] **Step 1: Create `scripts/01_generate_dataset.py`**

```python
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
```

- [ ] **Step 2: Verify the script is importable and help works**

```bash
python scripts/01_generate_dataset.py --help
```

Expected output:
```
usage: 01_generate_dataset.py [-h] [--concurrency CONCURRENCY] [--output-dir OUTPUT_DIR] [--dry-run]

Generate synthetic dispute messages from Bitext seeds.
...
```

- [ ] **Step 3: Run a dry-run to verify Bitext ingestion (no API credits spent)**

```bash
python scripts/01_generate_dataset.py --dry-run
```

Expected output (approximate — row counts depend on Bitext dataset):
```
Downloading Bitext dataset from HuggingFace…
Filtered to XXXX rows across 5 intents.
Sampled 2000 seeds (5 intents × 400 rows).

[DRY RUN] Stopping before API calls. Seeds prepared successfully.
```

If you see `[WARN] Intent 'X' has only N rows` for any intent, that intent has fewer than 400 available rows — that's acceptable, the script takes all available.

- [ ] **Step 4: Commit**

```bash
git add scripts/01_generate_dataset.py
git commit -m "feat: add script 01 — Bitext ingestion and synthetic generation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 9: `scripts/02_label_dataset.py` — Labeling Orchestrator

**Files:**
- Create: `scripts/02_label_dataset.py`

**Interfaces:**
- Consumes:
  - `load_checkpoint(path)` from `scripts.lib.checkpoint`
  - `append_row(path, row)` from `scripts.lib.checkpoint`
  - `DeepSeekClient(api_key, concurrency)` from `scripts.lib.deepseek_client`
  - `LABELING_SYSTEM_PROMPT`, `LABELING_SYSTEM_PROMPT_STRICT` from `scripts.lib.prompts`
  - `validate(response_str)` from `scripts.lib.validator`
  - `split_dataset(rows, seed)` from `scripts.lib.splitter`
  - `check_quality_gate(rows, min_rows)` from `scripts.lib.splitter`
- Consumes: `data/bitext_seeds.jsonl` (output of Task 8)
- Produces: `data/distilled_dataset.jsonl`, `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`

> Orchestrator — no unit tests. Verify with `--dry-run` before spending API credits.

- [ ] **Step 1: Create `scripts/02_label_dataset.py`**

```python
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

    # 1. Load seeds
    seeds = load_checkpoint(seeds_path)
    if not seeds:
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
```

- [ ] **Step 2: Verify the script help works**

```bash
python scripts/02_label_dataset.py --help
```

Expected output:
```
usage: 02_label_dataset.py [-h] [--concurrency CONCURRENCY] [--output-dir OUTPUT_DIR] [--dry-run]

Label synthetic disputes and split into train/val/test.
...
```

- [ ] **Step 3: Run full test suite — all 35 tests must pass**

```bash
pytest -v
```

Expected:
```
tests/lib/test_checkpoint.py::... 6 passed
tests/lib/test_validator.py::... 7 passed
tests/lib/test_deepseek_client.py::... 4 passed
tests/lib/test_bitext.py::... 9 passed
tests/lib/test_splitter.py::... 9 passed
35 passed
```

- [ ] **Step 4: Commit**

```bash
git add scripts/02_label_dataset.py
git commit -m "feat: add script 02 — golden labeling, quality gate, and train/val/test split

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push branch and open PR**

```bash
git push -u origin feature/week1-dataset-engineering
gh pr create \
  --title "Week 1: Dataset engineering pipeline" \
  --body "$(cat <<'EOF'
## Summary
- Add scripts/lib/ with checkpoint, validator, prompts, DeepSeek client, Bitext helpers, and splitter
- Add scripts/01_generate_dataset.py — Bitext ingestion + DeepSeek-V3 synthetic generation
- Add scripts/02_label_dataset.py — golden labeling, validation gate, train/val/test split
- 35 unit tests covering all lib/ modules

## Test plan
- [ ] `pytest -v` — all 35 tests pass
- [ ] `python scripts/01_generate_dataset.py --dry-run` — downloads Bitext, samples 2,000 seeds, stops before API calls
- [ ] `python scripts/02_label_dataset.py --dry-run` — counts rows, stops before API calls
- [ ] Full run with real API keys (spend ~$2.75): script 01 then script 02

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Running the Full Pipeline

Once the PR is merged and you're ready to spend the ~$2.75:

**1. Create your `.env` file:**
```bash
cp .env.example .env
# Edit .env and fill in DEEPSEEK_API_KEY and HF_TOKEN
```

**2. Run script 01 — Bitext ingestion + generation (~30–60 min):**
```bash
python scripts/01_generate_dataset.py
```
Watch for `[WARN]` lines (failures). If any appear, just re-run — checkpointing handles them.

**3. Run script 02 — labeling + split (~30–60 min):**
```bash
python scripts/02_label_dataset.py
```

**4. Verify outputs:**
```bash
wc -l data/bitext_seeds.jsonl        # should be ~2000
wc -l data/distilled_dataset.jsonl   # should be ≥1500
wc -l data/train.jsonl               # should be 1200
wc -l data/val.jsonl                 # should be 150
wc -l data/test.jsonl                # should be 150
```
