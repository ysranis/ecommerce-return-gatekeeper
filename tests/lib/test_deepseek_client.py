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
