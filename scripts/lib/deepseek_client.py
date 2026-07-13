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
