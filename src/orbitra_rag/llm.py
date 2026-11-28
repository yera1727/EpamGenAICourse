"""LLM client: chat completions through the OpenAI-compatible API (Gemini / DIAL / OpenAI)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings, get_settings
from .providers import create_openai_client

logger = logging.getLogger(__name__)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass
class ChatResponse:
    text: str
    model: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


class LLMClient:
    """Simple request/response (and streaming) access to a chat model."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: openai.OpenAI | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client or create_openai_client(self.settings)

    # ------------------------------------------------------------------ public API

    def ask(self, prompt: str, system: str | None = None) -> ChatResponse:
        """One question -> one answer."""
        return self.chat(self.build_messages(prompt, system))

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def chat(self, messages: list[Message]) -> ChatResponse:
        started = time.perf_counter()
        response = self._client.chat.completions.create(**self._request_kwargs(messages))
        latency = time.perf_counter() - started

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        logger.debug("LLM %s answered in %.2fs, usage=%s", self.model, latency, usage)
        return ChatResponse(
            text=(choice.message.content or "").strip(),
            model=response.model or self.model,
            latency_s=latency,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the answer piece by piece (for UIs that show text as it is generated)."""
        response = self._client.chat.completions.create(**self._request_kwargs(messages), stream=True)
        for event in response:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def build_messages(prompt: str, system: str | None = None) -> list[Message]:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request_kwargs(self, messages: list[Message]) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs
