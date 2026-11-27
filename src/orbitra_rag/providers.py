"""Factory for the OpenAI-compatible SDK client (Google Gemini, EPAM DIAL or OpenAI).

The same client is shared by the embeddings client and (later) the chat LLM client.
"""

from __future__ import annotations

from openai import AzureOpenAI, OpenAI

from .config import Settings


class ConfigurationError(RuntimeError):
    """Raised when required settings (e.g. API key) are missing."""


def create_openai_client(settings: Settings) -> OpenAI:
    if settings.provider == "dial":
        if not settings.dial_api_key or settings.dial_api_key.startswith("your-"):
            raise ConfigurationError("DIAL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        # DIAL exposes an Azure OpenAI-compatible API:
        #   {endpoint}/openai/deployments/{deployment}/embeddings?api-version=...
        return AzureOpenAI(
            api_key=settings.dial_api_key,
            azure_endpoint=settings.dial_endpoint,
            api_version=settings.dial_api_version,
            max_retries=0,  # retries are handled by our own clients
        )
    if settings.provider == "gemini":
        if not settings.gemini_api_key or settings.gemini_api_key.startswith("your-"):
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com and put it into .env."
            )
        # Google exposes an OpenAI-compatible endpoint for Gemini models.
        return OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url, max_retries=0)
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        return OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            max_retries=0,
        )
    raise ConfigurationError(f"Unknown PROVIDER '{settings.provider}'. Use 'gemini', 'dial' or 'openai'.")
