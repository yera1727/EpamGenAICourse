"""Application settings loaded from environment variables / the .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    return default if not value else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    provider: str
    gemini_api_key: str
    gemini_base_url: str
    dial_api_key: str
    dial_endpoint: str
    dial_api_version: str
    openai_api_key: str
    openai_base_url: str

    embedding_model: str
    embedding_dim: int
    embedding_send_dimensions: bool
    embedding_batch_size: int

    chat_model: str

    qdrant_url: str
    qdrant_collection: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        provider=_env("PROVIDER", "gemini").lower(),
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_base_url=_env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        dial_api_key=_env("DIAL_API_KEY"),
        dial_endpoint=_env("DIAL_ENDPOINT", "https://ai-proxy.lab.epam.com"),
        dial_api_version=_env("DIAL_API_VERSION", "2024-02-01"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        embedding_model=_env("EMBEDDING_MODEL", "gemini-embedding-001"),
        embedding_dim=int(_env("EMBEDDING_DIM", "3072")),
        embedding_send_dimensions=_env_bool("EMBEDDING_SEND_DIMENSIONS", False),
        embedding_batch_size=int(_env("EMBEDDING_BATCH_SIZE", "64")),
        chat_model=_env("CHAT_MODEL", "gemini-3.5-flash"),
        qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=_env("QDRANT_COLLECTION", "orbitra_kb"),
    )
