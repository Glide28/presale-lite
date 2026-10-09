"""Конфигурация приложения (переменные окружения и необязательный файл .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Мини-загрузчик .env (стандартная библиотека). Уже заданные переменные не перезаписывает."""
    file = Path(path)
    if not file.is_file():
        return
    for raw in file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _optional_float(name: str, default: str) -> float | None:
    raw = os.getenv(name, default).strip()
    if not raw:
        return None  # пустое значение = не передавать параметр модели
    try:
        return float(raw)
    except ValueError:
        return None


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() else None


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"  # mock | openai (любой OpenAI-совместимый API)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = 180
    llm_temperature: float | None = 0.2
    llm_connect_timeout: int = 15  # быстрая проверка доступности хоста
    llm_reasoning_effort: str = ""  # OpenRouter: none | low | medium | high ("" = не передавать)
    llm_provider_sort: str = ""  # OpenRouter: throughput | latency | price ("" = не передавать)
    llm_max_tokens: int | None = None
    db_path: str = "presale.db"
    max_upload_bytes: int = 5 * 1024 * 1024
    api_token: str = "demo-token"  # простой токен доступа к API

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "180")),
            llm_temperature=_optional_float("LLM_TEMPERATURE", "0.2"),
            llm_connect_timeout=int(os.getenv("LLM_CONNECT_TIMEOUT", "15")),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "").strip().lower(),
            llm_provider_sort=os.getenv("LLM_PROVIDER_SORT", "").strip().lower(),
            llm_max_tokens=_optional_int("LLM_MAX_TOKENS"),
            db_path=os.getenv("DB_PATH", "presale.db"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
            api_token=os.getenv("API_TOKEN", "demo-token"),
        )
