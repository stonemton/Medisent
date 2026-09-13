"""Конфигурация из окружения. Ключей в коде нет — только имена переменных."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Все настройки бота. Читаются из окружения или из .env рядом с проектом."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Telegram -------------------------------------------------------
    telegram_bot_token: str
    telegram_owner_id: int

    # --- База -----------------------------------------------------------
    database_url: str

    # --- LLM / Gemini ---------------------------------------------------
    # GEMINI_API_KEY оставлен для прямого Google Gemini.
    gemini_api_key: str = ""
    # RelayModels — OpenAI-compatible шлюз. Если ключ задан, он имеет приоритет.
    relaymodels_api_key: str = ""
    relaymodels_base_url: str = "https://api.relaymodels.com/v1"
    relaymodels_transcribe_model: str = "gpt-4o-transcribe"
    llm_report_model: str = "gemini-3.8-flash"
    llm_email_model: str = "gemini-3.8-flash"
    # Одиночный агент/арбитр. Railway может переопределять это значение.
    llm_agent_model: str = "gemini-3.8-flash"
    # Пакетный агент видит всю закупку одним запросом. По умолчанию — GPT через RelayModels.
    llm_batch_agent_model: str = "gpt-5.6-sol"

    # --- Поиск и скрейпинг ----------------------------------------------
    perplexity_api_key: str = ""
    firecrawl_api_key: str = ""

    # --- Почта ----------------------------------------------------------
    yandex_email: str = ""
    yandex_app_password: str = ""
    imap_host: str = "imap.yandex.ru"
    imap_port: int = 993
    smtp_host: str = "smtp.yandex.ru"
    smtp_port: int = 587
    forward_to_email: str = ""
    gmail_sender: str = ""

    # --- Реестры --------------------------------------------------------
    registry_elk_base: str = "https://elk.roszdravnadzor.gov.ru"
    registry_misearch_url: str = "https://roszdravnadzor.gov.ru/services/misearch"
    registry_unrega_url: str = "https://roszdravnadzor.gov.ru/services/unreg"
    registry_cache_days: int = 30

    # --- Эксплуатация ---------------------------------------------------
    log_level: str = "INFO"
    log_dir: str = "logs"
    daily_api_budget_usd: float = 5.0
    max_cost_per_request_usd: float = 0.5
    gmail_poll_seconds: int = 240
    http_timeout_connect: float = 10.0
    http_timeout_read: float = 60.0
    http_max_retries: int = 3
    scrape_concurrency: int = 5

    prompt_guard_enabled: bool = False

    prompts_dir: Path = Field(default=PROJECT_ROOT / "prompts")
    kp_builder_dir: Path = Field(default=PROJECT_ROOT / "skills" / "kp-builder")

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL должен начинаться с postgresql:// или postgresql+asyncpg://"
            )
        return value

    @field_validator("gmail_poll_seconds")
    @classmethod
    def _sane_poll_interval(cls, value: int) -> int:
        if value < 60:
            raise ValueError("GMAIL_POLL_SECONDS должен быть не меньше 60 секунд")
        return value

    @model_validator(mode="after")
    def _mail_aliases(self) -> "Settings":
        if self.yandex_email and not self.gmail_sender:
            self.gmail_sender = self.yandex_email
        return self

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    @property
    def search_enabled(self) -> bool:
        return bool(self.perplexity_api_key)

    @property
    def scrape_enabled(self) -> bool:
        return bool(self.firecrawl_api_key)

    @property
    def relaymodels_enabled(self) -> bool:
        return bool(self.relaymodels_api_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.relaymodels_api_key or self.gemini_api_key)

    @property
    def yandex_mail_enabled(self) -> bool:
        return bool(
            self.yandex_email
            and self.yandex_app_password
            and self.imap_host
            and self.smtp_host
        )

    @property
    def gmail_enabled(self) -> bool:
        return self.yandex_mail_enabled

    def warn_about_missing_keys(self) -> list[str]:
        warnings: list[str] = []
        if not self.gemini_enabled:
            warnings.append(
                "RELAYMODELS_API_KEY/GEMINI_API_KEY не задан — распознавание фото, голоса и файлов выключено"
            )
        if not self.search_enabled:
            warnings.append("PERPLEXITY_API_KEY не задан — поиск поставщиков выключен")
        if not self.scrape_enabled:
            warnings.append("FIRECRAWL_API_KEY не задан — скрейп сайтов поставщиков выключен")
        if not self.yandex_mail_enabled:
            warnings.append("Яндекс Почта не настроена — отправка писем и приём ответов выключены")
        for text in warnings:
            logger.warning("%s", text)
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки читаются один раз за процесс."""
    return Settings()  # type: ignore[call-arg]
