"""Поиск поставщиков через Perplexity Agent API.

Вызов прямой, не через Membrane. Ответ приходит как typed ``output``:
сообщение модели содержит итоговый текст, а ``search_results`` — источники,
из которых добираются кандидаты для последующей проверки Firecrawl.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bot.config import get_settings
from bot.logging_setup import log_extra
from bot.services import pricing
from bot.services.contacts import is_contact_email
from bot.services.domains import normalise_domain
from bot.services.http import ApiClient
from bot.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

# Perplexity migrated Sonar workloads from Chat Completions to Agent API.
API_URL = "https://api.perplexity.ai/v1/agent"
DEFAULT_MODEL = "perplexity/sonar"

# Домены, которые поиск возвращает постоянно и которые поставщиками не являются:
# агрегаторы, маркетплейсы, справочники, соцсети. Их отсекаем на входе.
NON_SUPPLIER_DOMAINS = frozenset(
    {
        "wikipedia.org",
        "ru.wikipedia.org",
        "youtube.com",
        "vk.com",
        "ok.ru",
        "t.me",
        "telegram.me",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "avito.ru",
        "ozon.ru",
        "wildberries.ru",
        "market.yandex.ru",
        "aliexpress.ru",
        "rusprofile.ru",
        "list-org.com",
        "zachestnyibiznes.ru",
        "sbis.ru",
        "roszdravnadzor.gov.ru",
        "zakupki.gov.ru",
        "consultant.ru",
        "garant.ru",
    }
)

SUPPLIERS_SCHEMA_HINT = """Формат ответа (только JSON, без пояснений вокруг):
{
  "suppliers": [
    {"name": "...", "site": "https://...", "email": "...", "phone": "...", "note": "чем занимается"}
  ]
}
Пустые поля оставляй пустой строкой."""


@dataclass(slots=True)
class FoundSupplier:
    name: str
    site: str = ""
    email: str = ""
    phone: str = ""
    note: str = ""
    source_url: str = ""


@dataclass(slots=True)
class SearchOutcome:
    suppliers: list[FoundSupplier] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    query: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def domain_of(url: str) -> str:
    """Ключ дедупликации выдачи — ровно тот же, что и у записи в базу."""
    return normalise_domain(url) or ""


def is_supplier_domain(url: str) -> bool:
    """Отсеивает агрегаторы и справочники — они не поставщики."""
    domain = domain_of(url)
    if not domain or "." not in domain:
        return False
    return not any(domain == bad or domain.endswith("." + bad) for bad in NON_SUPPLIER_DOMAINS)


def _agent_text(payload: dict[str, Any]) -> str:
    """Извлечь итоговый текст из Agent API response."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


def _agent_citations(payload: dict[str, Any]) -> list[str]:
    """Собрать URL из search_results Agent API, сохранив порядок."""
    urls: list[str] = []
    seen: set[str] = set()

    # На случай совместимого ответа/будущего alias API поддерживаем citations.
    for raw in payload.get("citations") or []:
        url = str(raw or "").strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "search_results":
            continue
        for result in item.get("results") or []:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


class PerplexityService:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = ApiClient(
            "perplexity",
            headers={
                "Authorization": f"Bearer {settings.perplexity_api_key}",
                "Content-Type": "application/json",
            },
            timeout_read=90.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _instruction(self) -> str:
        """Системная инструкция из ``prompts/search.md`` — с диска, как и остальные."""
        path = Path(self._settings.prompts_dir) / "search.md"
        return path.read_text(encoding="utf-8")

    async def find_suppliers(
        self,
        product: str,
        *,
        requirements: list[str] | None = None,
        request_id: int | None = None,
        model: str = DEFAULT_MODEL,
    ) -> SearchOutcome:
        """Найти поставщиков изделия через Perplexity Agent API."""
        settings = self._settings
        if not settings.search_enabled:
            return SearchOutcome(error="PERPLEXITY_API_KEY не задан")

        extras = f" Дополнительные требования: {'; '.join(requirements)}." if requirements else ""
        query = (
            f"Российские поставщики и дистрибьюторы медицинского изделия: {product}.{extras} "
            "Нужны названия компаний, их официальные сайты и контакты."
        )

        # Старый код мог передать короткое имя sonar. Agent API ожидает namespace.
        agent_model = f"perplexity/{model}" if model and "/" not in model else model

        result = await self._client.post(
            API_URL,
            operation="search",
            request_id=request_id,
            cost_usd=pricing.flat_cost("perplexity"),
            json={
                "model": agent_model or DEFAULT_MODEL,
                "instructions": self._instruction() + "\n" + SUPPLIERS_SCHEMA_HINT,
                "input": query,
                "tools": [
                    {
                        "type": "web_search",
                        "filters": {"search_recency_filter": "year"},
                    }
                ],
            },
        )
        if not result.ok:
            return SearchOutcome(query=query, error=result.error or "Perplexity недоступен")

        payload = result.json or {}
        content = _agent_text(payload)
        citations = _agent_citations(payload)
        if not content:
            return SearchOutcome(query=query, error="пустой ответ Perplexity")

        suppliers = _parse_suppliers(content, citations)
        logger.info(
            "Perplexity: по «%s» кандидатов %s, источников %s",
            product,
            len(suppliers),
            len(citations),
            extra=log_extra(request_id),
        )
        return SearchOutcome(suppliers=suppliers, citations=citations, query=query)


def _parse_suppliers(content: str, citations: list[str]) -> list[FoundSupplier]:
    """Разбор ответа модели плюс добор из списка источников.

    Модель иногда возвращает JSON, иногда прозу вокруг него, иногда только
    ссылки. Берём что удалось разобрать, а недостающие домены добираем из
    citations — они приходят структурированно и врать не умеют.
    """
    suppliers: list[FoundSupplier] = []
    seen: set[str] = set()

    parsed = parse_llm_json(content)
    rows = parsed.get("suppliers", []) if isinstance(parsed, dict) else []

    for row in rows:
        if not isinstance(row, dict):
            continue
        site = str(row.get("site") or "").strip()
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if site and not is_supplier_domain(site):
            continue
        key = domain_of(site) if site else name.lower()
        if key in seen:
            continue
        seen.add(key)
        email = str(row.get("email") or "").strip()
        suppliers.append(
            FoundSupplier(
                name=name,
                site=site,
                email=email if is_contact_email(email) else "",
                phone=str(row.get("phone") or "").strip(),
                note=str(row.get("note") or "").strip(),
                source_url=site,
            )
        )

    # Источники, которых модель не назвала явно, но которые похожи на сайты
    # поставщиков, тоже стоит проверить — их обойдёт Firecrawl.
    for url in citations:
        if not is_supplier_domain(url):
            continue
        key = domain_of(url)
        if key in seen:
            continue
        seen.add(key)
        suppliers.append(FoundSupplier(name=key, site=url, source_url=url))

    return suppliers


_service: PerplexityService | None = None


def get_perplexity_service() -> PerplexityService:
    global _service
    if _service is None:
        _service = PerplexityService()
    return _service


async def close_perplexity_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None
