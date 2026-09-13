"""Поиск поставщиков через Perplexity Agent API.

Search v2 использует несколько независимых поисковых проходов: официальный
производитель/дистрибьюторы, коммерческие продавцы и, если известно, точный
номер РУ. Результаты объединяются по домену до проверки Firecrawl.
"""

from __future__ import annotations

import asyncio
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

API_URL = "https://api.perplexity.ai/v1/agent"
DEFAULT_MODEL = "perplexity/sonar"
MAX_SEARCH_PASSES = 3

NON_SUPPLIER_DOMAINS = frozenset(
    {
        "wikipedia.org", "ru.wikipedia.org", "youtube.com", "vk.com", "ok.ru",
        "t.me", "telegram.me", "facebook.com", "instagram.com", "twitter.com", "x.com",
        "avito.ru", "ozon.ru", "wildberries.ru", "market.yandex.ru", "aliexpress.ru",
        "made-in-china.com", "ru.made-in-china.com", "moy-zakupki.ru",
        "rusprofile.ru", "list-org.com", "zachestnyibiznes.ru", "sbis.ru",
        "roszdravnadzor.gov.ru", "zakupki.gov.ru", "consultant.ru", "garant.ru",
    }
)

SUPPLIERS_SCHEMA_HINT = """Формат ответа (только JSON, без пояснений вокруг):
{
  "suppliers": [
    {"name": "...", "site": "https://...", "email": "...", "phone": "...", "note": "почему это релевантный поставщик"}
  ]
}
Пустые поля оставляй пустой строкой.
В suppliers включай только реальные компании-производители, официальных дистрибьюторов
или продавцов, у которых можно запросить/купить именно указанное изделие. Не включай
маркетплейсы, каталоги, агрегаторы закупок, справочники и просто информационные источники.
Не добавляй компанию только потому, что её сайт встретился среди результатов поиска:
должна быть связь с конкретным изделием, брендом/моделью либо номером РУ."""


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
    return normalise_domain(url) or ""


def is_supplier_domain(url: str) -> bool:
    domain = domain_of(url)
    if not domain or "." not in domain:
        return False
    return not any(domain == bad or domain.endswith("." + bad) for bad in NON_SUPPLIER_DOMAINS)


def _agent_text(payload: dict[str, Any]) -> str:
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
    urls: list[str] = []
    seen: set[str] = set()
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


def _merge_suppliers(groups: list[list[FoundSupplier]]) -> list[FoundSupplier]:
    """Дедупликация по домену с сохранением наиболее полных контактов."""
    merged: dict[str, FoundSupplier] = {}
    order: list[str] = []
    for group in groups:
        for supplier in group:
            key = domain_of(supplier.site) if supplier.site else supplier.name.strip().lower()
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = supplier
                order.append(key)
                continue
            if not existing.email and supplier.email:
                existing.email = supplier.email
            if not existing.phone and supplier.phone:
                existing.phone = supplier.phone
            if not existing.site and supplier.site:
                existing.site = supplier.site
            if not existing.source_url and supplier.source_url:
                existing.source_url = supplier.source_url
            if supplier.note and supplier.note not in existing.note:
                existing.note = "; ".join(x for x in (existing.note, supplier.note) if x)
    return [merged[key] for key in order]


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
        path = Path(self._settings.prompts_dir) / "search.md"
        return path.read_text(encoding="utf-8")

    async def _search_once(
        self,
        query: str,
        *,
        request_id: int | None,
        model: str,
        pass_name: str,
    ) -> SearchOutcome:
        agent_model = f"perplexity/{model}" if model and "/" not in model else model
        result = await self._client.post(
            API_URL,
            operation=f"search.{pass_name}",
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
            return SearchOutcome(query=query, citations=citations, error="пустой ответ Perplexity")
        suppliers = _parse_suppliers(content)
        logger.info(
            "Perplexity %s: кандидатов %s, источников %s",
            pass_name,
            len(suppliers),
            len(citations),
            extra=log_extra(request_id),
        )
        return SearchOutcome(suppliers=suppliers, citations=citations, query=query)

    async def find_suppliers(
        self,
        product: str,
        *,
        requirements: list[str] | None = None,
        ru_number: str | None = None,
        holder: str | None = None,
        request_id: int | None = None,
        model: str = DEFAULT_MODEL,
    ) -> SearchOutcome:
        """Search v2: несколько поисковых стратегий и объединение результатов."""
        if not self._settings.search_enabled:
            return SearchOutcome(error="PERPLEXITY_API_KEY не задан")

        extras = f" Дополнительные требования: {'; '.join(requirements)}." if requirements else ""
        holder_hint = f" Держатель/производитель РУ: {holder}." if holder else ""

        queries: list[tuple[str, str]] = [
            (
                "official",
                f'Найди в России производителя, официальный сайт, официальных дистрибьюторов и дилеров медицинского изделия "{product}".'
                f"{holder_hint}{extras} Нужны только компании, реально связанные с этим товаром, с сайтами и контактами.",
            ),
            (
                "commercial",
                f'Найди российских продавцов и поставщиков, у которых можно купить или запросить КП на медицинское изделие "{product}".'
                f" Ищи также по сочетаниям: купить, поставщик, дилер, дистрибьютор, прайс, коммерческое предложение.{extras}",
            ),
        ]
        if ru_number:
            queries.append(
                (
                    "ru",
                    f'Найди российские компании и страницы товаров, где указан регистрационный номер "{ru_number}" для изделия "{product}".'
                    f"{holder_hint} Нужны реальные продавцы/дистрибьюторы, а не реестры, закупки и справочники.",
                )
            )

        queries = queries[:MAX_SEARCH_PASSES]
        outcomes = await asyncio.gather(
            *[
                self._search_once(
                    query,
                    request_id=request_id,
                    model=model,
                    pass_name=pass_name,
                )
                for pass_name, query in queries
            ]
        )

        good = [outcome for outcome in outcomes if outcome.ok]
        if not good:
            errors = "; ".join(outcome.error or "ошибка поиска" for outcome in outcomes)
            return SearchOutcome(query=" | ".join(q for _, q in queries), error=errors)

        suppliers = _merge_suppliers([outcome.suppliers for outcome in good])
        citations: list[str] = []
        seen_citations: set[str] = set()
        for outcome in good:
            for url in outcome.citations:
                if url not in seen_citations:
                    seen_citations.add(url)
                    citations.append(url)

        logger.info(
            "Perplexity Search v2: по «%s» проходов %s, уникальных кандидатов %s, источников %s",
            product,
            len(good),
            len(suppliers),
            len(citations),
            extra=log_extra(request_id),
        )
        return SearchOutcome(
            suppliers=suppliers,
            citations=citations,
            query=" | ".join(q for _, q in queries),
        )


def _parse_suppliers(content: str) -> list[FoundSupplier]:
    """Берём только поставщиков, которых модель явно включила в JSON.

    Citation-only домены больше не превращаются автоматически в кандидатов: это
    был главный источник случайных магазинов и информационных сайтов.
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
