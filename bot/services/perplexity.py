"""Поиск поставщиков через Perplexity Agent API.

Search v3: широкий поиск -> жёсткая последующая проверка Firecrawl. Модельные
кандидаты имеют приоритет, но подходящие citation-домены тоже сохраняются как
кандидаты для проверки, а не как автоматически подтверждённые поставщики.
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
MAX_SEARCH_PASSES = 4
MAX_CITATION_CANDIDATES = 12

NON_SUPPLIER_DOMAINS = frozenset({
    "wikipedia.org", "ru.wikipedia.org", "youtube.com", "vk.com", "ok.ru",
    "t.me", "telegram.me", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "avito.ru", "ozon.ru", "wildberries.ru", "market.yandex.ru", "aliexpress.ru",
    "made-in-china.com", "ru.made-in-china.com", "moy-zakupki.ru",
    "rusprofile.ru", "list-org.com", "zachestnyibiznes.ru", "sbis.ru",
    "roszdravnadzor.gov.ru", "zakupki.gov.ru", "consultant.ru", "garant.ru",
})

SUPPLIERS_SCHEMA_HINT = """Формат ответа (только JSON):
{"suppliers":[{"name":"...","site":"https://...","email":"...","phone":"...","note":"почему релевантен"}]}
В suppliers включай производителей, официальных дистрибьюторов и вероятных российских
продавцов именно этого изделия. Не включай маркетплейсы, каталоги, реестры и справочники.
Если связь вероятна, но не доказана, можешь включить компанию и явно написать это в note:
страница будет отдельно проверена Firecrawl. Ничего не выдумывай."""


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
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"].strip())
    return "\n".join(x for x in parts if x)


def _agent_citations(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw in payload.get("citations") or []:
        url = str(raw or "").strip()
        if url and url not in seen:
            seen.add(url); urls.append(url)
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "search_results":
            continue
        for result in item.get("results") or []:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            if url and url not in seen:
                seen.add(url); urls.append(url)
    return urls


def _merge_suppliers(groups: list[list[FoundSupplier]]) -> list[FoundSupplier]:
    merged: dict[str, FoundSupplier] = {}
    order: list[str] = []
    for group in groups:
        for supplier in group:
            key = domain_of(supplier.site) if supplier.site else supplier.name.strip().lower()
            if not key:
                continue
            old = merged.get(key)
            if old is None:
                merged[key] = supplier; order.append(key); continue
            if not old.email and supplier.email: old.email = supplier.email
            if not old.phone and supplier.phone: old.phone = supplier.phone
            if not old.site and supplier.site: old.site = supplier.site
            if supplier.note and supplier.note not in old.note:
                old.note = "; ".join(x for x in (old.note, supplier.note) if x)
    return [merged[key] for key in order]


def _citation_candidates(citations: list[str], known: list[FoundSupplier]) -> list[FoundSupplier]:
    """Citation — только сырой кандидат. Подтверждение делает Firecrawl/pipeline."""
    seen = {domain_of(x.site) for x in known if x.site}
    out: list[FoundSupplier] = []
    for url in citations:
        if not is_supplier_domain(url):
            continue
        domain = domain_of(url)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        out.append(FoundSupplier(
            name=domain,
            site=url,
            note="поисковый источник; требует проверки страницы товара",
            source_url=url,
        ))
        if len(out) >= MAX_CITATION_CANDIDATES:
            break
    return out


class PerplexityService:
    def __init__(self) -> None:
        settings = get_settings(); self._settings = settings
        self._client = ApiClient("perplexity", headers={"Authorization": f"Bearer {settings.perplexity_api_key}", "Content-Type": "application/json"}, timeout_read=90.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _instruction(self) -> str:
        return (Path(self._settings.prompts_dir) / "search.md").read_text(encoding="utf-8")

    async def _search_once(self, query: str, *, request_id: int | None, model: str, pass_name: str) -> SearchOutcome:
        agent_model = f"perplexity/{model}" if model and "/" not in model else model
        result = await self._client.post(API_URL, operation=f"search.{pass_name}", request_id=request_id, cost_usd=pricing.flat_cost("perplexity"), json={
            "model": agent_model or DEFAULT_MODEL,
            "instructions": self._instruction() + "\n" + SUPPLIERS_SCHEMA_HINT,
            "input": query,
            "tools": [{"type": "web_search"}],
        })
        if not result.ok:
            return SearchOutcome(query=query, error=result.error or "Perplexity недоступен")
        payload = result.json or {}; content = _agent_text(payload); citations = _agent_citations(payload)
        suppliers = _parse_suppliers(content) if content else []
        logger.info("Perplexity %s: кандидатов %s, источников %s", pass_name, len(suppliers), len(citations), extra=log_extra(request_id))
        return SearchOutcome(suppliers=suppliers, citations=citations, query=query, error=None if content or citations else "пустой ответ Perplexity")

    async def find_suppliers(self, product: str, *, requirements: list[str] | None = None, ru_number: str | None = None, holder: str | None = None, request_id: int | None = None, model: str = DEFAULT_MODEL) -> SearchOutcome:
        if not self._settings.search_enabled:
            return SearchOutcome(error="PERPLEXITY_API_KEY не задан")
        extras = f" Требования: {'; '.join(requirements)}." if requirements else ""
        holder_hint = f" Держатель РУ: {holder}." if holder else ""
        queries: list[tuple[str, str]] = [
            ("official", f'Россия: производитель, официальный сайт, дилеры и дистрибьюторы "{product}".{holder_hint}{extras}'),
            ("commercial", f'Купить "{product}" Россия поставщик продавец медицинское изделие прайс КП дилер дистрибьютор.{extras}'),
            ("exact", f'"{product}" поставщик OR дилер OR дистрибьютор OR купить Россия.{holder_hint}'),
        ]
        if ru_number:
            queries.append(("ru", f'"{ru_number}" "{product}" купить поставщик дилер дистрибьютор Россия. Не показывай реестры и госзакупки.'))
        outcomes = await asyncio.gather(*[self._search_once(q, request_id=request_id, model=model, pass_name=n) for n, q in queries[:MAX_SEARCH_PASSES]])
        good = [x for x in outcomes if x.ok]
        if not good:
            return SearchOutcome(query=" | ".join(q for _, q in queries), error="; ".join(x.error or "ошибка" for x in outcomes))
        model_suppliers = _merge_suppliers([x.suppliers for x in good])
        citations: list[str] = []; seen: set[str] = set()
        for outcome in good:
            for url in outcome.citations:
                if url not in seen:
                    seen.add(url); citations.append(url)
        raw_candidates = _citation_candidates(citations, model_suppliers)
        suppliers = _merge_suppliers([model_suppliers, raw_candidates])
        logger.info("Perplexity Search v3: «%s», подтверждённых моделью %s, сырых источников %s, всего на проверку %s", product, len(model_suppliers), len(raw_candidates), len(suppliers), extra=log_extra(request_id))
        return SearchOutcome(suppliers=suppliers, citations=citations, query=" | ".join(q for _, q in queries[:MAX_SEARCH_PASSES]))


def _parse_suppliers(content: str) -> list[FoundSupplier]:
    suppliers: list[FoundSupplier] = []; seen: set[str] = set()
    parsed = parse_llm_json(content); rows = parsed.get("suppliers", []) if isinstance(parsed, dict) else []
    for row in rows:
        if not isinstance(row, dict): continue
        site = str(row.get("site") or "").strip(); name = str(row.get("name") or "").strip()
        if not name or (site and not is_supplier_domain(site)): continue
        key = domain_of(site) if site else name.lower()
        if key in seen: continue
        seen.add(key); email = str(row.get("email") or "").strip()
        suppliers.append(FoundSupplier(name=name, site=site, email=email if is_contact_email(email) else "", phone=str(row.get("phone") or "").strip(), note=str(row.get("note") or "").strip(), source_url=site))
    return suppliers


_service: PerplexityService | None = None

def get_perplexity_service() -> PerplexityService:
    global _service
    if _service is None: _service = PerplexityService()
    return _service

async def close_perplexity_service() -> None:
    global _service
    if _service is not None: await _service.aclose()
    _service = None
