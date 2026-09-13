"""Поиск поставщиков через Perplexity Agent API.

Search v4: исследовательский цикл. Сначала несколько независимых поисковых раундов
по изделию, модели, РУ и держателю; затем при слабой выдаче — уточняющий поиск с
учётом уже найденных доменов. Любой результат остаётся кандидатом до проверки
страницы Firecrawl и Supplier Gate в pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import re
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
MAX_SEARCH_PASSES = 6
MAX_CITATION_CANDIDATES = 16
MIN_MODEL_SUPPLIERS_BEFORE_STOP = 5
MIN_TOTAL_CANDIDATES_BEFORE_STOP = 8

NON_SUPPLIER_DOMAINS = frozenset({
    "wikipedia.org", "ru.wikipedia.org", "youtube.com", "vk.com", "ok.ru",
    "t.me", "telegram.me", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "avito.ru", "ozon.ru", "wildberries.ru", "market.yandex.ru", "aliexpress.ru",
    "made-in-china.com", "ru.made-in-china.com", "moy-zakupki.ru",
    "rusprofile.ru", "list-org.com", "zachestnyibiznes.ru", "sbis.ru",
    "roszdravnadzor.gov.ru", "zakupki.gov.ru", "consultant.ru", "garant.ru",
})

SUPPLIERS_SCHEMA_HINT = """Формат ответа (только JSON):
{"suppliers":[{"name":"...","site":"https://...","email":"...","phone":"...","note":"почему релевантен и какое доказательство найдено"}]}
В suppliers включай производителей, официальных дистрибьюторов и вероятных российских
продавцов именно этого изделия. Для каждого кандидата ищи фактическую связь с товаром:
точную модель/артикул, бренд, номер РУ, карточку товара, прайс, каталог производителя,
страницу дилера или коммерческий контакт. Не включай маркетплейсы, каталоги общего
назначения, реестры, справочники и госзакупки как поставщиков. Такие страницы можно
использовать только как след для поиска настоящего сайта компании.
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
    merged: dict[str, FoundSupplier] = {}
    order: list[str] = []
    for group in groups:
        for supplier in group:
            key = domain_of(supplier.site) if supplier.site else supplier.name.strip().lower()
            if not key:
                continue
            old = merged.get(key)
            if old is None:
                merged[key] = supplier
                order.append(key)
                continue
            if not old.email and supplier.email:
                old.email = supplier.email
            if not old.phone and supplier.phone:
                old.phone = supplier.phone
            if not old.site and supplier.site:
                old.site = supplier.site
            if supplier.note and supplier.note not in old.note:
                old.note = "; ".join(x for x in (old.note, supplier.note) if x)
            if not old.source_url and supplier.source_url:
                old.source_url = supplier.source_url
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


def _product_identifiers(product: str) -> list[str]:
    """Достаёт артикулы/модели для отдельных точных поисков.

    Идентификатором считаем токен с цифрой и буквой/дефисом, например QPWB-35N,
    MPK165P.3 или 0301-01M. Чистые годы и количества не используем.
    """
    found: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9._/-]{2,}", product):
        token = raw.strip(".,;:()[]{}")
        if not any(ch.isdigit() for ch in token):
            continue
        if token.isdigit() or re.fullmatch(r"20\d{2}", token):
            continue
        if not any(ch.isalpha() for ch in token) and "-" not in token and "/" not in token:
            continue
        key = token.lower()
        if key not in seen:
            seen.add(key)
            found.append(token)
    return found[:4]


def _initial_queries(
    product: str,
    *,
    requirements: list[str] | None,
    ru_number: str | None,
    holder: str | None,
) -> list[tuple[str, str]]:
    extras = f" Требования: {'; '.join(requirements)}." if requirements else ""
    holder_hint = f" Держатель РУ: {holder}." if holder else ""
    queries: list[tuple[str, str]] = [
        (
            "official",
            f'Исследуй российский рынок по изделию "{product}". Найди производителя, '
            f'официальный сайт, держателя РУ, официальных дилеров и дистрибьюторов. '
            f'Для каждой компании найди прямое доказательство связи с изделием.{holder_hint}{extras}',
        ),
        (
            "commercial",
            f'Найди в России реальные страницы продажи или поставки "{product}": карточки товара, '
            f'прайсы, каталоги дилеров, страницы "запросить цену", коммерческие контакты. '
            f'Не используй маркетплейсы, реестры и госзакупки как поставщиков.{extras}',
        ),
        (
            "exact",
            f'Точный веб-поиск по "{product}": поставщик OR продавец OR дилер OR дистрибьютор OR '
            f'производитель OR "запросить цену" Россия.{holder_hint}',
        ),
    ]
    identifiers = _product_identifiers(product)
    if identifiers:
        quoted = " ".join(f'"{x}"' for x in identifiers)
        queries.append((
            "identifier",
            f'Найди российские сайты, где встречаются точные модель/артикул {quoted}. '
            f'Ищи производителя, дилеров, дистрибьюторов, карточки товара и прайсы. '
            f'Сопоставь найденное именно с изделием "{product}".',
        ))
    if ru_number:
        queries.append((
            "ru",
            f'По номеру РУ "{ru_number}" найди коммерческую цепочку изделия "{product}": '
            f'держателя/производителя, официальный сайт, дилеров и продавцов в России. '
            f'Сам реестр не считай поставщиком.{holder_hint}',
        ))
    if holder:
        queries.append((
            "holder",
            f'Исследуй компанию "{holder}" в связи с изделием "{product}". Найди её официальный '
            f'сайт, продуктовую страницу, российских партнёров, дилеров и дистрибьюторов. '
            f'Не делай вывод о продаже без источника.',
        ))
    return queries[:MAX_SEARCH_PASSES]


def _refinement_query(
    product: str,
    *,
    known: list[FoundSupplier],
    ru_number: str | None,
    holder: str | None,
) -> str:
    domains = [domain_of(x.site) for x in known if x.site]
    excluded = ", ".join(x for x in domains[:12] if x)
    context: list[str] = []
    if ru_number:
        context.append(f"РУ {ru_number}")
    if holder:
        context.append(f"держатель {holder}")
    identifiers = _product_identifiers(product)
    if identifiers:
        context.append("модель/артикул " + ", ".join(identifiers))
    context_text = "; ".join(context) or "точное название изделия"
    exclude_text = f" Уже найдены домены: {excluded}. Не повторяй их." if excluded else ""
    return (
        f'Сделай второй исследовательский проход по изделию "{product}". Используй как опорные '
        f'признаки: {context_text}. Ищи альтернативных российских поставщиков через страницы '
        f'производителя, списки официальных партнёров, каталоги дилеров, точные артикулы, прайсы '
        f'и страницы товара. Перепроверяй, что компания действительно связана с этим изделием.'
        f'{exclude_text}'
    )


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
        return (Path(self._settings.prompts_dir) / "search.md").read_text(encoding="utf-8")

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
                "tools": [{"type": "web_search"}],
            },
        )
        if not result.ok:
            return SearchOutcome(query=query, error=result.error or "Perplexity недоступен")
        payload = result.json or {}
        content = _agent_text(payload)
        citations = _agent_citations(payload)
        suppliers = _parse_suppliers(content) if content else []
        logger.info(
            "Perplexity %s: кандидатов %s, источников %s",
            pass_name,
            len(suppliers),
            len(citations),
            extra=log_extra(request_id),
        )
        return SearchOutcome(
            suppliers=suppliers,
            citations=citations,
            query=query,
            error=None if content or citations else "пустой ответ Perplexity",
        )

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
        if not self._settings.search_enabled:
            return SearchOutcome(error="PERPLEXITY_API_KEY не задан")

        queries = _initial_queries(
            product,
            requirements=requirements,
            ru_number=ru_number,
            holder=holder,
        )
        outcomes = await asyncio.gather(*[
            self._search_once(q, request_id=request_id, model=model, pass_name=name)
            for name, q in queries
        ])
        good = [x for x in outcomes if x.ok]
        if not good:
            return SearchOutcome(
                query=" | ".join(q for _, q in queries),
                error="; ".join(x.error or "ошибка" for x in outcomes),
            )

        model_suppliers = _merge_suppliers([x.suppliers for x in good])
        citations = _merge_citations(good)
        raw_candidates = _citation_candidates(citations, model_suppliers)
        suppliers = _merge_suppliers([model_suppliers, raw_candidates])

        needs_refinement = (
            len(model_suppliers) < MIN_MODEL_SUPPLIERS_BEFORE_STOP
            or len(suppliers) < MIN_TOTAL_CANDIDATES_BEFORE_STOP
        )
        refinement_query = ""
        if needs_refinement:
            refinement_query = _refinement_query(
                product,
                known=suppliers,
                ru_number=ru_number,
                holder=holder,
            )
            refined = await self._search_once(
                refinement_query,
                request_id=request_id,
                model=model,
                pass_name="refine",
            )
            if refined.ok:
                model_suppliers = _merge_suppliers([model_suppliers, refined.suppliers])
                citations = _merge_citations(good + [refined])
                raw_candidates = _citation_candidates(citations, model_suppliers)
                suppliers = _merge_suppliers([model_suppliers, raw_candidates])

        all_queries = [q for _, q in queries]
        if refinement_query:
            all_queries.append(refinement_query)
        logger.info(
            "Perplexity Search v4: «%s», модельных %s, citation-кандидатов %s, всего на проверку %s, refinement=%s",
            product,
            len(model_suppliers),
            len(raw_candidates),
            len(suppliers),
            needs_refinement,
            extra=log_extra(request_id),
        )
        return SearchOutcome(
            suppliers=suppliers,
            citations=citations,
            query=" | ".join(all_queries),
        )


def _merge_citations(outcomes: list[SearchOutcome]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for outcome in outcomes:
        for url in outcome.citations:
            if url not in seen:
                seen.add(url)
                citations.append(url)
    return citations


def _parse_suppliers(content: str) -> list[FoundSupplier]:
    suppliers: list[FoundSupplier] = []
    seen: set[str] = set()
    parsed = parse_llm_json(content)
    rows = parsed.get("suppliers", []) if isinstance(parsed, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        site = str(row.get("site") or "").strip()
        name = str(row.get("name") or "").strip()
        if not name or (site and not is_supplier_domain(site)):
            continue
        key = domain_of(site) if site else name.lower()
        if key in seen:
            continue
        seen.add(key)
        email = str(row.get("email") or "").strip()
        suppliers.append(FoundSupplier(
            name=name,
            site=site,
            email=email if is_contact_email(email) else "",
            phone=str(row.get("phone") or "").strip(),
            note=str(row.get("note") or "").strip(),
            source_url=site,
        ))
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
