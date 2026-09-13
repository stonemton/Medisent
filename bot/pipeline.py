"""Оркестрация подбора: поиск → скрейп → реестр → кандидаты → отчёт.

Оркестратор — код бота. Ни n8n, ни CRM в схеме нет.

Проверка в реестре идёт по изделию один раз. После обхода сайтов отдельный
детерминированный шаг проверяет, можно ли связать товар каждого поставщика
именно с найденным РУ. Это не одно и то же.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from bot.db import repo
from bot.db.models import RegistryState, RequestStatus
from bot.db.repo import CandidateInput, SupplierInput
from bot.db.session import session_scope
from bot.logging_setup import log_extra
from bot.services import budget, guard
from bot.services import criteria as criteria_service
from bot.services.firecrawl import ScrapeResult, get_firecrawl_service
from bot.services.perplexity import get_perplexity_service
from bot.services.registry import RegistryResult, get_registry_service
from bot.services.report import CandidateView, Report, rank_candidates

logger = logging.getLogger(__name__)

MAX_SITES_TO_SCRAPE = 10
PAGE_RU_RE = re.compile(
    r"\b(?:РЗН|ФСР|ФСЗ)\s*(?:№\s*)?\d{4}/\d+(?:[-/]\d+)?\b",
    re.I | re.UNICODE,
)
_GENERIC_MATCH_TERMS = {
    "степлер", "кожный", "одноразовый", "одноразовая", "стерильный", "стерильная",
    "изделие", "медицинский", "медицинское", "набор", "система", "инструмент",
    "аппарат", "устройство", "скоба", "скобы", "скобами", "штук", "упаковка",
}


@dataclass(slots=True)
class SearchSummary:
    total_found: int = 0
    blacklisted: int = 0
    scraped: int = 0
    registry_state: str = RegistryState.UNAVAILABLE
    unrega_state: str = RegistryState.UNAVAILABLE
    errors: list[str] = field(default_factory=list)
    search_failed: bool = False
    budget_exceeded: bool = False


async def run_search(
    *,
    request_id: int,
    product: str,
    requirements: list[str],
) -> SearchSummary:
    extra = log_extra(request_id)
    summary = SearchSummary()
    registry_service = get_registry_service()

    # Search v2 сначала получает официальные признаки изделия из реестра, а затем
    # использует их как поисковые якоря: номер РУ и держателя/производителя.
    registry = await registry_service.check_product(
        product, request_id=request_id, cache=True
    )
    best = registry.best
    search = await get_perplexity_service().find_suppliers(
        product,
        requirements=requirements,
        ru_number=best.ru_number if best else None,
        holder=best.holder if best else None,
        request_id=request_id,
    )

    summary.registry_state = registry.state
    if not search.ok:
        summary.search_failed = True
        summary.budget_exceeded = budget.exceeded(request_id)
        summary.errors.append(search.error or "поиск не удался")
        logger.warning("Поиск не удался: %s", search.error, extra=extra)
        return summary
    if registry.unavailable:
        summary.errors.append("реестр недоступен")

    summary.total_found = len(search.suppliers)
    if not search.suppliers:
        return summary

    tainted_suppliers: set[str] = set()
    supplier_inputs: list[SupplierInput] = []
    for item in search.suppliers:
        screening = await guard.screen_third_party_async(
            f"{item.name} {item.note}", source="выдача поиска"
        )
        if screening.suspicious:
            tainted_suppliers.add(item.site or item.name)
            logger.warning(
                "Подозрительное название поставщика «%s»: %s",
                item.name,
                screening.summary,
                extra=extra,
            )
        supplier_inputs.append(
            SupplierInput(
                name=item.name,
                domain=item.site or None,
                email=item.email or None,
                phone=item.phone or None,
                found_via=search.query[:500],
            )
        )
    async with session_scope() as session:
        key_to_id = await repo.upsert_suppliers(session, supplier_inputs)

    to_scrape = [item for item in search.suppliers if item.site][:MAX_SITES_TO_SCRAPE]
    scrape_task = (
        get_firecrawl_service().scrape_many(
            [item.site for item in to_scrape], product, request_id=request_id
        )
        if to_scrape
        else _nothing()
    )
    unrega_task = registry_service.check_unrega(
        product, holder=best.holder if best else None, request_id=request_id
    )
    results, unrega = await asyncio.gather(scrape_task, unrega_task)
    scrapes: dict[str, ScrapeResult] = {result.url: result for result in results}
    summary.scraped = sum(1 for r in results if r.ok)
    summary.unrega_state = unrega.state
    if unrega.unavailable:
        summary.errors.append("информационные письма не проверены")

    candidates: list[CandidateInput] = []
    for item in search.suppliers:
        supplier_id = _resolve_supplier_id(item, key_to_id)
        if supplier_id is None:
            continue
        scrape = scrapes.get(item.site) if item.site else None
        site_ru_match, match_basis = _site_registry_match(product, scrape, best)
        candidates.append(
            CandidateInput(
                supplier_id=supplier_id,
                site_claims=scrape.claims_stock if scrape and scrape.ok else None,
                site_url=item.site or None,
                site_price=scrape.price if scrape and scrape.ok else None,
                ru_number=best.ru_number if best else None,
                ru_holder=best.holder if best else None,
                ru_valid=best.valid if best else None,
                ru_registry=best.registry if best else None,
                ru_checked_at=None if registry.unavailable else registry.checked_at,
                unrega_flags=_unrega_flags(
                    unrega,
                    ru_site_match=site_ru_match,
                    ru_match_basis=match_basis,
                ),
                raw={
                    "registry": registry.as_payload(),
                    "registry_match": {
                        "site_matches_ru": site_ru_match,
                        "basis": match_basis,
                    },
                    "search": {"note": item.note, "source": item.source_url},
                    "scrape": {
                        "ok": bool(scrape and scrape.ok),
                        "error": scrape.error if scrape else None,
                        "injection_suspected": bool(
                            (scrape and scrape.injection_suspected)
                            or (item.site or item.name) in tainted_suppliers
                        ),
                    },
                },
            )
        )

    async with session_scope() as session:
        if candidates:
            await repo.upsert_candidates(session, request_id, candidates)
        await _enrich_contacts(session, search.suppliers, scrapes, key_to_id)
        summary.blacklisted = await repo.count_blacklisted_in_request(session, request_id)
        await repo.transition(session, request_id, RequestStatus.REPORT)

    summary.budget_exceeded = budget.exceeded(request_id)
    logger.info(
        "Подбор: найдено %s, обойдено сайтов %s, реестр %s, письма %s, отсеяно чёрным списком %s",
        summary.total_found,
        summary.scraped,
        summary.registry_state,
        summary.unrega_state,
        summary.blacklisted,
        extra=extra,
    )
    return summary


async def close_request(session: AsyncSession, request_id: int) -> bool:
    closed = await repo.transition(session, request_id, RequestStatus.CLOSED)
    budget.forget(request_id)
    return closed


async def _nothing() -> list[ScrapeResult]:
    return []


def _normalise_ru(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]", "", (value or "").lower())


def _distinctive_terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{4,}", (text or "").lower())
    return {
        word
        for word in words
        if word not in _GENERIC_MATCH_TERMS and not word.isdigit()
    }


def _site_registry_match(
    product: str,
    scrape: ScrapeResult | None,
    best: Any,
) -> tuple[bool | None, str]:
    """Связать товар конкретного поставщика с найденным РУ без догадок."""
    if best is None or not getattr(best, "ru_number", None):
        return None, "РУ на изделие не найдено"
    if scrape is None or not scrape.ok or not scrape.markdown:
        return None, "страница поставщика не проверена"

    page_text = scrape.markdown.lower()
    expected_ru = _normalise_ru(str(best.ru_number))
    page_compact = _normalise_ru(page_text)
    if expected_ru and expected_ru in page_compact:
        return True, "точный номер РУ найден на странице поставщика"

    page_ru_numbers = {_normalise_ru(match.group(0)) for match in PAGE_RU_RE.finditer(scrape.markdown)}
    page_ru_numbers.discard("")
    if page_ru_numbers and expected_ru not in page_ru_numbers:
        return False, "на странице поставщика указан другой номер РУ"

    raw = getattr(best, "raw", {}) or {}
    raw_text = raw.get("text") if isinstance(raw, dict) else ""
    registry_text = " ".join(
        str(value or "")
        for value in (
            getattr(best, "product_name", None),
            getattr(best, "holder", None),
            raw_text,
        )
    ).lower()
    distinct = _distinctive_terms(product)
    strong = {term for term in distinct if term in registry_text}
    matched = {term for term in strong if term in page_text}
    if matched:
        return True, "совпал отличительный бренд/модель: " + ", ".join(sorted(matched)[:3])

    return None, "на странице поставщика недостаточно данных для привязки к РУ"


def _unrega_flags(
    unrega: RegistryResult,
    *,
    ru_site_match: bool | None,
    ru_match_basis: str,
) -> dict[str, Any]:
    return {
        "state": unrega.state,
        "items": [r.product_name or r.status_text or "письмо" for r in unrega.records],
        "errors": unrega.errors,
        "ru_site_match": ru_site_match,
        "ru_match_basis": ru_match_basis,
    }


def _resolve_supplier_id(item: object, key_to_id: dict[str, int]) -> int | None:
    site = getattr(item, "site", "") or ""
    name = getattr(item, "name", "") or ""
    if site:
        domain = repo.normalise_domain(site)
        if domain and domain in key_to_id:
            return key_to_id[domain]
    return key_to_id.get(name.strip())


async def _enrich_contacts(
    session: AsyncSession,
    suppliers: Sequence[object],
    scrapes: dict[str, ScrapeResult],
    key_to_id: dict[str, int],
) -> None:
    updates: list[SupplierInput] = []
    for item in suppliers:
        site = getattr(item, "site", "") or ""
        scrape = scrapes.get(site)
        if not scrape or not scrape.ok:
            continue
        if not scrape.email and not scrape.phone:
            continue
        updates.append(
            SupplierInput(
                name=str(getattr(item, "name", "")),
                domain=site or None,
                email=scrape.email,
                phone=scrape.phone,
            )
        )
    if updates:
        await repo.upsert_suppliers(session, updates)


async def build_report(
    session: AsyncSession,
    *,
    request_id: int,
    product: str,
    qty: str,
    requirements: list[str],
) -> Report:
    request = await repo.get_request(session, request_id)
    token = request.token if request else "?"

    rows = await repo.list_candidates_for_report(session, request_id)
    views = [
        CandidateView(
            candidate_id=int(row.id),
            supplier_id=int(row.supplier_id or 0),
            supplier_name=str(row.supplier_name),
            domain=row.domain,
            email=row.email,
            phone=row.phone,
            site_claims=row.site_claims,
            site_url=row.site_url,
            site_price=row.site_price if isinstance(row.site_price, Decimal) else None,
            ru_number=row.ru_number,
            ru_holder=row.ru_holder,
            ru_valid=row.ru_valid,
            ru_registry=row.ru_registry,
            registry_state=_registry_state(row),
            ru_site_match=_ru_site_match(row.unrega_flags),
            ru_match_basis=_ru_match_basis(row.unrega_flags),
            unrega_flags=_unrega_items(row.unrega_flags),
            injection_suspected=bool(row.injection_suspected),
        )
        for row in rows
    ]
    unrega_state = _unrega_state(rows)

    known_criteria = await criteria_service.for_prompt(session)
    ordered, summary, missing, failed = await rank_candidates(
        views,
        product=product,
        qty=qty,
        requirements=requirements,
        criteria=known_criteria,
        request_id=request_id,
        unrega_state=unrega_state,
    )

    await repo.set_candidate_ranks(
        session, {view.candidate_id: index for index, view in enumerate(ordered, start=1)}
    )
    await repo.transition(session, request_id, RequestStatus.AWAITING_CHOICE)
    return Report(
        request_token=token,
        product=product,
        candidates=ordered,
        summary=summary,
        missing_data=missing,
        llm_failed=failed,
        unrega_state=unrega_state,
    )


def _registry_state(row: object) -> str:
    state = getattr(row, "registry_state", None)
    if state in (RegistryState.FOUND, RegistryState.NOT_FOUND, RegistryState.UNAVAILABLE):
        return str(state)
    if getattr(row, "ru_number", None):
        return RegistryState.FOUND
    return RegistryState.UNAVAILABLE


def _unrega_items(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(item) for item in value.get("items", []) or []]
    return []


def _ru_site_match(value: object) -> bool | None:
    if isinstance(value, dict):
        match = value.get("ru_site_match")
        return match if isinstance(match, bool) else None
    return None


def _ru_match_basis(value: object) -> str | None:
    if isinstance(value, dict):
        basis = value.get("ru_match_basis")
        return str(basis) if basis else None
    return None


def _unrega_state(rows: Sequence[Any]) -> str:
    for row in rows:
        flags = getattr(row, "unrega_flags", None)
        if isinstance(flags, dict) and flags.get("state") in (
            RegistryState.FOUND,
            RegistryState.NOT_FOUND,
            RegistryState.UNAVAILABLE,
        ):
            return str(flags["state"])
    return RegistryState.UNAVAILABLE if rows else RegistryState.NOT_FOUND
