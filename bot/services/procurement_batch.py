"""Пакетный закупочный конвейер для заявок из нескольких позиций.

Каждая позиция проверяется отдельно, но если у всего списка один основной
держатель РУ, он становится якорным производителем batch. Альтернативный
производитель из другого РУ не может подменить основную товарную линию.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bot.db import repo
from bot.db.models import RequestStatus
from bot.db.repo import CandidateInput
from bot.db.session import session_scope
from bot.pipeline import build_report, close_request, run_search
from bot.services.batch_intake import ProcurementItem
from bot.services.registry import get_registry_service
from bot.services.report import Report

logger = logging.getLogger(__name__)
ROLE_ORDER = {"manufacturer": 0, "official_distributor": 1, "seller": 2, "candidate": 3}


@dataclass(slots=True)
class SplitSupplier:
    supplier_name: str
    covered_items: list[str]
    total_items: int


@dataclass(slots=True)
class BatchProcurementResult:
    report: Report | None = None
    searched_items: int = 0
    full_coverage_suppliers: int = 0
    errors: list[str] = field(default_factory=list)
    split_plan: list[SplitSupplier] = field(default_factory=list)

    @property
    def needs_split(self) -> bool:
        return self.report is None and bool(self.split_plan)


@dataclass(slots=True)
class _Hit:
    item_index: int
    item: ProcurementItem
    row: Any


def _flags(row: Any) -> dict[str, Any]:
    value = getattr(row, "unrega_flags", None)
    return dict(value) if isinstance(value, dict) else {}


def _role(row: Any) -> str:
    return str(_flags(row).get("supplier_role") or "candidate")


def _company_key(value: str | None) -> str:
    value = str(value or "").lower().replace("ё", "е")
    value = re.sub(r"\b(?:ооо|ао|зао|пао|оао|ип)\b", " ", value)
    return re.sub(r"[^a-zа-я0-9]+", "", value)


def _supplier_sort_key(hits: list[_Hit]) -> tuple[int, int, int]:
    best_role = min((ROLE_ORDER.get(_role(hit.row), 3) for hit in hits), default=3)
    has_email = any(bool(getattr(hit.row, "email", None)) for hit in hits)
    has_phone = any(bool(getattr(hit.row, "phone", None)) for hit in hits)
    return (best_role, -int(has_email), -int(has_phone))


def _uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in out:
            out.append(value)
    return out


def _merged_candidate(hits: list[_Hit], total_items: int) -> CandidateInput:
    first = hits[0].row
    flags_list = [_flags(hit.row) for hit in hits]
    _, role = min(((ROLE_ORDER.get(str(f.get("supplier_role") or "candidate"), 3), str(f.get("supplier_role") or "candidate")) for f in flags_list), default=(3, "candidate"))
    evidence_url = next((str(f.get("role_evidence_url") or "") for f in flags_list if f.get("role_evidence_url")), "")
    evidence = next((str(f.get("role_evidence") or "") for f in flags_list if f.get("role_evidence")), "")
    unrega_items: list[str] = []
    unrega_errors: list[str] = []
    for flags in flags_list:
        if isinstance(flags.get("items"), list):
            unrega_items.extend(str(x) for x in flags["items"] if str(x).strip())
        if isinstance(flags.get("errors"), list):
            unrega_errors.extend(str(x) for x in flags["errors"] if str(x).strip())
    ru_numbers = _uniq([str(getattr(hit.row, "ru_number", "") or "") for hit in hits])
    holders = _uniq([str(getattr(hit.row, "ru_holder", "") or "") for hit in hits])
    ru_values = [getattr(hit.row, "ru_valid", None) for hit in hits]
    ru_valid = False if any(v is False for v in ru_values) else (True if ru_values and all(v is True for v in ru_values) else None)
    site_claims_values = [getattr(hit.row, "site_claims", None) for hit in hits]
    site_claims = True if site_claims_values and all(v is True for v in site_claims_values) else None
    site_url = next((str(getattr(hit.row, "site_url", "") or "") for hit in hits if getattr(hit.row, "site_url", None)), None)
    merged_flags: dict[str, Any] = {
        "state": "found" if ru_valid is True else "unavailable",
        "items": _uniq(unrega_items), "errors": _uniq(unrega_errors),
        "supplier_role": role, "role_evidence_url": evidence_url, "role_evidence": evidence,
        "batch_complete": True, "batch_total": total_items,
        "batch_coverage": [hit.item.product for hit in hits],
    }
    return CandidateInput(
        supplier_id=int(getattr(first, "supplier_id", 0) or 0), site_claims=site_claims,
        site_url=site_url, site_price=None,
        ru_number="; ".join(ru_numbers) if ru_numbers else None,
        ru_holder="; ".join(holders) if holders else None, ru_valid=ru_valid,
        ru_registry="batch", ru_checked_at=None, unrega_flags=merged_flags,
        raw={"batch": {"complete": True, "total_items": total_items, "coverage": [
            {"product": hit.item.product, "qty": hit.item.qty, "unit": hit.item.unit,
             "site_url": getattr(hit.row, "site_url", None), "ru_number": getattr(hit.row, "ru_number", None)}
            for hit in hits]}, "scrape": {"injection_suspected": any(bool(getattr(hit.row, "injection_suspected", False)) for hit in hits)}},
    )


def _greedy_split(by_supplier: dict[int, list[_Hit]], items: list[ProcurementItem]) -> list[SplitSupplier]:
    uncovered = set(range(len(items)))
    plan: list[SplitSupplier] = []
    remaining = dict(by_supplier)
    while uncovered and remaining:
        ranked = []
        for supplier_id, hits in remaining.items():
            new_hits = [hit for hit in hits if hit.item_index in uncovered]
            if new_hits:
                ranked.append((-len({h.item_index for h in new_hits}), _supplier_sort_key(hits), supplier_id, new_hits))
        if not ranked:
            break
        ranked.sort(key=lambda x: (x[0], x[1]))
        _, _, supplier_id, selected = ranked[0]
        row = selected[0].row
        covered = sorted({hit.item_index for hit in selected})
        plan.append(SplitSupplier(str(getattr(row, "supplier_name", "поставщик")), [items[i].product for i in covered], len(items)))
        uncovered.difference_update(covered)
        remaining.pop(supplier_id, None)
    return plan


async def _primary_holder(items: list[ProcurementItem], request_id: int) -> str | None:
    """Возвращает общего ОСНОВНОГО держателя РУ, не учитывая alternatives."""
    service = get_registry_service()
    holders: list[str] = []
    for item in items:
        check = await service.check_product(item.product, request_id=request_id, cache=False)
        best = check.best
        if best is None or not best.holder:
            return None
        holders.append(str(best.holder))
    keys = {_company_key(holder) for holder in holders}
    if len(keys) != 1 or not next(iter(keys), ""):
        return None
    return holders[0]


async def run_batch_procurement(*, master_request_id: int, items: list[ProcurementItem], master_product: str) -> BatchProcurementResult:
    result = BatchProcurementResult()
    by_supplier: dict[int, list[_Hit]] = {}
    primary_holder = await _primary_holder(items, master_request_id)
    if primary_holder:
        logger.info("Batch %s: общий основной держатель РУ — %s", master_request_id, primary_holder)

    for index, item in enumerate(items):
        async with session_scope() as session:
            child = await repo.create_request(session, product=item.product, raw_input=f"BATCH_CHILD:{master_request_id}:{index + 1}", input_kind="batch-item")
            child_id = int(child.id)
        summary = await run_search(request_id=child_id, product=item.product, requirements=item.requirements)
        if summary.search_failed:
            result.errors.append(f"{item.product}: {summary.errors[0] if summary.errors else 'поиск не удался'}")
        async with session_scope() as session:
            rows = await repo.list_candidates_for_report(session, child_id)
            for row in rows:
                supplier_id = int(getattr(row, "supplier_id", 0) or 0)
                if supplier_id:
                    by_supplier.setdefault(supplier_id, []).append(_Hit(index, item, row))
            request = await repo.get_request(session, child_id)
            if request is not None and request.status != RequestStatus.CLOSED:
                await close_request(session, child_id)
        result.searched_items += 1

    # Если реестр подтвердил одного держателя по всему списку, его собственный
    # закупочный канал считаем покрывающим всю зарегистрированную линейку даже
    # если Firecrawl из-за rate-limit не успел подтвердить каждую product-page.
    # Наличие при этом НЕ объявляем подтверждённым: это предмет запроса КП.
    if primary_holder:
        holder_key = _company_key(primary_holder)
        holder_supplier_id: int | None = None
        holder_row: Any | None = None
        for supplier_id, hits in by_supplier.items():
            for hit in hits:
                if _company_key(getattr(hit.row, "supplier_name", None)) == holder_key:
                    holder_supplier_id, holder_row = supplier_id, hit.row
                    break
            if holder_row is not None:
                break
        if holder_supplier_id is not None and holder_row is not None:
            existing = {h.item_index: h for h in by_supplier[holder_supplier_id]}
            for index, item in enumerate(items):
                existing.setdefault(index, _Hit(index, item, holder_row))
            by_supplier[holder_supplier_id] = [existing[i] for i in sorted(existing)]
            logger.info("Batch %s: основной держатель %s закреплён как direct manufacturer с покрытием %s/%s", master_request_id, primary_holder, len(items), len(items))

    complete: list[tuple[int, list[_Hit]]] = []
    for supplier_id, hits in by_supplier.items():
        covered = {hit.item_index for hit in hits}
        if len(covered) != len(items):
            continue
        per_item: dict[int, _Hit] = {}
        for hit in hits:
            per_item.setdefault(hit.item_index, hit)
        selected = [per_item[i] for i in sorted(per_item)]
        # Другой производитель под альтернативным РУ не имеет права заменить
        # основную линию. Дистрибьюторы/продавцы основного изделия остаются.
        if primary_holder:
            supplier_name = str(getattr(selected[0].row, "supplier_name", "") or "")
            roles = {_role(hit.row) for hit in selected}
            if "manufacturer" in roles and _company_key(supplier_name) != _company_key(primary_holder):
                logger.info("Batch %s: альтернативный производитель %s исключён из primary coverage", master_request_id, supplier_name)
                continue
        complete.append((supplier_id, selected))

    if not complete:
        result.split_plan = _greedy_split(by_supplier, items)
        async with session_scope() as session:
            await close_request(session, master_request_id)
        return result

    complete.sort(key=lambda pair: _supplier_sort_key(pair[1]))
    shortlist = complete[:5]
    candidates = [_merged_candidate(hits, len(items)) for _, hits in shortlist]
    async with session_scope() as session:
        await repo.upsert_candidates(session, master_request_id, candidates)
        await repo.transition(session, master_request_id, RequestStatus.REPORT)
        report = await build_report(session, request_id=master_request_id, product=master_product, qty="см. количество по позициям", requirements=_uniq([req for item in items for req in item.requirements]))
    result.report = report
    result.full_coverage_suppliers = len(complete)
    return result
