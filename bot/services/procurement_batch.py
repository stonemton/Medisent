"""Пакетный закупочный конвейер для заявок из нескольких позиций.

Идея: каждую позицию проверяем и ищем отдельно, но наружу владельцу показываем
одну master-заявку. Поставщик попадает в master-shortlist только если реально
прошёл Supplier Gate по КАЖДОЙ позиции. Поэтому единое письмо нельзя случайно
отправить магазину, у которого подтверждена лишь часть списка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bot.db import repo
from bot.db.models import RequestStatus
from bot.db.repo import CandidateInput
from bot.db.session import session_scope
from bot.pipeline import build_report, close_request, run_search
from bot.services.batch_intake import ProcurementItem
from bot.services.report import Report


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
    role_index, role = min(
        ((ROLE_ORDER.get(str(flags.get("supplier_role") or "candidate"), 3), str(flags.get("supplier_role") or "candidate")) for flags in flags_list),
        default=(3, "candidate"),
    )
    del role_index
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
    ru_valid: bool | None
    if any(value is False for value in ru_values):
        ru_valid = False
    elif ru_values and all(value is True for value in ru_values):
        ru_valid = True
    else:
        ru_valid = None

    site_claims_values = [getattr(hit.row, "site_claims", None) for hit in hits]
    site_claims = True if site_claims_values and all(v is True for v in site_claims_values) else None
    site_url = next((str(getattr(hit.row, "site_url", "") or "") for hit in hits if getattr(hit.row, "site_url", None)), None)

    merged_flags: dict[str, Any] = {
        "state": "found" if ru_valid is True else "unavailable",
        "items": _uniq(unrega_items),
        "errors": _uniq(unrega_errors),
        "supplier_role": role,
        "role_evidence_url": evidence_url,
        "role_evidence": evidence,
        "batch_complete": True,
        "batch_total": total_items,
        "batch_coverage": [hit.item.product for hit in hits],
    }
    return CandidateInput(
        supplier_id=int(getattr(first, "supplier_id", 0) or 0),
        site_claims=site_claims,
        site_url=site_url,
        site_price=None,
        ru_number="; ".join(ru_numbers) if ru_numbers else None,
        ru_holder="; ".join(holders) if holders else None,
        ru_valid=ru_valid,
        ru_registry="batch",
        ru_checked_at=None,
        unrega_flags=merged_flags,
        raw={
            "batch": {
                "complete": True,
                "total_items": total_items,
                "coverage": [
                    {
                        "product": hit.item.product,
                        "qty": hit.item.qty,
                        "unit": hit.item.unit,
                        "site_url": getattr(hit.row, "site_url", None),
                        "ru_number": getattr(hit.row, "ru_number", None),
                    }
                    for hit in hits
                ],
            },
            "scrape": {"injection_suspected": any(bool(getattr(hit.row, "injection_suspected", False)) for hit in hits)},
        },
    )


def _greedy_split(by_supplier: dict[int, list[_Hit]], items: list[ProcurementItem]) -> list[SplitSupplier]:
    uncovered = set(range(len(items)))
    plan: list[SplitSupplier] = []
    remaining = dict(by_supplier)
    while uncovered and remaining:
        ranked: list[tuple[int, tuple[int, int, int], int, list[_Hit]]] = []
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
        plan.append(
            SplitSupplier(
                supplier_name=str(getattr(row, "supplier_name", "поставщик")),
                covered_items=[items[i].product for i in covered],
                total_items=len(items),
            )
        )
        uncovered.difference_update(covered)
        remaining.pop(supplier_id, None)
    return plan


async def run_batch_procurement(
    *,
    master_request_id: int,
    items: list[ProcurementItem],
    master_product: str,
) -> BatchProcurementResult:
    result = BatchProcurementResult()
    by_supplier: dict[int, list[_Hit]] = {}

    for index, item in enumerate(items):
        async with session_scope() as session:
            child = await repo.create_request(
                session,
                product=item.product,
                raw_input=f"BATCH_CHILD:{master_request_id}:{index + 1}",
                input_kind="batch-item",
            )
            child_id = int(child.id)
        summary = await run_search(
            request_id=child_id,
            product=item.product,
            requirements=item.requirements,
        )
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

    complete: list[tuple[int, list[_Hit]]] = []
    for supplier_id, hits in by_supplier.items():
        covered = {hit.item_index for hit in hits}
        if len(covered) == len(items):
            # Дедуп: на одну позицию оставляем один hit поставщика.
            per_item: dict[int, _Hit] = {}
            for hit in hits:
                per_item.setdefault(hit.item_index, hit)
            complete.append((supplier_id, [per_item[i] for i in sorted(per_item)]))

    if not complete:
        result.split_plan = _greedy_split(by_supplier, items)
        async with session_scope() as session:
            await close_request(session, master_request_id)
        return result

    complete.sort(key=lambda pair: _supplier_sort_key(pair[1]))
    # Не перегружаем отчёт: максимум пять поставщиков, каждый подтверждён по всему списку.
    shortlist = complete[:5]
    candidates = [_merged_candidate(hits, len(items)) for _, hits in shortlist]
    async with session_scope() as session:
        await repo.upsert_candidates(session, master_request_id, candidates)
        await repo.transition(session, master_request_id, RequestStatus.REPORT)
        report = await build_report(
            session,
            request_id=master_request_id,
            product=master_product,
            qty="см. количество по позициям",
            requirements=_uniq([req for item in items for req in item.requirements]),
        )
    result.report = report
    result.full_coverage_suppliers = len(complete)
    return result
