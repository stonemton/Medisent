"""Policy layer for multi-item procurement reports.

The registry may legitimately find equivalent products under another RU. Those
alternatives are useful to show, but they must not become the primary
manufacturer for a batch whose confirmed RU holder is someone else.
"""
from __future__ import annotations

import re
from typing import Any

from bot import texts

_INSTALLED = False


def _org_terms(value: str | None) -> set[str]:
    stop = {
        "ооо", "ао", "пао", "оао", "зао", "общество", "ограниченной",
        "ответственностью", "акционерное", "компания", "россия", "рф",
    }
    return {
        token
        for token in re.findall(r"[a-zа-яё0-9]{4,}", (value or "").lower())
        if token not in stop
    }


def _supplier_matches_holder(row: Any) -> bool:
    holder = str(getattr(row, "ru_holder", "") or "")
    if not holder:
        return False
    haystack = " ".join(
        str(getattr(row, key, "") or "")
        for key in ("supplier_name", "domain")
    ).lower()
    terms = _org_terms(holder)
    return bool(terms and any(term in haystack for term in terms))


def install_batch_supplier_policy() -> None:
    """Install strict primary-RU semantics without changing DB schema."""
    global _INSTALLED
    if _INSTALLED:
        return

    from bot.services import procurement_batch, report

    original_batch_role = procurement_batch._role

    def batch_role(row: Any) -> str:
        role = original_batch_role(row)
        # Search engines can call a company "manufacturer" because it makes an
        # equivalent product under another RU. In the PRIMARY branch this is
        # only manufacturer when its identity matches the confirmed RU holder.
        if role == "manufacturer" and not _supplier_matches_holder(row):
            return "seller"
        return role

    procurement_batch._role = batch_role

    def strict_holder_or_manufacturer(view: Any) -> bool:
        # Never trust the discovery label alone. Manufacturer priority is tied
        # to the RU holder confirmed for the selected product line.
        holder_terms = _org_terms(getattr(view, "ru_holder", None))
        if not holder_terms:
            return False
        haystack = f"{getattr(view, 'supplier_name', '')} {getattr(view, 'domain', '') or ''}".lower()
        return any(term in haystack for term in holder_terms)

    report._is_holder_or_manufacturer = strict_holder_or_manufacturer

    # Reduce Firecrawl pressure: batch search has several item passes and the
    # old 24-sites-per-item ceiling quickly hit the 15 req/min plan limit.
    from bot import pipeline
    pipeline.MAX_SITES_TO_SCRAPE = min(int(getattr(pipeline, "MAX_SITES_TO_SCRAPE", 24)), 8)

    texts.REPORT_FOOTER = (
        "Выберите основной канал закупки. Альтернативные производители под другим РУ "
        "не подменяют подтверждённую товарную линию."
    )
    _INSTALLED = True
