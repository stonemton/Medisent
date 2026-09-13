"""Policy layer for procurement reports.

Verified RU holder remains the strongest manufacturer evidence. When RU is not
confirmed, however, procurement must still work: an explicitly discovered
manufacturer may be treated as the manufacturer channel, but never as a
verified RU holder. This keeps missing RU from flattening every supplier into
"прочий поставщик".
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
    """Install strict primary-RU semantics while allowing unverified-RU procurement."""
    global _INSTALLED
    if _INSTALLED:
        return

    from bot.services import procurement_batch, report

    original_batch_role = procurement_batch._role

    def batch_role(row: Any) -> str:
        role = original_batch_role(row)
        holder = str(getattr(row, "ru_holder", "") or "").strip()
        # With a confirmed holder, another company's discovery label must not
        # override the RU branch. Without a confirmed holder, an explicit
        # manufacturer discovery is useful commercial evidence and is kept.
        if role == "manufacturer" and holder and not _supplier_matches_holder(row):
            return "seller"
        return role

    procurement_batch._role = batch_role

    def strict_holder_or_manufacturer(view: Any) -> bool:
        holder_terms = _org_terms(getattr(view, "ru_holder", None))
        if holder_terms:
            haystack = f"{getattr(view, 'supplier_name', '')} {getattr(view, 'domain', '') or ''}".lower()
            return any(term in haystack for term in holder_terms)
        # RU is not verified: do not call anyone a holder, but preserve an
        # independently discovered manufacturer role for procurement ranking.
        return str(getattr(view, "supplier_role", "") or "") == "manufacturer"

    report._is_holder_or_manufacturer = strict_holder_or_manufacturer

    # Reduce Firecrawl pressure: batch search has several item passes and the
    # old 24-sites-per-item ceiling quickly hit the request-rate limit.
    from bot import pipeline
    pipeline.MAX_SITES_TO_SCRAPE = min(int(getattr(pipeline, "MAX_SITES_TO_SCRAPE", 24)), 8)

    texts.REPORT_FOOTER = (
        "На этом этапе основной поставщик ещё не выбирается. Запросите КП у производителя/"
        "основного канала и 1–2 сильных альтернатив, затем сравните цену, наличие, срок поставки "
        "и подтверждение РУ."
    )
    _INSTALLED = True
