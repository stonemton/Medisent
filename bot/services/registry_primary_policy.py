"""Post-policy for registry fallback results.

The universal matcher may find several equally plausible RUs for a short trade
name. Known verified product-line hints are allowed to choose the primary card,
but all other cards stay visible as alternatives. This module does not perform
searches and therefore does not increase Firecrawl traffic.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from bot.services.registry_endpoints import RegistryRecord

_ORIGINAL_CHECK: Callable[..., Awaitable[tuple[list[RegistryRecord], str | None]]] | None = None


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", str(value or "").lower().replace("ё", "е"))


def _is_cyolyclone_abd(name: str | None) -> bool:
    low = str(name or "").lower().replace("ё", "е")
    return "цоликлон" in low and any(x in low for x in ("анти-а", "анти-в", "анти-d", "anti-a", "anti-b", "anti-d"))


def _payload(record: RegistryRecord) -> dict[str, Any]:
    return {
        "ru_number": record.ru_number,
        "holder": record.holder,
        "product_name": record.product_name,
        "valid": record.valid,
        "card_url": record.card_url,
        "registry": record.registry,
        "source": record.raw.get("source") if isinstance(record.raw, dict) else None,
    }


def _record_from_alt(item: dict[str, Any], template: RegistryRecord) -> RegistryRecord:
    return RegistryRecord(
        registry=str(item.get("registry") or template.registry),
        ru_number=item.get("ru_number"),
        holder=item.get("holder"),
        product_name=item.get("product_name"),
        valid=item.get("valid"),
        status_text=None,
        card_url=item.get("card_url"),
        raw={"source": item.get("source") or "registry_alternative"},
    )


def install_registry_primary_policy() -> None:
    """Keep universal matching, but prevent a generic match from overriding a verified line."""
    global _ORIGINAL_CHECK
    if _ORIGINAL_CHECK is not None:
        return

    from bot.services.registry import RegistryService

    _ORIGINAL_CHECK = RegistryService._check_elk
    original = _ORIGINAL_CHECK

    async def wrapped(self: RegistryService, name: str | None, ru_number: str | None, request_id: int | None):
        records, error = await original(self, name, ru_number, request_id)
        if ru_number or not name or not records or not _is_cyolyclone_abd(name):
            return records, error

        current = records[0]
        raw = dict(current.raw) if isinstance(current.raw, dict) else {}
        alternatives = [x for x in raw.get("registry_alternatives", []) if isinstance(x, dict)]

        # This product line was explicitly verified earlier against official ELK cards.
        # Prefer Hematolog when it is among the verified candidates; keep Mediclone and
        # every other RU as alternatives. The generic matcher remains active for all
        # unrelated medical products.
        candidates = [current, *[_record_from_alt(x, current) for x in alternatives]]
        preferred = next((r for r in candidates if "гематолог" in _norm(r.holder)), None)
        if preferred is None or preferred is current:
            return records, error

        rest = [r for r in candidates if not (_norm(r.ru_number) == _norm(preferred.ru_number) and _norm(r.holder) == _norm(preferred.holder))]
        preferred_raw = dict(preferred.raw) if isinstance(preferred.raw, dict) else {}
        preferred_raw["registry_alternatives"] = [_payload(r) for r in rest[:5]]
        return [replace(preferred, raw=preferred_raw)], error

    RegistryService._check_elk = wrapped
