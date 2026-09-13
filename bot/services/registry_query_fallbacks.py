"""Точный fallback-поиск РУ для коротких закупочных наименований Цоликлонов.

Вместо серии широких Firecrawl-поисков используем уже подтверждённые через
официальные карточки ELK семейства РУ как поисковые подсказки, а затем всё
равно валидируем карточку: нужный антиген должен присутствовать в полном
тексте карточки, а основной матч должен принадлежать ООО «ГЕМАТОЛОГ».

Совпадающее изделие другого держателя сохраняется как альтернативное РУ.
Точные RU-поиски дедуплицируются даже при параллельной обработке batch, чтобы
A/B/D не запускали одинаковую проверку альтернативного РУ одновременно.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from bot.logging_setup import log_extra
from bot.services.registry_endpoints import RegistryRecord

logger = logging.getLogger(__name__)

_ORIGINAL_CHECK: Callable[..., Awaitable[tuple[list[RegistryRecord], str | None]]] | None = None

_PACKAGING_RE = re.compile(
    r"\b(?:\d+\s*[xх×]\s*)?\d+(?:[.,]\d+)?\s*(?:мл|ml|фл\.?|флак\.?|флакон(?:а|ов)?|шт\.?)\b",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(
    r"\b(?:жидк(?:ий|ая|ое)|готов(?:ый|ая|ое)|реагент|диагностическ(?:ий|ая|ое))\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")

_GEMATOLOG_ABO_RU = "ФСР 2008/04007"
_GEMATOLOG_RH_KELL_KIDD_RU = "ФСР 2012/12983"
_MEDIKLON_ABO_RH_KELL_RU = "ФСР 2009/06043"


def _compact_name(name: str) -> str:
    value = (name or "").replace("ё", "е")
    value = _PACKAGING_RE.sub(" ", value)
    value = _NOISE_RE.sub(" ", value)
    value = re.sub(r"[(),.;:]+", " ", value)
    value = _SPACE_RE.sub(" ", value).strip(" -–—")
    return value


def _antigen_kind(name: str) -> str | None:
    low = _compact_name(name).lower().replace("anti–", "anti-").replace("anti—", "anti-")
    if "цоликлон" not in low:
        return None
    if "анти-а" in low or "anti-a" in low:
        return "a"
    if "анти-в" in low or "anti-b" in low or "анти-b" in low:
        return "b"
    if "анти-d" in low or "anti-d" in low:
        return "d"
    return None


def _record_text(record: RegistryRecord) -> str:
    raw_text = ""
    if isinstance(record.raw, dict):
        raw_text = str(record.raw.get("text") or "")
    return " ".join(
        str(x or "") for x in (record.product_name, record.holder, record.ru_number, raw_text)
    ).lower().replace("anti–", "anti-").replace("anti—", "anti-")


def _antigen_matches(kind: str, record: RegistryRecord) -> bool:
    haystack = _record_text(record)
    expected = {
        "a": ("анти-а", "anti-a"),
        "b": ("анти-в", "anti-b", "анти-b"),
        "d": ("анти-d", "anti-d", "rho(d)", "rhod"),
    }[kind]
    return any(token in haystack for token in expected)


def _holder_matches(record: RegistryRecord, expected: str) -> bool:
    holder = re.sub(r"[^a-zа-яё0-9]+", "", str(record.holder or "").lower())
    needle = re.sub(r"[^a-zа-яё0-9]+", "", expected.lower())
    return bool(holder and needle and needle in holder)


def _alternative_payload(record: RegistryRecord) -> dict[str, Any]:
    return {
        "ru_number": record.ru_number,
        "holder": record.holder,
        "product_name": record.product_name,
        "valid": record.valid,
        "card_url": record.card_url,
    }


def _attach_alternatives(
    records: list[RegistryRecord], alternatives: list[RegistryRecord]
) -> list[RegistryRecord]:
    if not alternatives:
        return records
    payloads: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for alt in alternatives:
        key = (
            str(alt.ru_number or "").strip().lower(),
            str(alt.holder or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        payloads.append(_alternative_payload(alt))

    attached: list[RegistryRecord] = []
    for record in records:
        raw = dict(record.raw) if isinstance(record.raw, dict) else {}
        raw["registry_alternatives"] = [dict(item) for item in payloads]
        attached.append(replace(record, raw=raw))
    return attached


def install_registry_query_fallbacks() -> None:
    """Один раз добавляет точный fallback к RegistryService._check_elk."""
    global _ORIGINAL_CHECK
    if _ORIGINAL_CHECK is not None:
        return

    from bot.services.registry import RegistryService

    _ORIGINAL_CHECK = RegistryService._check_elk
    original_check = _ORIGINAL_CHECK

    exact_cache: dict[str, tuple[list[RegistryRecord], str | None]] = {}
    exact_inflight: dict[str, asyncio.Task[tuple[list[RegistryRecord], str | None]]] = {}

    async def exact_ru(
        self: RegistryService, ru: str, request_id: int | None
    ) -> tuple[list[RegistryRecord], str | None]:
        cached = exact_cache.get(ru)
        if cached is not None:
            return cached

        task = exact_inflight.get(ru)
        if task is None:
            task = asyncio.create_task(original_check(self, None, ru, request_id))
            exact_inflight[ru] = task
        try:
            result = await task
        finally:
            if exact_inflight.get(ru) is task and task.done():
                exact_inflight.pop(ru, None)

        if result[0]:
            exact_cache[ru] = result
        return result

    async def _check_elk_with_fallbacks(
        self: RegistryService,
        name: str | None,
        ru_number: str | None,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], str | None]:
        if ru_number or not name:
            return await original_check(self, name, ru_number, request_id)

        kind = _antigen_kind(name)
        if kind is None:
            return await original_check(self, name, ru_number, request_id)

        primary_ru = _GEMATOLOG_ABO_RU if kind in {"a", "b"} else _GEMATOLOG_RH_KELL_KIDD_RU

        (primary_records, primary_error), (alt_records, alt_error) = await asyncio.gather(
            exact_ru(self, primary_ru, request_id),
            exact_ru(self, _MEDIKLON_ABO_RH_KELL_RU, request_id),
        )

        primary = [
            record for record in primary_records
            if _holder_matches(record, "гематолог") and _antigen_matches(kind, record)
        ]
        alternatives = [
            record for record in alt_records
            if not _holder_matches(record, "гематолог") and _antigen_matches(kind, record)
        ]

        if primary:
            primary = _attach_alternatives(primary, alternatives)
            logger.info(
                "ELK exact fallback: «%s» → основной %s / %s; альтернатив %s",
                name,
                primary[0].ru_number or "—",
                primary[0].holder or "—",
                len(alternatives),
                extra=log_extra(request_id),
            )
            return primary, None

        logger.warning(
            "ELK exact fallback: «%s» не подтвердил основной RU=%s; primary_error=%s alt_error=%s",
            name,
            primary_ru,
            primary_error or "—",
            alt_error or "—",
            extra=log_extra(request_id),
        )
        return [], primary_error or alt_error or "официальная карточка нужного семейства не подтвердила позицию"

    RegistryService._check_elk = _check_elk_with_fallbacks  # type: ignore[method-assign]
