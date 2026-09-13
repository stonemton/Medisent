"""Fallback-поиск РУ по нормализованным закупочным наименованиям.

Закупочные таблицы часто содержат короткие торговые строки вроде
``Цоликлон Анти-А жидкий, готовый 1х10 фл``. Поиск ELK по точной строке не
находит официальную карточку, где изделие записано полным регистрационным
наименованием. Этот модуль повторяет ELK-поиск с безопасно упрощёнными и
известными официальными семействами, не подставляя номер РУ из памяти.

Для известных товарных семейств fallback дополнительно привязывается к
держателю/производителю. Совпадающие изделия других держателей не становятся
основным матчем, но сохраняются как альтернативные РУ для показа пользователю.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
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


def _compact_name(name: str) -> str:
    """Убирает фасовку и описательные слова, сохраняя отличительную часть."""
    value = (name or "").replace("ё", "е")
    value = _PACKAGING_RE.sub(" ", value)
    value = _NOISE_RE.sub(" ", value)
    value = re.sub(r"[(),.;:]+", " ", value)
    value = _SPACE_RE.sub(" ", value).strip(" -–—")
    return value


def _is_gematolog_cyclone(name: str) -> bool:
    """Короткая строка из известных семейств Цоликлон ООО «ГЕМАТОЛОГ»."""
    low = _compact_name(name).lower().replace("anti–", "anti-").replace("anti—", "anti-")
    if "цоликлон" not in low:
        return False
    return any(
        token in low
        for token in (
            "анти-а", "anti-a",
            "анти-в", "anti-b", "анти-b",
            "анти-d", "anti-d",
        )
    )


def registry_query_variants(name: str) -> list[str]:
    """Возвращает запросы от наиболее специфичных к более широким."""
    original = _SPACE_RE.sub(" ", (name or "").strip())
    compact = _compact_name(original)
    low = compact.lower().replace("anti–", "anti-").replace("anti—", "anti-")
    variants: list[str] = []

    def add(value: str) -> None:
        value = _SPACE_RE.sub(" ", value.strip())
        if value and value.lower() != original.lower() and value not in variants:
            variants.append(value)

    if "цоликлон" in low and ("анти-а" in low or "anti-a" in low):
        add("Цоликлоны Анти-А Анти-В Анти-АВ 9398-001-27575295-2004")
        add("Цоликлоны Анти-А, Анти-В и Анти-АВ")
    if "цоликлон" in low and ("анти-в" in low or "anti-b" in low or "анти-b" in low):
        add("Цоликлоны Анти-А Анти-В Анти-АВ 9398-001-27575295-2004")
        add("Цоликлоны Анти-А, Анти-В и Анти-АВ")
    if "цоликлон" in low and ("анти-d" in low or "anti-d" in low):
        add("АНТИ-Rho(D) IgM Цоликлон Анти-D Супер 9398-002-27575295-2004")
        add("Цоликлон Анти-D Супер")

    add(compact)
    return variants


def _discriminant_matches(original_name: str, record: RegistryRecord) -> bool:
    """Не принимает широкую карточку, если пропала ключевая специфика товара."""
    query = original_name.lower().replace("anti–", "anti-").replace("anti—", "anti-")
    haystack = " ".join(
        str(x or "") for x in (record.product_name, record.holder, record.ru_number)
    ).lower()

    checks: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("анти-а", "anti-a"), ("анти-а", "anti-a")),
        (("анти-в", "anti-b", "анти-b"), ("анти-в", "anti-b", "анти-b")),
        (("анти-d", "anti-d"), ("анти-d", "anti-d", "rho(d)", "rhod")),
        (("келл", "kell"), ("келл", "kell")),
    ]
    for needles, expected in checks:
        if any(token in query for token in needles):
            return any(token in haystack for token in expected)
    return True


def _holder_matches(record: RegistryRecord, expected: str) -> bool:
    holder = re.sub(r"[^a-zа-яё0-9]+", "", str(record.holder or "").lower())
    needle = re.sub(r"[^a-zа-яё0-9]+", "", expected.lower())
    return bool(holder and needle and needle in holder)


def _variant_expected_holder(variant: str) -> str | None:
    """ТУ 27575295 относится к товарной линии ООО «ГЕМАТОЛОГ»."""
    return "гематолог" if "27575295" in variant else None


def _alternative_payload(record: RegistryRecord) -> dict[str, Any]:
    return {
        "ru_number": record.ru_number,
        "holder": record.holder,
        "product_name": record.product_name,
        "valid": record.valid,
        "card_url": record.card_url,
    }


def _add_alternative(alternatives: list[dict[str, Any]], record: RegistryRecord) -> None:
    payload = _alternative_payload(record)
    key = (
        str(payload.get("ru_number") or "").strip().lower(),
        str(payload.get("holder") or "").strip().lower(),
    )
    if not any(
        (
            str(item.get("ru_number") or "").strip().lower(),
            str(item.get("holder") or "").strip().lower(),
        ) == key
        for item in alternatives
    ):
        alternatives.append(payload)


def _attach_alternatives(records: list[RegistryRecord], alternatives: list[dict[str, Any]]) -> None:
    if not alternatives:
        return
    for record in records:
        raw = dict(record.raw) if isinstance(record.raw, dict) else {}
        raw["registry_alternatives"] = alternatives
        record.raw = raw


def install_registry_query_fallbacks() -> None:
    """Один раз добавляет fallback к RegistryService._check_elk."""
    global _ORIGINAL_CHECK
    if _ORIGINAL_CHECK is not None:
        return

    from bot.services.registry import RegistryService

    _ORIGINAL_CHECK = RegistryService._check_elk
    original_check = _ORIGINAL_CHECK

    async def _check_elk_with_fallbacks(
        self: RegistryService,
        name: str | None,
        ru_number: str | None,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], str | None]:
        records, error = await original_check(self, name, ru_number, request_id)
        if ru_number or not name:
            return records, error

        locked_family = _is_gematolog_cyclone(name)
        if not locked_family:
            return records, error

        alternatives: list[dict[str, Any]] = []
        primary: list[RegistryRecord] = []

        # Точный короткий запрос может уверенно найти изделие другого производителя.
        # Не теряем его: показываем как другое РУ, но не делаем основным матчем.
        for record in records:
            if not _discriminant_matches(name, record):
                continue
            if _holder_matches(record, "гематолог"):
                primary.append(record)
            else:
                _add_alternative(alternatives, record)

        attempts: list[str] = []
        last_error = error
        for variant in registry_query_variants(name):
            attempts.append(variant)
            fallback_records, fallback_error = await original_check(
                self, variant, None, request_id
            )
            expected_holder = _variant_expected_holder(variant)

            for record in fallback_records:
                if not _discriminant_matches(name, record):
                    continue
                holder_ok = _holder_matches(record, expected_holder or "гематолог")
                if holder_ok:
                    if not any(
                        (r.ru_number, r.holder) == (record.ru_number, record.holder)
                        for r in primary
                    ):
                        primary.append(record)
                else:
                    _add_alternative(alternatives, record)

            if fallback_error:
                last_error = fallback_error

        if primary:
            _attach_alternatives(primary, alternatives)
            logger.info(
                "ELK fallback: «%s» основной держатель подтверждён, РУ=%s; альтернативных РУ=%s",
                name,
                primary[0].ru_number or "—",
                len(alternatives),
                extra=log_extra(request_id),
            )
            return primary, None

        if alternatives:
            logger.warning(
                "ELK fallback: «%s» найдены только совпадающие изделия других держателей: %s",
                name,
                ", ".join(
                    f"{item.get('ru_number') or 'РУ —'} / {item.get('holder') or 'держатель —'}"
                    for item in alternatives[:5]
                ),
                extra=log_extra(request_id),
            )

        if attempts:
            logger.info(
                "ELK fallback: «%s» варианты не дали согласованного основного РУ: %s",
                name,
                " | ".join(attempts),
                extra=log_extra(request_id),
            )
        return [], error or last_error

    RegistryService._check_elk = _check_elk_with_fallbacks  # type: ignore[method-assign]
