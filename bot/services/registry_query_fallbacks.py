"""Fallback-поиск РУ по нормализованным закупочным наименованиям.

Закупочные таблицы часто содержат короткие торговые строки вроде
``Цоликлон Анти-А жидкий, готовый 1х10 фл``. Поиск ELK по точной строке не
находит официальную карточку, где изделие записано полным регистрационным
наименованием. Этот модуль повторяет ELK-поиск с безопасно упрощёнными и
известными официальными семействами, не подставляя номер РУ из памяти.

Для известных товарных семейств fallback дополнительно привязывается к
держателю/производителю. Это не даёт принять корректное РУ от другого
производителя только потому, что короткое торговое наименование совпало.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

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

    # Сначала специфичные официальные семейства/ТУ. Только после них —
    # короткое торговое имя, которое может совпадать у разных производителей.
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
        if records or ru_number or not name:
            return records, error

        attempts: list[str] = []
        last_error = error
        locked_family = _is_gematolog_cyclone(name)
        for variant in registry_query_variants(name):
            attempts.append(variant)
            fallback_records, fallback_error = await original_check(
                self, variant, None, request_id
            )
            if fallback_records:
                expected_holder = _variant_expected_holder(variant)
                relevant = [
                    record for record in fallback_records
                    if _discriminant_matches(name, record)
                    and (expected_holder is None or _holder_matches(record, expected_holder))
                ]
                if relevant:
                    # Для коротких Цоликлонов не принимаем карточку чужого
                    # производителя на широком fallback. Сначала должна быть
                    # подтверждена специфичная линия/ТУ ООО «ГЕМАТОЛОГ».
                    if locked_family and expected_holder is None:
                        gematolog_records = [r for r in relevant if _holder_matches(r, "гематолог")]
                        if not gematolog_records:
                            logger.warning(
                                "ELK fallback: «%s» через «%s» дал только другого держателя — игнорирую",
                                name,
                                variant,
                                extra=log_extra(request_id),
                            )
                            continue
                        relevant = gematolog_records

                    logger.info(
                        "ELK fallback: «%s» найдено через «%s», записей %s, держатель=%s",
                        name,
                        variant,
                        len(relevant),
                        relevant[0].holder or "—",
                        extra=log_extra(request_id),
                    )
                    return relevant, None
            if fallback_error:
                last_error = fallback_error

        if attempts:
            logger.info(
                "ELK fallback: «%s» варианты не дали согласованного РУ: %s",
                name,
                " | ".join(attempts),
                extra=log_extra(request_id),
            )
        return records, error or last_error

    RegistryService._check_elk = _check_elk_with_fallbacks  # type: ignore[method-assign]
