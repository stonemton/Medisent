"""Fallback-поиск РУ по нормализованным закупочным наименованиям.

Закупочные таблицы часто содержат короткие торговые строки вроде
``Цоликлон Анти-А жидкий, готовый 1х10 фл``. Поиск ELK по точной строке не
находит официальную карточку, где изделие записано полным регистрационным
наименованием. Этот модуль повторяет ELK-поиск с безопасно упрощёнными и
известными официальными семействами, не подставляя номер РУ из памяти.
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


def registry_query_variants(name: str) -> list[str]:
    """Возвращает запросы от самого близкого к исходнику к более широкому."""
    original = _SPACE_RE.sub(" ", (name or "").strip())
    compact = _compact_name(original)
    low = compact.lower().replace("anti–", "anti-").replace("anti—", "anti-")
    variants: list[str] = []

    def add(value: str) -> None:
        value = _SPACE_RE.sub(" ", value.strip())
        if value and value.lower() != original.lower() and value not in variants:
            variants.append(value)

    add(compact)

    # ООО «ГЕМАТОЛОГ»: короткие строки Anti-A/Anti-B входят в одно РУ семейства АВО.
    # Ищем официальную карточку по полному семейству, а не подставляем номер РУ.
    if "цоликлон" in low and ("анти-а" in low or "anti-a" in low):
        add("Цоликлоны Анти-А, Анти-В и Анти-АВ")
        add("Цоликлоны Анти-А Анти-В Анти-АВ 9398-001-27575295-2004")
    if "цоликлон" in low and ("анти-в" in low or "anti-b" in low or "анти-b" in low):
        add("Цоликлоны Анти-А, Анти-В и Анти-АВ")
        add("Цоликлоны Анти-А Анти-В Анти-АВ 9398-001-27575295-2004")

    # В прайсах Анти-D Супер часто сокращают до «Анти-D жидкий, готовый».
    # Сначала используем торговое имя, затем формальное наименование/ТУ.
    if "цоликлон" in low and ("анти-d" in low or "anti-d" in low):
        add("Цоликлон Анти-D Супер")
        add("АНТИ-Rho(D) IgM Цоликлон Анти-D Супер 9398-002-27575295-2004")

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
        for variant in registry_query_variants(name):
            attempts.append(variant)
            fallback_records, fallback_error = await original_check(
                self, variant, None, request_id
            )
            if fallback_records:
                relevant = [
                    record for record in fallback_records
                    if _discriminant_matches(name, record)
                ]
                if relevant:
                    logger.info(
                        "ELK fallback: «%s» найдено через «%s», записей %s",
                        name,
                        variant,
                        len(relevant),
                        extra=log_extra(request_id),
                    )
                    return relevant, None
            if fallback_error:
                last_error = fallback_error

        if attempts:
            logger.info(
                "ELK fallback: «%s» варианты не дали РУ: %s",
                name,
                " | ".join(attempts),
                extra=log_extra(request_id),
            )
        return records, error or last_error

    RegistryService._check_elk = _check_elk_with_fallbacks  # type: ignore[method-assign]
