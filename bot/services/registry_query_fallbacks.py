"""Универсальный fallback-поиск РУ для закупочных наименований.

Сначала используется обычный поиск реестра. Если короткое/прайсовое название
не дало уверенного результата, строятся нормализованные поисковые варианты без
количества, фасовки и служебных слов. Найденные карточки ранжируются по
совпадению значимых терминов и идентификаторов; другие подходящие РУ
сохраняются как альтернативы, а не подменяют основной выбор.

Для известных проблемных семейств можно оставить точечные подсказки как
ускоритель. Они не являются основой алгоритма: для любой другой медпродукции
работает общий механизм ниже.
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
    r"\b(?:\d+\s*[xх×]\s*)?\d+(?:[.,]\d+)?\s*(?:мл|ml|мг|mg|г|g|см3|cm3|фл\.?|флак\.?|флакон(?:а|ов)?|шт\.?|уп\.?|упак\.?|доз(?:а|ы|ов)?)\b",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(
    r"\b(?:жидк(?:ий|ая|ое)|готов(?:ый|ая|ое)|стерильн(?:ый|ая|ое)|одноразов(?:ый|ая|ое)|"
    r"реагент|диагностическ(?:ий|ая|ое)|медицинск(?:ий|ая|ое)|издели(?:е|я)|набор)\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9._/+()-]{2,}")
_TECH_ID_RE = re.compile(r"(?i)(?=[A-Za-zА-Яа-я0-9._/-]*\d)[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9._/-]{3,}")

_STOP = {
    "для", "или", "при", "без", "над", "под", "между", "через", "крови", "человека",
    "готовый", "жидкий", "изделие", "медицинское", "медицинский", "набор", "реагентов",
    "реагент", "шт", "флакон", "флаконе", "упаковка", "стерильный", "одноразовый",
}

# Точечные подсказки для известного семейства. Общий алгоритм работает и без них,
# но ELK сейчас нестабильно ищет некоторые старые номера этой линейки.
_GEMATOLOG_ABO_RU = "ФСР 2008/04007"
_GEMATOLOG_ANTI_D_PRIMARY_RU = "ФСР 2012/12983"
_GEMATOLOG_ANTI_D_LEGACY_RU = "ФСР 2009/05552"
_MEDIKLON_ABO_RH_KELL_RU = "ФСР 2009/06043"
_MEDIKLON_FAMILY_QUERY = (
    "Набор реагентов для определения групп крови человека систем АВО, Резус и Kell "
    "Цоликлоны 9398-101-51203590-2009"
)


def _normalise(value: str | None) -> str:
    value = str(value or "").lower().replace("ё", "е")
    value = value.replace("–", "-").replace("—", "-").replace("×", "x").replace("х", "x")
    return _SPACE_RE.sub(" ", value).strip()


def _compact_name(name: str) -> str:
    value = _normalise(name)
    value = _PACKAGING_RE.sub(" ", value)
    value = _NOISE_RE.sub(" ", value)
    value = re.sub(r"[,:;]+", " ", value)
    value = _SPACE_RE.sub(" ", value).strip(" -–—.,()")
    return value


def _terms(value: str) -> list[str]:
    out: list[str] = []
    for token in _TOKEN_RE.findall(_normalise(value)):
        cleaned = token.strip("._/+()-")
        if len(cleaned) < 3 or cleaned in _STOP or cleaned.isdigit():
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out


def _tech_ids(value: str) -> set[str]:
    return {m.group(0).lower() for m in _TECH_ID_RE.finditer(_normalise(value)) if len(m.group(0)) >= 4}


def _record_text(record: RegistryRecord) -> str:
    raw_text = str(record.raw.get("text") or "") if isinstance(record.raw, dict) else ""
    return _normalise(" ".join(str(x or "") for x in (record.product_name, record.holder, record.ru_number, raw_text)))


def _holder_matches(record: RegistryRecord, expected: str) -> bool:
    holder = re.sub(r"[^a-zа-яё0-9]+", "", _normalise(record.holder))
    needle = re.sub(r"[^a-zа-яё0-9]+", "", _normalise(expected))
    return bool(holder and needle and needle in holder)


def _ru_matches(record: RegistryRecord, expected: str) -> bool:
    normalise = lambda value: re.sub(r"[^a-zа-яё0-9]+", "", _normalise(value))
    return normalise(record.ru_number) == normalise(expected)


def _antigen_kind(name: str) -> str | None:
    low = _compact_name(name)
    if "цоликлон" not in low:
        return None
    if "анти-а" in low or "anti-a" in low:
        return "a"
    if "анти-в" in low or "anti-b" in low or "анти-b" in low:
        return "b"
    if "анти-d" in low or "anti-d" in low:
        return "d"
    return None


def _antigen_matches(kind: str, record: RegistryRecord) -> bool:
    haystack = _record_text(record)
    expected = {
        "a": ("анти-а", "anti-a"),
        "b": ("анти-в", "anti-b", "анти-b"),
        "d": ("анти-d", "anti-d", "rho(d)", "rhod"),
    }[kind]
    return any(token in haystack for token in expected)


def _identity_score(name: str, record: RegistryRecord) -> tuple[float, int, int]:
    """Сходство карточки с исходной строкой: 0..1 + бонусы за идентификаторы."""
    query_terms = set(_terms(_compact_name(name)))
    haystack = _record_text(record)
    if not query_terms:
        return (0.0, 0, int(record.valid is True))
    hits = {term for term in query_terms if term in haystack}
    ratio = len(hits) / len(query_terms)
    q_ids = _tech_ids(name)
    id_hits = sum(1 for token in q_ids if token in haystack)
    return (ratio, id_hits, int(record.valid is True))


def _generic_confident(name: str, record: RegistryRecord) -> bool:
    ratio, id_hits, _ = _identity_score(name, record)
    term_count = len(set(_terms(_compact_name(name))))
    if id_hits:
        return ratio >= 0.25
    if term_count <= 2:
        return ratio >= 1.0
    return ratio >= 0.50


def _query_variants(name: str) -> list[str]:
    """Поисковые варианты от точного к более общему, без подстановки фактов."""
    variants: list[str] = []
    compact = _compact_name(name)
    if compact and compact != _normalise(name):
        variants.append(compact)

    # Если есть артикул/модель, сохраняем его вместе с несколькими смысловыми словами.
    ids = list(_tech_ids(name))
    words = _terms(compact)
    if ids:
        model_query = " ".join([*words[:5], *ids[:2]]).strip()
        if model_query:
            variants.append(model_query)

    # Убираем хвост после перечисления характеристик, но не делаем запрос из одного общего слова.
    if len(words) >= 3:
        variants.append(" ".join(words[:8]))

    out: list[str] = []
    seen: set[str] = set()
    for value in variants:
        key = _normalise(value)
        if len(key) >= 6 and key not in seen:
            seen.add(key)
            out.append(value)
    return out[:3]


def _dedup_records(records: list[RegistryRecord]) -> list[RegistryRecord]:
    out: list[RegistryRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            _normalise(record.ru_number),
            _normalise(record.holder),
            _normalise(record.product_name),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _alternative_payload(record: RegistryRecord) -> dict[str, Any]:
    source = None
    if isinstance(record.raw, dict):
        source = record.raw.get("source")
    return {
        "ru_number": record.ru_number,
        "holder": record.holder,
        "product_name": record.product_name,
        "valid": record.valid,
        "card_url": record.card_url,
        "registry": record.registry,
        "source": source,
    }


def _attach_alternatives(primary: RegistryRecord, alternatives: list[RegistryRecord]) -> RegistryRecord:
    payloads: list[dict[str, Any]] = []
    primary_key = (_normalise(primary.ru_number), _normalise(primary.holder))
    seen: set[tuple[str, str]] = set()
    for alt in alternatives:
        key = (_normalise(alt.ru_number), _normalise(alt.holder))
        if not key[0] or key == primary_key or key in seen:
            continue
        seen.add(key)
        payloads.append(_alternative_payload(alt))
    raw = dict(primary.raw) if isinstance(primary.raw, dict) else {}
    if payloads:
        raw["registry_alternatives"] = payloads[:5]
    return replace(primary, raw=raw)


def install_registry_query_fallbacks() -> None:
    global _ORIGINAL_CHECK
    if _ORIGINAL_CHECK is not None:
        return

    from bot.services.registry import RegistryService

    _ORIGINAL_CHECK = RegistryService._check_elk
    original_check = _ORIGINAL_CHECK
    cache: dict[tuple[str, str], tuple[list[RegistryRecord], str | None]] = {}
    inflight: dict[tuple[str, str], asyncio.Task[tuple[list[RegistryRecord], str | None]]] = {}
    records_by_ru: dict[str, list[RegistryRecord]] = {}
    records_by_card: dict[str, RegistryRecord] = {}

    def remember(records: list[RegistryRecord]) -> list[RegistryRecord]:
        """Сохраняет уже подтверждённые карточки независимо от поискового запроса.

        Один и тот же ELK record часто находится по разным строкам одной закупки.
        Запоминаем его по URL карточки и номеру РУ, чтобы соседняя позиция не
        скачивала ту же страницу повторно и не теряла кандидата при 429.
        """
        canonical: list[RegistryRecord] = []
        for record in records:
            card_key = _normalise(record.card_url)
            if card_key:
                existing = records_by_card.get(card_key)
                if existing is not None:
                    record = existing
                else:
                    records_by_card[card_key] = record
            ru_key = _normalise(record.ru_number)
            if ru_key:
                bucket = records_by_ru.setdefault(ru_key, [])
                identity = (
                    _normalise(record.card_url),
                    _normalise(record.holder),
                    _normalise(record.product_name),
                )
                if not any(
                    (
                        _normalise(item.card_url),
                        _normalise(item.holder),
                        _normalise(item.product_name),
                    ) == identity
                    for item in bucket
                ):
                    bucket.append(record)
            canonical.append(record)
        return _dedup_records(canonical)

    async def shared_check(
        self: RegistryService,
        *,
        name: str | None,
        ru: str | None,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], str | None]:
        key = (_normalise(name), _normalise(ru))
        cached = cache.get(key)
        if cached is not None:
            return cached

        # Самое важное для пакетной закупки: если карточка этого РУ уже была
        # успешно прочитана для соседней позиции, повторно Firecrawl не вызываем.
        if ru:
            reused = records_by_ru.get(_normalise(ru))
            if reused:
                logger.info(
                    "ELK shared card cache: РУ %s переиспользовано (%s карточек)",
                    ru,
                    len(reused),
                    extra=log_extra(request_id),
                )
                result = (list(reused), None)
                cache[key] = result
                return result

        task = inflight.get(key)
        if task is None:
            task = asyncio.create_task(original_check(self, name, ru, request_id))
            inflight[key] = task
        try:
            result = await task
        finally:
            if inflight.get(key) is task and task.done():
                inflight.pop(key, None)
        if result[0]:
            remembered = remember(result[0])
            result = (remembered, result[1])
            cache[key] = result
            # Создаём алиасы по номеру РУ. Следующий exact-RU hint получит
            # карточку из памяти даже если она была найдена обычным name search.
            for record in remembered:
                ru_key = _normalise(record.ru_number)
                if ru_key:
                    alias_key = ("", ru_key)
                    alias_records = records_by_ru.get(ru_key, [record])
                    cache[alias_key] = (list(alias_records), None)
        return result

    async def generic_candidates(
        self: RegistryService, name: str, request_id: int | None
    ) -> tuple[list[RegistryRecord], list[str]]:
        records: list[RegistryRecord] = []
        errors: list[str] = []
        exact_records, exact_error = await shared_check(self, name=name, ru=None, request_id=request_id)
        records.extend(exact_records)
        if exact_error:
            errors.append(exact_error)

        # Если точный поиск уже дал уверенную карточку, делаем максимум один
        # нормализованный проход для поиска альтернативных РУ. Иначе пробуем до 3 вариантов.
        variants = _query_variants(name)
        max_variants = 1 if any(_generic_confident(name, r) for r in exact_records) else len(variants)
        for variant in variants[:max_variants]:
            variant_records, variant_error = await shared_check(
                self, name=variant, ru=None, request_id=request_id
            )
            records.extend(variant_records)
            if variant_error:
                errors.append(variant_error)
            if variant_records and any(_generic_confident(name, r) for r in variant_records):
                # После первого уверенного расширенного результата дальше не расширяем запрос.
                break
        return _dedup_records(records), errors

    async def cyolyclone_hint(
        self: RegistryService,
        name: str,
        kind: str,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], list[RegistryRecord]]:
        """Совместимость со старым edge-case; не используется для другой продукции."""
        primary_ru = _GEMATOLOG_ABO_RU if kind in {"a", "b"} else _GEMATOLOG_ANTI_D_PRIMARY_RU
        (primary_records, _), (alt_records, _) = await asyncio.gather(
            shared_check(self, name=None, ru=primary_ru, request_id=request_id),
            shared_check(self, name=None, ru=_MEDIKLON_ABO_RH_KELL_RU, request_id=request_id),
        )
        primary = [
            r for r in primary_records
            if _ru_matches(r, primary_ru)
            and _holder_matches(r, "гематолог")
            and _antigen_matches(kind, r)
        ]
        alternatives = [
            r for r in alt_records
            if _ru_matches(r, _MEDIKLON_ABO_RH_KELL_RU)
            and _holder_matches(r, "медиклон")
            and _antigen_matches(kind, r)
        ]
        if not alternatives:
            family_records, _ = await shared_check(
                self, name=_MEDIKLON_FAMILY_QUERY, ru=None, request_id=request_id
            )
            alternatives = [
                r for r in family_records
                if _ru_matches(r, _MEDIKLON_ABO_RH_KELL_RU)
                and _holder_matches(r, "медиклон")
                and _antigen_matches(kind, r)
            ]
        return primary, alternatives

    async def _check_elk_with_fallbacks(
        self: RegistryService,
        name: str | None,
        ru_number: str | None,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], str | None]:
        if ru_number or not name:
            return await shared_check(self, name=name, ru=ru_number, request_id=request_id)

        # 1) Универсальный путь для любой медпродукции.
        candidates, errors = await generic_candidates(self, name, request_id)
        confident = [record for record in candidates if _generic_confident(name, record)]
        confident.sort(key=lambda record: _identity_score(name, record), reverse=True)

        # 2) Известный проблемный кейс может уточнить primary, но не меняет общую архитектуру.
        kind = _antigen_kind(name)
        hinted_primary: list[RegistryRecord] = []
        hinted_alternatives: list[RegistryRecord] = []
        if kind is not None:
            hinted_primary, hinted_alternatives = await cyolyclone_hint(self, name, kind, request_id)

        if hinted_primary:
            primary = hinted_primary[0]
            pool = _dedup_records([*hinted_alternatives, *confident])
        elif confident:
            primary = confident[0]
            pool = confident[1:]
        else:
            # Не угадываем по слабому совпадению. Возвращаем понятную ошибку вместо
            # старого сообщения «карточка найдена», если фактически уверенной записи нет.
            logger.info(
                "ELK universal fallback: «%s» — уверенной карточки нет; кандидатов %s",
                name,
                len(candidates),
                extra=log_extra(request_id),
            )
            return [], (errors[-1] if errors else "официальная карточка ELK по этому наименованию не найдена")

        # Альтернативами считаем только записи, которые сами уверенно соответствуют исходной позиции.
        alternatives = [
            record for record in pool
            if _generic_confident(name, record)
            and (_normalise(record.ru_number), _normalise(record.holder))
            != (_normalise(primary.ru_number), _normalise(primary.holder))
        ]
        primary = _attach_alternatives(primary, alternatives)
        logger.info(
            "ELK universal fallback: «%s» → %s / %s; альтернатив %s",
            name,
            primary.ru_number or "—",
            primary.holder or "—",
            len(alternatives),
            extra=log_extra(request_id),
        )
        return [primary], None

    RegistryService._check_elk = _check_elk_with_fallbacks  # type: ignore[method-assign]
