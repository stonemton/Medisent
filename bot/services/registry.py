"""Проверка изделий в реестрах Росздравнадзора."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from bot.config import get_settings
from bot.db.models import RegistryState
from bot.logging_setup import log_extra
from bot.services import pricing, registry_endpoints as endpoints
from bot.services.http import ApiClient
from bot.services.registry_endpoints import RegistryRecord

logger = logging.getLogger(__name__)
Outcome = tuple[list[RegistryRecord], str | None]

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"
ELK_CARD_RE = re.compile(
    r"^https://elk\.roszdravnadzor\.gov\.ru/widget/med-product/(\d+)(?:[/?#].*)?$",
    re.I,
)
RU_NUMBER_RE = re.compile(
    r"\b(?:РЗН|ФСР|ФСЗ)\s*(?:№\s*)?\d{4}/\d+(?:[-/]\d+)?\b",
    re.I | re.UNICODE,
)
ORG_NAME_RE = re.compile(
    r"\b(?:ООО|АО|ПАО|ОАО|ЗАО|НАО)\s*[«\"“]?[^\n;|]{2,120}?[»\"”]?(?=$|\n|;|\|)",
    re.I | re.UNICODE,
)


def derive_state(outcomes: Sequence[Outcome]) -> str:
    if not outcomes:
        return RegistryState.UNAVAILABLE
    if any(records for records, _ in outcomes):
        return RegistryState.FOUND
    if any(error for _, error in outcomes):
        return RegistryState.UNAVAILABLE
    return RegistryState.NOT_FOUND


@dataclass(slots=True)
class RegistryResult:
    state: str
    records: list[RegistryRecord] = field(default_factory=list)
    checked_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    errors: dict[str, str] = field(default_factory=dict)
    from_cache: bool = False

    @property
    def best(self) -> RegistryRecord | None:
        if not self.records:
            return None
        return sorted(self.records, key=lambda r: (r.valid is not True, r.registry != "elk"))[0]

    @property
    def unavailable(self) -> bool:
        return self.state == RegistryState.UNAVAILABLE

    def as_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "errors": self.errors,
            "records": [
                {
                    "registry": r.registry,
                    "ru_number": r.ru_number,
                    "holder": r.holder,
                    "product_name": r.product_name,
                    "valid": r.valid,
                    "status_text": r.status_text,
                    "card_url": r.card_url,
                }
                for r in self.records
            ],
        }


def cache_key(name: str, ru_number: str | None) -> str:
    normalised = " ".join(name.lower().split())
    payload = f"{normalised}|{(ru_number or '').strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_STOP_WORDS = {
    "изделие", "изделия", "изделий", "медицинское", "медицинский", "медицинская",
    "напиши", "наименование", "название", "нужен", "нужна", "нужно", "шт", "штук",
    "для", "при", "или", "и", "с", "в", "на", "по", "из", "от", "номер",
}


def _significant_terms(text: str) -> list[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{3,}", (text or "").lower())
    return [w for w in words if w not in _STOP_WORDS and not w.isdigit()]


def _record_matches_query(record: RegistryRecord, query: str) -> bool:
    terms = _significant_terms(query)
    if not terms:
        return False
    raw_text = ""
    if isinstance(record.raw, dict):
        raw_text = str(record.raw.get("text") or "")
        cells = record.raw.get("cells") or []
        if isinstance(cells, list):
            raw_text += " " + " ".join(str(x) for x in cells)
    haystack = " ".join(
        str(x or "") for x in (record.product_name, record.holder, record.ru_number, raw_text)
    ).lower()
    hits = sum(1 for term in set(terms) if term in haystack)
    return hits >= (1 if len(set(terms)) <= 2 else 2)


def _clean_md_value(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"^[#>*_`\-\s]+|[#>*_`\s]+$", "", value).strip()
    return value or None


def _field_after_label(text: str, *labels: str) -> str | None:
    for label in labels:
        escaped = re.escape(label)
        patterns = (
            rf"(?im)^\s*(?:[#>*_`-]+\s*)?{escaped}\s*(?:[*_`]*)\s*$\n+\s*([^\n]+)",
            rf"(?im)^\s*(?:[#>*_`-]+\s*)?{escaped}\s*[:—-]\s*([^\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return _clean_md_value(match.group(1))
    return None


def _holder_from_context(text: str) -> str | None:
    """Резервно извлекает юрлицо из карточки ELK, когда подпись поля изменилась."""
    labels = (
        "держател",
        "заявител",
        "уполномоченн",
        "производител",
        "организац",
        "регистрационное удостоверение выдано",
    )
    lines = [re.sub(r"\s+", " ", line).strip(" *_`#>-\t") for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not line or not any(marker in line.lower() for marker in labels):
            continue
        # Иногда значение стоит на той же строке после двоеточия.
        same = ORG_NAME_RE.search(line)
        if same:
            return _clean_md_value(same.group(0))
        # В ELK значение обычно идёт следующей строкой/строками.
        for candidate in lines[index + 1 : index + 5]:
            found = ORG_NAME_RE.search(candidate)
            if found:
                return _clean_md_value(found.group(0))

    # Последний безопасный fallback: если на карточке вообще только одно
    # российское юрлицо, оно гораздо надёжнее «держатель не указан».
    matches = []
    for found in ORG_NAME_RE.finditer(text):
        value = _clean_md_value(found.group(0))
        if value and value not in matches:
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


def _parse_elk_card_text(text: str, url: str, query: str) -> RegistryRecord | None:
    ru_number = _field_after_label(
        text,
        "Регистрационный номер медицинского изделия",
        "Регистрационный номер",
        "Номер ЕРУЛ",
    )
    if not ru_number:
        match = RU_NUMBER_RE.search(text)
        ru_number = match.group(0).strip() if match else None

    product_name = _field_after_label(
        text,
        "Наименование медицинского изделия",
        "Наименование изделия",
    )
    status_text = _field_after_label(text, "Статус", "Статус регистрационного удостоверения")
    holder = _field_after_label(
        text,
        "Наименования организации - уполномоченного представителя производителя (изготовителя) медицинского изделия",
        "Наименование организации - уполномоченного представителя производителя (изготовителя) медицинского изделия",
        "Наименования организации - производителя медицинского изделия или организации - изготовителя медицинского изделия",
        "Наименование организации - производителя медицинского изделия или организации - изготовителя медицинского изделия",
        "Наименование организации, на имя которой выдано регистрационное удостоверение",
        "Наименование организации, на имя которой выдано РУ",
        "Юридическое лицо, на имя которого выдано регистрационное удостоверение",
        "Организация-заявитель",
        "Наименование заявителя",
        "Заявитель",
        "Держатель регистрационного удостоверения",
        "Производитель",
    )
    holder = holder or _holder_from_context(text)

    if not ru_number and not product_name:
        return None

    lowered = (status_text or "").lower()
    valid: bool | None
    if any(marker in lowered for marker in ("аннулир", "прекращ", "приостанов", "недейств", "отмен")):
        valid = False
    elif "действ" in lowered:
        valid = True
    else:
        valid = None

    record = RegistryRecord(
        registry="elk",
        ru_number=ru_number,
        holder=holder,
        product_name=product_name,
        valid=valid,
        status_text=status_text,
        card_url=url,
        raw={"source": "official_elk_page", "text": text[:20000]},
    )
    return record if _record_matches_query(record, query) else None


class RegistryService:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = ApiClient(
            "registry",
            timeout_read=45.0,
            headers={
                "Accept": "application/json, text/html;q=0.9",
                "User-Agent": "Medisent-Bot/0.1 (medical device procurement)",
            },
        )
        self._firecrawl = ApiClient(
            "firecrawl",
            headers={
                "Authorization": f"Bearer {settings.firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            timeout_read=120.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._firecrawl.aclose()

    async def _scrape_elk_card(
        self, url: str, *, request_id: int | None = None
    ) -> tuple[str, str | None]:
        """Загружает именно найденную официальную карточку ELK целиком."""
        result = await self._firecrawl.post(
            FIRECRAWL_SCRAPE_URL,
            operation="elk.card_scrape",
            request_id=request_id,
            cost_usd=pricing.flat_cost("firecrawl"),
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": False,
                "waitFor": 5000,
                "timeout": 60000,
            },
        )
        if not result.ok:
            return "", result.error or "карточка ELK не загрузилась"
        payload = (result.json or {}).get("data") or {}
        if not isinstance(payload, dict):
            return "", "Firecrawl вернул карточку ELK в неизвестном формате"
        metadata = payload.get("metadata") or {}
        title = str(metadata.get("title") or "") if isinstance(metadata, dict) else ""
        description = str(metadata.get("description") or "") if isinstance(metadata, dict) else ""
        markdown = str(payload.get("markdown") or "")
        text = "\n".join((title, description, markdown)).strip()
        if not text:
            return "", "карточка ELK пустая"
        return text, None

    async def _check_elk(
        self, name: str | None, ru_number: str | None, request_id: int | None
    ) -> tuple[list[RegistryRecord], str | None]:
        if not self._settings.firecrawl_api_key:
            return [], "для проверки официальных карточек ELK нужен FIRECRAWL_API_KEY"

        query = (ru_number or name or "").strip()
        if not query:
            return [], "пустой запрос к ELK"

        result = await self._firecrawl.post(
            FIRECRAWL_SEARCH_URL,
            operation="elk.official_search",
            request_id=request_id,
            cost_usd=pricing.flat_cost("firecrawl"),
            json={
                "query": f'"{query}" inurl:/widget/med-product/',
                "limit": 8,
                "sources": ["web"],
                "includeDomains": ["elk.roszdravnadzor.gov.ru"],
                "country": "RU",
                "timeout": 60000,
                "ignoreInvalidURLs": True,
            },
        )
        if not result.ok:
            return [], result.error or "поиск официальных карточек ELK недоступен"

        payload = result.json or {}
        data = payload.get("data") or {}
        rows = data.get("web") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return [], "Firecrawl не вернул список официальных карточек ELK"

        logger.info("ELK: поиск по «%s» вернул %s результатов", query, len(rows), extra=log_extra(request_id))

        records: list[RegistryRecord] = []
        errors: list[str] = []
        seen_urls: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not ELK_CARD_RE.match(url) or url in seen_urls:
                continue
            seen_urls.add(url)

            # Search отдаёт только короткий сниппет карточки (в нашем случае ~127
            # символов). Поэтому после обнаружения официального URL всегда
            # скрейпим саму карточку целиком и только её считаем источником РУ.
            full_text, scrape_error = await self._scrape_elk_card(url, request_id=request_id)
            if scrape_error:
                errors.append(f"{url}: {scrape_error}")
                logger.warning("ELK: карточка %s не загрузилась: %s", url, scrape_error, extra=log_extra(request_id))
                continue

            record = _parse_elk_card_text(full_text, url, query)
            logger.info(
                "ELK: карточка %s len=%s ru=%r holder=%r product=%r",
                url,
                len(full_text),
                record.ru_number if record else None,
                record.holder if record else None,
                (record.product_name[:160] if record and record.product_name else None),
                extra=log_extra(request_id),
            )
            if record is not None:
                records.append(record)

        if records:
            logger.info("ELK: по «%s» подтверждено карточек %s", query, len(records), extra=log_extra(request_id))
            return records, None

        if errors:
            return [], "официальные карточки ELK найдены, но не загрузились/не разобрались: " + "; ".join(errors[:3])
        return [], "официальная карточка ELK найдена поиском, но РУ из неё извлечь не удалось"

    async def _check_misearch(
        self, name: str | None, ru_number: str | None, request_id: int | None
    ) -> tuple[list[RegistryRecord], str | None]:
        result = await self._client.get(
            self._settings.registry_misearch_url,
            operation="misearch.search",
            request_id=request_id,
            params=endpoints.build_misearch_params(name=name, ru_number=ru_number),
            expect_json=False,
        )
        if not result.ok:
            return [], result.error or "misearch недоступен"
        outcome = endpoints.parse_misearch_html(result.text)
        if not outcome.understood:
            return [], f"страница misearch не разобрана: {outcome.note}"
        return outcome.records, None

    async def check_product(
        self,
        name: str,
        ru_number: str | None = None,
        *,
        request_id: int | None = None,
        session: Any = None,
        cache: bool = False,
    ) -> RegistryResult:
        extra = log_extra(request_id)
        key = cache_key(name, ru_number)

        cached = None
        if session is not None:
            from bot.db import repo
            cached = await repo.get_registry_cache(session, key, self._settings.registry_cache_days)
        elif cache:
            from bot.db import repo
            from bot.db.session import session_scope
            async with session_scope() as own:
                cached = await repo.get_registry_cache(own, key, self._settings.registry_cache_days)

        if cached is not None:
            payload = cached.payload
            return RegistryResult(
                state=cached.state,
                records=[
                    RegistryRecord(
                        registry=r.get("registry", "?"),
                        ru_number=r.get("ru_number"),
                        holder=r.get("holder"),
                        product_name=r.get("product_name"),
                        valid=r.get("valid"),
                        status_text=r.get("status_text"),
                        card_url=r.get("card_url"),
                        raw={},
                    )
                    for r in payload.get("records", [])
                ],
                checked_at=cached.checked_at,
                errors=payload.get("errors", {}),
                from_cache=True,
            )

        logger.info("Реестр: проверяю «%s» (РУ %s)", name, ru_number or "—", extra=extra)
        (elk_records, elk_error), (mi_records, mi_error) = await asyncio.gather(
            self._check_elk(name, ru_number, request_id),
            self._check_misearch(name, ru_number, request_id),
        )

        records = [*elk_records, *mi_records]
        errors: dict[str, str] = {}
        if elk_error:
            errors["elk"] = elk_error
        if mi_error:
            errors["misearch"] = mi_error

        if records:
            state = RegistryState.FOUND
        elif elk_error is None:
            state = RegistryState.NOT_FOUND
        else:
            state = derive_state([(elk_records, elk_error), (mi_records, mi_error)])

        result = RegistryResult(state=state, records=records, errors=errors)
        if state == RegistryState.UNAVAILABLE:
            logger.warning("Реестр: проверить «%s» не удалось — %s", name, errors, extra=extra)
        else:
            logger.info("Реестр: «%s» → %s, записей %s", name, state, len(records), extra=extra)

        if state != RegistryState.UNAVAILABLE:
            if session is not None:
                from bot.db import repo
                await repo.put_registry_cache(session, key, state, result.as_payload())
            elif cache:
                from bot.db import repo
                from bot.db.session import session_scope
                async with session_scope() as own:
                    await repo.put_registry_cache(own, key, state, result.as_payload())
        return result

    async def check_unrega(
        self, name: str, holder: str | None = None, *, request_id: int | None = None
    ) -> RegistryResult:
        extra = log_extra(request_id)
        query = f"{name} {holder}".strip() if holder else name

        if not _significant_terms(query):
            logger.info("unrega: запрос «%s» слишком общий — пропускаю", query, extra=extra)
            return RegistryResult(state=RegistryState.NOT_FOUND, records=[])

        result = await self._client.get(
            self._settings.registry_unrega_url,
            operation="unrega.search",
            request_id=request_id,
            params=endpoints.build_unrega_params(name=query),
            expect_json=False,
        )
        records: list[RegistryRecord] = []
        error: str | None = None
        if not result.ok:
            error = result.error or "unrega недоступен"
        else:
            outcome = endpoints.parse_unrega_html(result.text)
            if not outcome.understood:
                error = f"страница unrega не разобрана: {outcome.note}"
            else:
                records = [r for r in outcome.records if _record_matches_query(r, query)]

        state = derive_state([(records, error)])
        if error:
            logger.warning("unrega: проверить «%s» не удалось — %s", name, error, extra=extra)
        else:
            logger.info("unrega: «%s» → %s, релевантных писем %s", name, state, len(records), extra=extra)
        return RegistryResult(
            state=state,
            records=records,
            errors={"unrega": error} if error else {},
        )


_service: RegistryService | None = None


def get_registry_service() -> RegistryService:
    global _service
    if _service is None:
        _service = RegistryService()
    return _service


async def close_registry_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None
