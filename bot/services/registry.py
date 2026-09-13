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
from bot.services import registry_endpoints as endpoints
from bot.services.http import ApiClient
from bot.services.registry_endpoints import RegistryRecord

logger = logging.getLogger(__name__)
Outcome = tuple[list[RegistryRecord], str | None]


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
    """Отбрасывает таблицы/служебные строки unrega, не относящиеся к изделию."""
    terms = _significant_terms(query)
    if not terms:
        return False
    haystack = " ".join(
        str(x or "")
        for x in (
            record.product_name,
            record.holder,
            record.ru_number,
            " ".join(record.raw.get("cells", [])) if isinstance(record.raw, dict) else "",
        )
    ).lower()
    hits = sum(1 for term in set(terms) if term in haystack)
    # Для короткого конкретного запроса достаточно одного сильного совпадения,
    # для длинного требуем минимум два, чтобы шапка таблицы не считалась письмом.
    return hits >= (1 if len(set(terms)) <= 2 else 2)


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

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _check_elk(
        self, name: str | None, ru_number: str | None, request_id: int | None
    ) -> tuple[list[RegistryRecord], str | None]:
        url = self._settings.registry_elk_base.rstrip("/") + endpoints.ELK_SEARCH_PATH
        result = await self._client.post(
            url,
            operation="elk.search",
            request_id=request_id,
            json=endpoints.build_elk_query(name=name, ru_number=ru_number),
        )
        if not result.ok:
            return [], result.error or "gateway elk недоступен"
        outcome = endpoints.parse_elk_payload(result.json)
        if not outcome.understood:
            return [], f"ответ elk не разобран: {outcome.note}"
        return outcome.records, None

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

        # ELK — текущий основной реестр. Если он дал понятный ответ, старый
        # misearch не имеет права превращать результат обратно в unavailable.
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

        # Не отправляем в реестр бессодержательные фразы вроде
        # «напиши наименование изделий»: они давали ложные совпадения с шапкой.
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
