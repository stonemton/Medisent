"""Prefer the official ELK JSON gateway over Firecrawl.

The existing registry implementation remains as a fallback because the public
ELK gateway has changed shape in the past. Useful direct JSON results cost no
Firecrawl credits; Firecrawl is used only when the direct gateway is unavailable,
cannot be parsed, or does not return a relevant record.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from bot.logging_setup import log_extra
from bot.services import registry_endpoints as endpoints
from bot.services.registry import RegistryService
from bot.services.registry_endpoints import RegistryRecord

logger = logging.getLogger(__name__)
_ORIGINAL_CHECK: Callable[..., Awaitable[tuple[list[RegistryRecord], str | None]]] | None = None


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", str(value or "").lower().replace("ё", "е"))


def _relevant(record: RegistryRecord, *, name: str | None, ru_number: str | None) -> bool:
    if ru_number:
        return bool(record.ru_number and _norm(record.ru_number) == _norm(ru_number))
    if not name:
        return True
    terms = [
        token for token in re.findall(r"[a-zа-яё0-9-]{3,}", name.lower())
        if token not in {"для", "крови", "человека", "жидкий", "готовый", "флаконе", "флакон"}
        and not token.isdigit()
    ]
    if not terms:
        return True
    haystack = " ".join(
        str(x or "") for x in (record.product_name, record.holder, record.ru_number)
    ).lower()
    hits = sum(1 for token in set(terms) if token in haystack)
    return hits >= (1 if len(set(terms)) <= 2 else 2)


async def _direct_elk(
    self: RegistryService,
    name: str | None,
    ru_number: str | None,
    request_id: int | None,
) -> tuple[list[RegistryRecord], str | None, bool]:
    """Return records, error, understood."""
    url = self._settings.registry_elk_base.rstrip("/") + endpoints.ELK_SEARCH_PATH
    result = await self._client.post(
        url,
        operation="elk.direct_search",
        request_id=request_id,
        json=endpoints.build_elk_query(name=name, ru_number=ru_number),
        expect_json=True,
        retries=1,
    )
    if not result.ok:
        return [], result.error or "прямой ELK gateway недоступен", False

    outcome = endpoints.parse_elk_payload(result.json)
    if not outcome.understood:
        return [], f"прямой ELK gateway не разобран: {outcome.note}", False

    records = [r for r in outcome.records if _relevant(r, name=name, ru_number=ru_number)]
    logger.info(
        "ELK direct: «%s» → разобрано %s, релевантных %s",
        ru_number or name or "",
        len(outcome.records),
        len(records),
        extra=log_extra(request_id),
    )
    return records, None, True


def install_registry_direct_policy() -> None:
    global _ORIGINAL_CHECK
    if _ORIGINAL_CHECK is not None:
        return
    _ORIGINAL_CHECK = RegistryService._check_elk
    original = _ORIGINAL_CHECK

    async def wrapped(
        self: RegistryService,
        name: str | None,
        ru_number: str | None,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], str | None]:
        direct_records, direct_error, understood = await _direct_elk(
            self, name, ru_number, request_id
        )
        if understood and direct_records:
            return direct_records, None

        reason = direct_error or "релевантных записей нет"
        logger.info(
            "ELK direct не дал полезного результата (%s) — включаю Firecrawl fallback",
            reason,
            extra=log_extra(request_id),
        )
        fallback_records, fallback_error = await original(self, name, ru_number, request_id)
        if fallback_records:
            return fallback_records, None
        if fallback_error:
            return [], fallback_error
        return [], (None if understood else direct_error)

    RegistryService._check_elk = wrapped  # type: ignore[method-assign]
