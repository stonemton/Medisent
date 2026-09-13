"""Agent arbitration for ambiguous verified registry matches.

Search remains deterministic and evidence-based. When the universal matcher has
more than one verified RU candidate, the central Medisent agent chooses among
those candidates only. It cannot invent a new RU. No extra Firecrawl calls are
performed here.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Iterator

from bot.services.agent import choose_registry_primary
from bot.services.registry_endpoints import RegistryRecord

_ORIGINAL_CHECK: Callable[..., Awaitable[tuple[list[RegistryRecord], str | None]]] | None = None
_SUSPEND_AGENT: ContextVar[bool] = ContextVar("registry_primary_policy_suspend_agent", default=False)


@contextmanager
def suspend_registry_primary_agent() -> Iterator[None]:
    """Temporarily keep deterministic registry ordering without per-item LLM calls.

    Used by batch intake so the whole procurement can be arbitrated in one GPT
    request instead of one model request per line.
    """
    token = _SUSPEND_AGENT.set(True)
    try:
        yield
    finally:
        _SUSPEND_AGENT.reset(token)


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", str(value or "").lower().replace("ё", "е"))


def _payload(record: RegistryRecord) -> dict[str, Any]:
    return {
        "ru_number": record.ru_number,
        "holder": record.holder,
        "product_name": record.product_name,
        "valid": record.valid,
        "status_text": record.status_text,
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
        status_text=item.get("status_text"),
        card_url=item.get("card_url"),
        raw={"source": item.get("source") or "registry_alternative"},
    )


def _dedup(records: list[RegistryRecord]) -> list[RegistryRecord]:
    out: list[RegistryRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (_norm(record.ru_number), _norm(record.holder))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def install_registry_primary_policy() -> None:
    global _ORIGINAL_CHECK
    if _ORIGINAL_CHECK is not None:
        return

    from bot.services.registry import RegistryService

    _ORIGINAL_CHECK = RegistryService._check_elk
    original = _ORIGINAL_CHECK

    async def wrapped(
        self: RegistryService,
        name: str | None,
        ru_number: str | None,
        request_id: int | None,
    ) -> tuple[list[RegistryRecord], str | None]:
        records, error = await original(self, name, ru_number, request_id)
        if ru_number or not name or not records or _SUSPEND_AGENT.get():
            return records, error

        current = records[0]
        raw = dict(current.raw) if isinstance(current.raw, dict) else {}
        alternatives = [x for x in raw.get("registry_alternatives", []) if isinstance(x, dict)]
        candidates = _dedup([current, *[_record_from_alt(x, current) for x in alternatives]])
        if len(candidates) <= 1:
            return records, error

        decision = await choose_registry_primary(
            product=name,
            candidates=candidates,
            request_id=request_id,
        )

        selected_index = decision.primary_index
        if selected_index < 0 or selected_index >= len(candidates):
            selected = current
        else:
            selected = candidates[selected_index]

        rest = [
            candidate
            for candidate in candidates
            if not (
                _norm(candidate.ru_number) == _norm(selected.ru_number)
                and _norm(candidate.holder) == _norm(selected.holder)
            )
        ]
        selected_raw = dict(selected.raw) if isinstance(selected.raw, dict) else {}
        if rest:
            selected_raw["registry_alternatives"] = [_payload(candidate) for candidate in rest[:5]]
        selected_raw["agent_decision"] = {
            "primary_index": decision.primary_index,
            "confidence": decision.confidence,
            "needs_review": decision.needs_review,
            "reasoning": decision.reasoning,
            "used_llm": decision.used_llm,
        }
        return [replace(selected, raw=selected_raw)], error

    RegistryService._check_elk = wrapped
