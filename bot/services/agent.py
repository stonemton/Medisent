"""Central decision layer for Medisent.

The agent never discovers registry facts itself. It receives already verified
candidates from tools/services, asks the configured LLM to arbitrate ambiguous
matches, and validates that the model selected only an existing candidate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from bot.config import get_settings
from bot.logging_setup import log_extra
from bot.services.gemini import GeminiError, Part, get_gemini_service
from bot.services.registry_endpoints import RegistryRecord

logger = logging.getLogger(__name__)

REGISTRY_DECISION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "primary_index": {"type": "INTEGER"},
        "confidence": {"type": "NUMBER"},
        "needs_review": {"type": "BOOLEAN"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["primary_index", "confidence", "needs_review", "reasoning"],
}


@dataclass(slots=True)
class RegistryDecision:
    primary_index: int
    confidence: float
    needs_review: bool
    reasoning: str
    used_llm: bool = True


def _candidate_payload(record: RegistryRecord, index: int) -> dict[str, Any]:
    raw_text = ""
    source = None
    if isinstance(record.raw, dict):
        source = record.raw.get("source")
        raw_text = str(record.raw.get("text") or "")[:5000]
    return {
        "index": index,
        "ru_number": record.ru_number,
        "holder": record.holder,
        "product_name": record.product_name,
        "valid": record.valid,
        "status_text": record.status_text,
        "card_url": record.card_url,
        "registry": record.registry,
        "source": source,
        "registry_excerpt": raw_text,
    }


async def choose_registry_primary(
    *,
    product: str,
    candidates: list[RegistryRecord],
    request_id: int | None = None,
    procurement_context: list[str] | None = None,
) -> RegistryDecision:
    """Choose one verified registry candidate without allowing invented facts.

    If the LLM fails or returns an invalid index, candidate 0 is preserved. This
    makes the agent an arbitration layer rather than a new point of failure.
    """
    if not candidates:
        return RegistryDecision(-1, 0.0, True, "нет кандидатов", used_llm=False)
    if len(candidates) == 1:
        return RegistryDecision(0, 1.0, False, "единственный подтверждённый кандидат", used_llm=False)

    settings = get_settings()
    payload = {
        "requested_product": product,
        "procurement_context": procurement_context or [],
        "candidates": [_candidate_payload(record, index) for index, record in enumerate(candidates)],
        "constraints": {
            "allowed_indexes": list(range(len(candidates))),
            "may_return_no_choice": -1,
            "facts_must_come_only_from_candidates": True,
        },
    }

    service = get_gemini_service()
    try:
        parsed = await service.generate_json(
            parts=[Part(text=json.dumps(payload, ensure_ascii=False, indent=2))],
            system_instruction=service.load_instruction("agent_registry"),
            schema=REGISTRY_DECISION_SCHEMA,
            model=getattr(settings, "llm_agent_model", settings.llm_report_model),
            request_id=request_id,
            operation="agent.registry",
            temperature=0.0,
        )
    except GeminiError as exc:
        logger.warning("Agent registry fallback to current primary: %s", exc, extra=log_extra(request_id))
        return RegistryDecision(0, 0.0, True, f"LLM недоступна: {exc}", used_llm=False)

    try:
        index = int(parsed.get("primary_index", -1))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        index, confidence = -1, 0.0
    needs_review = bool(parsed.get("needs_review", False))
    reasoning = str(parsed.get("reasoning") or "").strip()[:1200]

    if index == -1:
        logger.info(
            "Agent registry: «%s» → manual review (confidence %.2f): %s",
            product,
            confidence,
            reasoning,
            extra=log_extra(request_id),
        )
        return RegistryDecision(-1, confidence, True, reasoning or "модель не выбрала кандидата")
    if index < 0 or index >= len(candidates):
        logger.warning(
            "Agent registry returned invalid index %s for %s candidates; preserving current primary",
            index,
            len(candidates),
            extra=log_extra(request_id),
        )
        return RegistryDecision(0, 0.0, True, "модель вернула индекс вне списка", used_llm=False)

    logger.info(
        "Agent registry: «%s» → candidate %s/%s confidence %.2f review=%s: %s",
        product,
        index,
        len(candidates),
        confidence,
        needs_review,
        reasoning,
        extra=log_extra(request_id),
    )
    return RegistryDecision(index, confidence, needs_review, reasoning)
