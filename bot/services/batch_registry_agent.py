"""Single-call registry arbitration for a whole procurement batch.

The batch agent never discovers registry facts. It receives only candidates
already found by deterministic registry tools and may select only their indexes.
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

BATCH_REGISTRY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "decisions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "item_index": {"type": "INTEGER"},
                    "primary_index": {"type": "INTEGER"},
                    "confidence": {"type": "NUMBER"},
                    "needs_review": {"type": "BOOLEAN"},
                    "reasoning": {"type": "STRING"},
                },
                "required": [
                    "item_index",
                    "primary_index",
                    "confidence",
                    "needs_review",
                    "reasoning",
                ],
            },
        },
    },
    "required": ["decisions"],
}


@dataclass(slots=True)
class BatchRegistryDecision:
    item_index: int
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
        raw_text = str(record.raw.get("text") or "")[:3500]
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


async def choose_registry_batch(
    *,
    products: list[str],
    candidate_groups: list[list[RegistryRecord]],
    request_id: int | None = None,
) -> list[BatchRegistryDecision]:
    """Choose registry candidates for the complete batch in one LLM call."""
    decisions: list[BatchRegistryDecision | None] = [None] * len(products)
    pending: list[dict[str, Any]] = []

    for item_index, (product, candidates) in enumerate(zip(products, candidate_groups, strict=True)):
        if not candidates:
            decisions[item_index] = BatchRegistryDecision(
                item_index, -1, 0.0, True, "нет подтверждённых кандидатов", used_llm=False
            )
            continue
        if len(candidates) == 1:
            decisions[item_index] = BatchRegistryDecision(
                item_index, 0, 1.0, False, "единственный подтверждённый кандидат", used_llm=False
            )
            continue
        pending.append(
            {
                "item_index": item_index,
                "requested_product": product,
                "candidates": [_candidate_payload(record, i) for i, record in enumerate(candidates)],
                "allowed_indexes": list(range(len(candidates))),
            }
        )

    if pending:
        settings = get_settings()
        model = getattr(settings, "llm_batch_agent_model", "gpt-5.6-sol")
        payload = {
            "procurement_items": pending,
            "constraints": {
                "may_return_no_choice": -1,
                "facts_must_come_only_from_candidates": True,
                "preserve_item_index": True,
            },
        }
        service = get_gemini_service()
        try:
            parsed = await service.generate_json(
                parts=[Part(text=json.dumps(payload, ensure_ascii=False, indent=2))],
                system_instruction=service.load_instruction("agent_registry_batch"),
                schema=BATCH_REGISTRY_SCHEMA,
                model=model,
                request_id=request_id,
                operation="agent.registry_batch",
                temperature=0.0,
            )
        except GeminiError as exc:
            logger.warning(
                "Batch registry agent unavailable, preserving deterministic primaries: %s",
                exc,
                extra=log_extra(request_id),
            )
            for row in pending:
                idx = int(row["item_index"])
                decisions[idx] = BatchRegistryDecision(
                    idx, 0, 0.0, True, f"LLM недоступна: {exc}", used_llm=False
                )
        else:
            by_index: dict[int, dict[str, Any]] = {}
            raw_decisions = parsed.get("decisions")
            if isinstance(raw_decisions, list):
                for row in raw_decisions:
                    if not isinstance(row, dict):
                        continue
                    try:
                        idx = int(row.get("item_index", -1))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(products) and idx not in by_index:
                        by_index[idx] = row

            pending_indexes = {int(row["item_index"]) for row in pending}
            for idx in pending_indexes:
                row = by_index.get(idx)
                candidates = candidate_groups[idx]
                if row is None:
                    decisions[idx] = BatchRegistryDecision(
                        idx, 0, 0.0, True, "модель не вернула решение по позиции", used_llm=False
                    )
                    continue
                try:
                    primary = int(row.get("primary_index", -1))
                    confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0)))
                except (TypeError, ValueError):
                    primary, confidence = -1, 0.0
                reasoning = str(row.get("reasoning") or "").strip()[:1200]
                needs_review = bool(row.get("needs_review", False))
                if primary != -1 and not (0 <= primary < len(candidates)):
                    primary = -1
                    needs_review = True
                    reasoning = reasoning or "модель вернула индекс вне списка"
                if primary == -1:
                    needs_review = True
                decisions[idx] = BatchRegistryDecision(
                    idx, primary, confidence, needs_review, reasoning, used_llm=True
                )
                logger.info(
                    "Batch registry agent item %s → candidate %s/%s confidence %.2f review=%s: %s",
                    idx,
                    primary,
                    len(candidates),
                    confidence,
                    needs_review,
                    reasoning,
                    extra=log_extra(request_id),
                )

    return [
        decision
        if decision is not None
        else BatchRegistryDecision(i, -1, 0.0, True, "решение не сформировано", used_llm=False)
        for i, decision in enumerate(decisions)
    ]
