"""Central decision layer for Medisent.

The agent never discovers registry facts itself. It receives already verified
facts from tools/services, arbitrates ambiguous matches, and plans the next
procurement step. Every model decision is bounded by an explicit allow-list and
validated before the application uses it.
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

WORKFLOW_ACTIONS = {
    "search_suppliers",
    "ask_clarification",
    "prepare_rfqs",
    "split_procurement",
    "compare_quotes",
    "finish",
}

WORKFLOW_DECISION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "user_message": {"type": "STRING"},
        "clarification_question": {"type": "STRING"},
        "reasoning": {"type": "STRING"},
    },
    "required": [
        "action",
        "confidence",
        "user_message",
        "clarification_question",
        "reasoning",
    ],
}


@dataclass(slots=True)
class RegistryDecision:
    primary_index: int
    confidence: float
    needs_review: bool
    reasoning: str
    used_llm: bool = True


@dataclass(slots=True)
class WorkflowDecision:
    action: str
    confidence: float
    user_message: str
    clarification_question: str = ""
    reasoning: str = ""
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
    """Choose one verified registry candidate without allowing invented facts."""
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


def _deterministic_workflow_fallback(stage: str, state: dict[str, Any], reason: str) -> WorkflowDecision:
    items = state.get("items") if isinstance(state, dict) else None
    if stage == "registry_review":
        rows = items if isinstance(items, list) else []
        unresolved = any(bool(row.get("unresolved")) for row in rows if isinstance(row, dict))
        review = any(bool(row.get("needs_review")) for row in rows if isinstance(row, dict))
        if unresolved or review:
            return WorkflowDecision(
                "ask_clarification",
                0.0,
                "По части позиций остаётся неоднозначность. Перед поиском поставщиков лучше уточнить изделие.",
                "Пришлите точное наименование, модель/артикул или номер РУ по спорной позиции.",
                reason,
                used_llm=False,
            )
        return WorkflowDecision(
            "search_suppliers",
            0.0,
            "Позиции идентифицированы достаточно уверенно — можно переходить к поиску производителя и поставщиков.",
            reasoning=reason,
            used_llm=False,
        )
    if stage == "supplier_search":
        if bool(state.get("full_coverage")):
            return WorkflowDecision(
                "prepare_rfqs",
                0.0,
                "Есть подтверждённые каналы, закрывающие весь список. Следующий шаг — подготовить запросы КП.",
                reasoning=reason,
                used_llm=False,
            )
        if bool(state.get("split_plan")):
            return WorkflowDecision(
                "split_procurement",
                0.0,
                "Одного подтверждённого поставщика на весь список нет — закупку лучше разделить по найденным каналам.",
                reasoning=reason,
                used_llm=False,
            )
    return WorkflowDecision("finish", 0.0, "Текущий этап завершён.", reasoning=reason, used_llm=False)


async def plan_procurement_next(
    *,
    stage: str,
    state: dict[str, Any],
    request_id: int | None = None,
) -> WorkflowDecision:
    """Plan the next step from tool-produced state only.

    The model may choose only from WORKFLOW_ACTIONS. Invalid or unavailable LLM
    output falls back to a conservative deterministic plan.
    """
    settings = get_settings()
    payload = {
        "stage": stage,
        "state": state,
        "constraints": {
            "allowed_actions": sorted(WORKFLOW_ACTIONS),
            "facts_must_come_only_from_state": True,
        },
    }
    service = get_gemini_service()
    try:
        parsed = await service.generate_json(
            parts=[Part(text=json.dumps(payload, ensure_ascii=False, indent=2, default=str))],
            system_instruction=service.load_instruction("agent_workflow"),
            schema=WORKFLOW_DECISION_SCHEMA,
            model=getattr(settings, "llm_agent_model", settings.llm_report_model),
            request_id=request_id,
            operation=f"agent.workflow.{stage}",
            temperature=0.0,
        )
    except GeminiError as exc:
        logger.warning("Agent workflow fallback (%s): %s", stage, exc, extra=log_extra(request_id))
        return _deterministic_workflow_fallback(stage, state, f"LLM недоступна: {exc}")

    action = str(parsed.get("action") or "").strip()
    if action not in WORKFLOW_ACTIONS:
        logger.warning("Agent workflow invalid action %r", action, extra=log_extra(request_id))
        return _deterministic_workflow_fallback(stage, state, "модель вернула недопустимое действие")
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    user_message = str(parsed.get("user_message") or "").strip()[:1200]
    clarification = str(parsed.get("clarification_question") or "").strip()[:1200]
    reasoning = str(parsed.get("reasoning") or "").strip()[:1600]

    if action == "ask_clarification" and not clarification:
        clarification = "Уточните точное наименование, модель/артикул или номер РУ по спорной позиции."
    if not user_message:
        user_message = _deterministic_workflow_fallback(stage, state, reasoning).user_message

    logger.info(
        "Agent workflow %s → %s confidence %.2f: %s",
        stage,
        action,
        confidence,
        reasoning,
        extra=log_extra(request_id),
    )
    return WorkflowDecision(action, confidence, user_message, clarification, reasoning, used_llm=True)
