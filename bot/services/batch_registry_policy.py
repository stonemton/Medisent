"""Whole-list registry arbitration with non-blocking missing-RU handling.

Verified candidates are arbitrated by GPT once for the batch. A genuinely
ambiguous set of candidates may still require clarification, but having no
verified RU candidate is not a procurement stop condition: supplier discovery
continues and the RFQ asks the supplier for RU evidence.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from aiogram.types import Message

from bot import texts
from bot.db import repo
from bot.db.session import session_scope
from bot.services.batch_intake import ProcurementBatch, serialise_batch
from bot.services.batch_registry_agent import choose_registry_batch
from bot.services.registry import get_registry_service
from bot.services.registry_endpoints import RegistryRecord
from bot.services.registry_primary_policy import suspend_registry_primary_agent

_INSTALLED = False
_ORIGINAL_START_BATCH = None


def _norm(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").lower().replace("ё", "е") if ch.isalnum())


def _alt_payload(record: RegistryRecord) -> dict[str, Any]:
    return {
        "ru_number": record.ru_number, "holder": record.holder,
        "product_name": record.product_name, "valid": record.valid,
        "status_text": record.status_text, "card_url": record.card_url,
        "registry": record.registry,
        "source": record.raw.get("source") if isinstance(record.raw, dict) else None,
    }


def _record_from_alt(item: dict[str, Any], template: RegistryRecord) -> RegistryRecord:
    return RegistryRecord(
        registry=str(item.get("registry") or template.registry),
        ru_number=item.get("ru_number"), holder=item.get("holder"),
        product_name=item.get("product_name"), valid=item.get("valid"),
        status_text=item.get("status_text"), card_url=item.get("card_url"),
        raw={"source": item.get("source") or "registry_alternative"},
    )


def _candidates(record: RegistryRecord | None) -> list[RegistryRecord]:
    if record is None:
        return []
    rows = [record]
    raw = record.raw if isinstance(record.raw, dict) else {}
    for item in raw.get("registry_alternatives", []):
        if isinstance(item, dict):
            rows.append(_record_from_alt(item, record))
    out: list[RegistryRecord] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (_norm(row.ru_number), _norm(row.holder))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _selected_record(candidates: list[RegistryRecord], *, primary_index: int,
                     confidence: float, needs_review: bool, reasoning: str,
                     used_llm: bool) -> RegistryRecord | None:
    if not candidates:
        return None
    selected = candidates[primary_index] if 0 <= primary_index < len(candidates) else candidates[0]
    rest = [row for row in candidates if row is not selected]
    raw = dict(selected.raw) if isinstance(selected.raw, dict) else {}
    if rest:
        raw["registry_alternatives"] = [_alt_payload(row) for row in rest[:5]]
    raw["agent_decision"] = {
        "primary_index": primary_index, "confidence": confidence,
        "needs_review": needs_review, "reasoning": reasoning,
        "used_llm": used_llm, "batch": True,
    }
    return replace(selected, raw=raw)


def install_batch_registry_policy() -> None:
    global _INSTALLED, _ORIGINAL_START_BATCH
    if _INSTALLED:
        return
    from bot.handlers import intake
    _ORIGINAL_START_BATCH = intake._start_batch

    async def batch_start(message: Message, batch: ProcurementBatch, input_kind: str) -> None:
        items = batch.recognised_items
        if len(items) < 2:
            assert _ORIGINAL_START_BATCH is not None
            await _ORIGINAL_START_BATCH(message, batch, input_kind)
            return

        async with session_scope() as session:
            request = await repo.create_request(
                session, product=batch.product_text(), raw_input=serialise_batch(batch),
                input_kind=f"{input_kind}-batch",
            )
            request_id, token = int(request.id), request.token
            await repo.attach_orphan_api_calls(session, request_id, operation_prefix="intake.")

        intro = [f"<b>{texts.esc(token)} · распознано позиций: {len(items)}</b>", ""]
        intro.extend(texts.esc(item.line(i)) for i, item in enumerate(items, start=1))
        await intake._send_chunks(message, intro, parse_mode="HTML")
        if not intake.get_settings().search_enabled:
            await message.answer(texts.SEARCH_OFF)
            return

        await message.answer(
            "🔎 Проверяю позиции по реестру Росздравнадзора. Отсутствие найденного РУ "
            "не остановит закупку: по таким строкам продолжу поиск поставщиков и запрошу РУ у них."
        )
        service = get_registry_service()
        with suspend_registry_primary_agent():
            checks = await asyncio.gather(*[
                service.check_product(item.product, request_id=request_id, cache=False) for item in items
            ])

        candidate_groups = [_candidates(registry.best) for registry in checks]
        decisions = await choose_registry_batch(
            products=[item.product for item in items], candidate_groups=candidate_groups,
            request_id=request_id,
        )
        selected = [
            _selected_record(candidates, primary_index=decision.primary_index,
                             confidence=decision.confidence, needs_review=decision.needs_review,
                             reasoning=decision.reasoning, used_llm=decision.used_llm)
            for candidates, decision in zip(candidate_groups, decisions, strict=True)
        ]

        lines = ["<b>Проверка позиций по РУ</b>", ""]
        review_rows: list[tuple[int, str, str]] = []
        missing_ru_count = 0
        for index, (item, candidates, best, decision) in enumerate(
            zip(items, candidate_groups, selected, decisions, strict=True), start=1
        ):
            if not candidates:
                missing_ru_count += 1
                lines.append(
                    f"⚠️ {index}. {texts.esc(item.product)}\n"
                    "   РУ автоматически не подтверждено · закупку продолжаю · РУ запрошу у поставщика"
                )
                continue
            if best is None or decision.primary_index == -1:
                reason = decision.reasoning or "между подтверждёнными вариантами нельзя безопасно выбрать"
                review_rows.append((index, item.product, reason))
                lines.append(f"⚠️ {index}. {texts.esc(item.product)} — нужно уточнение: {texts.esc(reason)}")
                continue

            status = "действует" if best.valid is True else (
                "не действует" if best.valid is False else "статус не определён"
            )
            lines.append(
                f"✅ {index}. {texts.esc(item.product)}\n"
                f"   РУ: <b>{texts.esc(best.ru_number or '—')}</b> · {status}\n"
                f"   Держатель: {texts.esc(best.holder or 'не указан')}"
            )
            lines.extend(intake._alternate_ru_lines(best, indent="   "))
            if decision.reasoning:
                lines.append(f"   🧠 {texts.esc(decision.reasoning)}")
            if decision.needs_review and decision.confidence < 0.70:
                review_rows.append((index, item.product, decision.reasoning))

        await intake._send_chunks(message, lines, parse_mode="HTML")

        # Only ambiguity between real verified candidates blocks automatic choice.
        # A total absence of RU candidates does not.
        if review_rows:
            details = [
                f"{index}. {texts.esc(product)} — {texts.esc(reason or 'нужна дополнительная проверка')}"
                for index, product, reason in review_rows[:5]
            ]
            await message.answer(
                "⚠️ По этим строкам есть несколько реально разных подтверждённых вариантов, "
                "поэтому выбор наугад может привести к закупке другого изделия:\n\n"
                + "\n".join(details)
                + "\n\nПришлите точное наименование, модель/артикул или номер РУ по спорной позиции.",
                parse_mode="HTML",
            )
            return

        note = ""
        if missing_ru_count:
            note = (
                f"\n\nПо {missing_ru_count} поз. РУ пока не подтверждено. Это не блокирует поиск: "
                "в КП потребую номер и копию/официальную ссылку на действующее РУ либо письменное "
                "подтверждение, что регистрация для товара не требуется."
            )
        await message.answer(
            "🧠 Реестровая проверка завершена. Подтверждённые РУ использую как основной ориентир; "
            "неподтверждённые строки передаю в закупочный поиск без выдумывания РУ."
            + note + "\n\nИскать производителя и поставщиков сразу по всему списку?",
            parse_mode="HTML",
            reply_markup=intake._batch_keyboard(request_id).as_markup(),
        )

    intake._start_batch = batch_start
    _INSTALLED = True
