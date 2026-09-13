"""Enhanced batch procurement callback: agent decision -> real draft action.

This router is registered before the legacy intake router and owns `batch:*`
callbacks. It keeps external side effects safe: RFQ emails are drafted and
shown, but every send still requires the existing explicit mail approval.
"""
from __future__ import annotations

import asyncio

from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.db import repo
from bot.db.session import session_scope
from bot.pipeline import close_request
from bot.services.agent import plan_procurement_next
from bot.services.batch_intake import deserialise_batch
from bot.services.procurement_batch import run_batch_procurement
from bot.services.report import render
from bot.services.rfq_drafts import RfqDraft, RfqDraftError, prepare_rfq_draft

router = Router(name="batch_actions")


def _mail_keyboard(approval_id: int) -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Отправить", callback_data=f"mail:yes:{approval_id}"),
        InlineKeyboardButton(text="❌ Не отправлять", callback_data=f"mail:no:{approval_id}"),
    )
    return builder


def _role(row: object) -> str:
    value = getattr(row, "supplier_role", None)
    return str(value or "candidate")


def _evidence_label(row: object) -> str:
    role = _role(row)
    if role == "manufacturer":
        return "прямой производитель товарной линии; наличие и объём требуют подтверждения"
    if getattr(row, "site_claims", None) is True:
        return "товар/наличие подтверждены на сайте; объём требует подтверждения"
    if role == "official_distributor":
        return "официальный дистрибьютор; наличие и объём требуют подтверждения"
    return "поставщик найден для товарной линии; наличие и объём требуют подтверждения"


def _shortlist(report: object, limit: int = 3) -> list[object]:
    candidates = list(getattr(report, "candidates", []) or [])
    with_email = [row for row in candidates if getattr(row, "email", None)]
    selected: list[object] = []

    manufacturer = next((row for row in with_email if _role(row) == "manufacturer"), None)
    if manufacturer is not None:
        selected.append(manufacturer)

    for wanted in ("official_distributor", "seller", "candidate"):
        for row in with_email:
            if row in selected or _role(row) != wanted:
                continue
            selected.append(row)
            if len(selected) >= limit:
                return selected

    for row in with_email:
        if row not in selected:
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


async def _prepare_drafts(message: Message, *, request_id: int, product: str, report: object) -> None:
    rows = _shortlist(report)
    if not rows:
        await message.answer(
            "🧠 Агент решил перейти к запросам КП, но у выбранных каналов нет e-mail. "
            "Автоматическую отправку не делаю — нужны контакты."
        )
        return

    await message.answer(
        f"✉️ Агент выбрал действие: подготовить запросы КП. "
        f"Составляю {len(rows)} черновик(а) параллельно; ничего не отправляю без подтверждения."
    )

    tasks = [
        prepare_rfq_draft(
            request_id=request_id,
            product=product,
            supplier_id=int(getattr(row, "supplier_id", 0) or 0),
        )
        for row in rows
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    prepared = 0
    for row, result in zip(rows, results, strict=True):
        supplier_name = str(getattr(row, "supplier_name", "поставщик"))
        if isinstance(result, Exception):
            reason = str(result) if isinstance(result, RfqDraftError) else "не удалось составить письмо"
            await message.answer(
                f"⚠️ <b>{texts.esc(supplier_name)}</b>: черновик не создан — {texts.esc(reason)}",
                parse_mode="HTML",
            )
            continue

        draft: RfqDraft = result
        prepared += 1
        evidence = _evidence_label(row)
        await message.answer(
            f"<b>{texts.esc(draft.supplier_name)}</b>\n"
            f"Статус канала: {texts.esc(evidence)}\n"
            f"E-mail: {texts.esc(draft.email)}\n\n"
            f"<b>Черновик запроса КП</b>\n{texts.esc(draft.body)}",
            parse_mode="HTML",
            reply_markup=_mail_keyboard(draft.approval_id).as_markup(),
        )

    if prepared:
        await message.answer(
            "✅ Черновики готовы. Нажмите «Отправить» только у тех поставщиков, которым действительно нужно направить запрос."
        )


@router.callback_query(F.data.startswith("batch:"))
async def batch_confirm(callback: CallbackQuery) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        return
    decision, raw_id = parts[1], parts[2]
    try:
        request_id = int(raw_id)
    except ValueError:
        return

    await callback.answer()
    if not isinstance(callback.message, Message):
        return

    async with session_scope() as session:
        request = await repo.get_request(session, request_id)
    if request is None:
        await callback.message.answer("Заявка не найдена или устарела.")
        return

    batch = deserialise_batch(request.raw_input)
    if batch is None or len(batch.recognised_items) < 2:
        await callback.message.answer("Не удалось восстановить список позиций. Пришлите заявку ещё раз.")
        return

    if decision != "yes":
        async with session_scope() as session:
            await close_request(session, request_id)
        await callback.message.answer("Хорошо. Пакетный поиск не запускаю.")
        return

    items = batch.recognised_items
    await callback.message.answer(
        f"🔎 Запускаю закупочный поиск по {len(items)} позициям. "
        "После поиска агент сам выберет следующий безопасный шаг."
    )

    result = await run_batch_procurement(
        master_request_id=request_id,
        items=items,
        master_product=request.product,
    )

    supplier_state: list[dict[str, object]] = []
    if result.report is not None:
        for row in result.report.candidates:
            supplier_state.append({
                "supplier_id": row.supplier_id,
                "supplier": row.supplier_name,
                "role": row.supplier_role,
                "has_email": bool(row.email),
                "site_claims": row.site_claims,
                "evidence_level": _evidence_label(row),
            })

    workflow = await plan_procurement_next(
        stage="supplier_search",
        state={
            "total_items": len(items),
            "full_coverage": result.report is not None,
            "full_coverage_suppliers": result.full_coverage_suppliers,
            "suppliers": supplier_state,
            "split_plan": [
                {"supplier": row.supplier_name, "covered_items": row.covered_items}
                for row in result.split_plan
            ],
            "errors": result.errors,
        },
        request_id=request_id,
    )

    if result.report is not None:
        await callback.message.answer(
            f"✅ Найдены {result.full_coverage_suppliers} канал(а) закупки для полного списка. "
            "Важно: это не означает подтверждённое наличие всех позиций — уровень доказательств указан по каждому каналу ниже."
        )
        for chunk in render(result.report):
            await callback.message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)

        if workflow.user_message:
            await callback.message.answer("🧠 " + workflow.user_message)

        if workflow.action == "prepare_rfqs":
            await _prepare_drafts(
                callback.message,
                request_id=request_id,
                product=request.product,
                report=result.report,
            )
        elif workflow.action == "ask_clarification":
            question = workflow.clarification_question or "Уточните, каким поставщикам готовить запросы КП."
            await callback.message.answer("❓ " + question)
        return

    if result.split_plan:
        lines = [
            "<b>Одного канала на весь список не найдено.</b>",
            "Оптимальный план разделения закупки:",
            "",
        ]
        for index, plan in enumerate(result.split_plan, start=1):
            lines.append(f"<b>{index}. {texts.esc(plan.supplier_name)}</b>")
            for product in plan.covered_items:
                lines.append(f"• {texts.esc(product)}")
        if result.errors:
            lines.extend(["", "Ошибки отдельных проходов: " + texts.esc("; ".join(result.errors))])
        await callback.message.answer("\n".join(lines), parse_mode="HTML")
        if workflow.user_message:
            await callback.message.answer("🧠 " + workflow.user_message)
        return

    error = "; ".join(result.errors) if result.errors else "подходящих каналов закупки не найдено"
    await callback.message.answer(texts.search_failed(error))
