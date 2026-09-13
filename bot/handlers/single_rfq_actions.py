"""RFQ actions for a single-product supplier report.

The report is a decision screen, not the end of the workflow. These callbacks
let the owner prepare drafts for the recommended shortlist, only the
manufacturer, or one manually chosen supplier. Nothing is sent automatically:
every draft still requires the existing mail:yes approval.
"""
from __future__ import annotations

import asyncio
from typing import Any

from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.db import repo
from bot.db.session import session_scope
from bot.services.rfq_drafts import RfqDraft, RfqDraftError, prepare_rfq_draft

router = Router(name="single_rfq_actions")


def _role(row: Any) -> str:
    value = getattr(row, "supplier_role", None)
    if value:
        return str(value)
    flags = getattr(row, "unrega_flags", None)
    if isinstance(flags, dict):
        return str(flags.get("supplier_role") or "candidate")
    return "candidate"


def _mail_keyboard(approval_id: int) -> InlineKeyboardBuilder:
    keyboard = InlineKeyboardBuilder()
    keyboard.row(
        InlineKeyboardButton(text="✅ Отправить", callback_data=f"mail:yes:{approval_id}"),
        InlineKeyboardButton(text="❌ Не отправлять", callback_data=f"mail:no:{approval_id}"),
    )
    return keyboard


def _sort_key(row: Any) -> tuple[int, int, int, float, int]:
    role = _role(row)
    role_order = {"manufacturer": 0, "official_distributor": 1, "seller": 2, "candidate": 3}
    has_product_page = 0 if getattr(row, "site_url", None) else 1
    stock = 0 if getattr(row, "site_claims", None) is True else 1
    price = getattr(row, "site_price", None)
    price_key = float(price) if price is not None else 10**18
    supplier_id = int(getattr(row, "supplier_id", 0) or 0)
    return (role_order.get(role, 3), has_product_page, stock, price_key, supplier_id)


async def _load_rows(request_id: int) -> tuple[Any | None, list[Any]]:
    async with session_scope() as session:
        request = await repo.get_request(session, request_id)
        rows = await repo.list_candidates_for_report(session, request_id)
    rows = [row for row in rows if getattr(row, "email", None)]
    rows.sort(key=_sort_key)
    return request, rows


def _recommended(rows: list[Any], limit: int = 3) -> list[Any]:
    """Manufacturer first, then the strongest distinct commercial channels."""
    if not rows:
        return []
    selected: list[Any] = []
    maker = next((row for row in rows if _role(row) == "manufacturer"), None)
    if maker is not None:
        selected.append(maker)
    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


async def _show_drafts(message: Message, request_id: int, rows: list[Any]) -> None:
    request, _ = await _load_rows(request_id)
    if request is None:
        await message.answer("Заявка не найдена или устарела.")
        return
    if not rows:
        await message.answer("У выбранных поставщиков не найден e-mail для запроса КП.")
        return

    await message.answer(
        f"✉️ Готовлю {len(rows)} черновик(а) запроса КП. Ничего не отправляю автоматически."
    )
    results = await asyncio.gather(
        *[
            prepare_rfq_draft(
                request_id=request_id,
                product=request.product,
                supplier_id=int(getattr(row, "supplier_id", 0) or 0),
            )
            for row in rows
        ],
        return_exceptions=True,
    )
    prepared = 0
    for row, result in zip(rows, results, strict=True):
        supplier_name = str(getattr(row, "supplier_name", "поставщик") or "поставщик")
        if isinstance(result, Exception):
            reason = str(result) if isinstance(result, RfqDraftError) else "не удалось составить письмо"
            await message.answer(
                f"⚠️ <b>{texts.esc(supplier_name)}</b>: {texts.esc(reason)}",
                parse_mode="HTML",
            )
            continue
        draft: RfqDraft = result
        prepared += 1
        await message.answer(
            f"<b>{texts.esc(draft.supplier_name)}</b>\n"
            f"E-mail: {texts.esc(draft.email)}\n\n"
            f"<b>Черновик запроса КП</b>\n{texts.esc(draft.body)}",
            parse_mode="HTML",
            reply_markup=_mail_keyboard(draft.approval_id).as_markup(),
        )
    if prepared:
        await message.answer(
            "✅ Черновики готовы. Отправьте нужные кнопкой «Отправить»; остальные можно оставить без отправки."
        )


@router.callback_query(F.data.startswith("rfq:auto:"))
async def rfq_auto(callback: CallbackQuery) -> None:
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    try:
        request_id = int((callback.data or "").rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return
    _, rows = await _load_rows(request_id)
    await _show_drafts(callback.message, request_id, _recommended(rows))


@router.callback_query(F.data.startswith("rfq:maker:"))
async def rfq_maker(callback: CallbackQuery) -> None:
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    try:
        request_id = int((callback.data or "").rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return
    _, rows = await _load_rows(request_id)
    maker = next((row for row in rows if _role(row) == "manufacturer"), None)
    if maker is None:
        await callback.message.answer("Производитель с e-mail среди найденных каналов не определён. Выберите поставщика вручную.")
        return
    await _show_drafts(callback.message, request_id, [maker])


@router.callback_query(F.data.startswith("rfq:manual:"))
async def rfq_manual(callback: CallbackQuery) -> None:
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    try:
        request_id = int((callback.data or "").rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return
    _, rows = await _load_rows(request_id)
    if not rows:
        await callback.message.answer("Поставщиков с e-mail не найдено.")
        return
    keyboard = InlineKeyboardBuilder()
    for row in rows[:8]:
        supplier_id = int(getattr(row, "supplier_id", 0) or 0)
        name = str(getattr(row, "supplier_name", "поставщик") or "поставщик")
        keyboard.row(
            InlineKeyboardButton(
                text=f"✉️ {name[:42]}",
                callback_data=f"rfq:one:{request_id}:{supplier_id}",
            )
        )
    await callback.message.answer("Кому подготовить запрос КП?", reply_markup=keyboard.as_markup())


@router.callback_query(F.data.startswith("rfq:one:"))
async def rfq_one(callback: CallbackQuery) -> None:
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    parts = (callback.data or "").split(":")
    if len(parts) != 4:
        return
    try:
        request_id, supplier_id = int(parts[2]), int(parts[3])
    except ValueError:
        return
    _, rows = await _load_rows(request_id)
    row = next((item for item in rows if int(getattr(item, "supplier_id", 0) or 0) == supplier_id), None)
    if row is None:
        await callback.message.answer("Этот поставщик больше не доступен для выбора.")
        return
    await _show_drafts(callback.message, request_id, [row])
