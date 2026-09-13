"""Turn a single-product supplier report into an actionable procurement step."""
from __future__ import annotations

from aiogram.types import InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.db.session import session_scope
from bot.pipeline import build_report, run_search
from bot.services.report import render

_INSTALLED = False
_ORIGINAL_RUN_SUPPLIERS = None


def _actions_keyboard(request_id: int) -> InlineKeyboardBuilder:
    keyboard = InlineKeyboardBuilder()
    keyboard.row(
        InlineKeyboardButton(
            text="✅ Подготовить КП рекомендованным",
            callback_data=f"rfq:auto:{request_id}",
        )
    )
    keyboard.row(
        InlineKeyboardButton(
            text="🏭 Только производителю",
            callback_data=f"rfq:maker:{request_id}",
        ),
        InlineKeyboardButton(
            text="⚙️ Выбрать вручную",
            callback_data=f"rfq:manual:{request_id}",
        ),
    )
    return keyboard


def install_single_report_actions_policy() -> None:
    global _INSTALLED, _ORIGINAL_RUN_SUPPLIERS
    if _INSTALLED:
        return

    from bot.handlers import intake

    _ORIGINAL_RUN_SUPPLIERS = intake._run_suppliers

    async def run_suppliers(
        message: Message,
        request_id: int,
        product: str,
        qty: str = "",
        requirements: list[str] | None = None,
    ) -> None:
        await message.answer(
            "🔎 Ищу производителя, официальные каналы и продавцов. РУ помогает проверить товар, "
            "но отсутствие подтверждённого РУ не блокирует запрос КП."
        )
        summary = await run_search(
            request_id=request_id,
            product=product,
            requirements=requirements or [],
        )
        if summary.search_failed:
            await message.answer(texts.search_failed("; ".join(summary.errors) or "сервис не ответил"))
            return
        if summary.total_found == 0:
            await message.answer(texts.SEARCH_NOTHING)
            return

        await message.answer(texts.search_found(summary.total_found, summary.blacklisted))
        await message.answer(texts.REPORT_BUILDING)
        async with session_scope() as session:
            report = await build_report(
                session,
                request_id=request_id,
                product=product,
                qty=qty,
                requirements=requirements or [],
            )
        for chunk in render(report):
            await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)

        with_email = [row for row in report.candidates if row.email]
        if not with_email:
            await message.answer(
                "Контакты для автоматического запроса КП не найдены. Можно использовать телефоны/сайты из отчёта."
            )
            return

        await message.answer(
            "Что делаем дальше? Я могу сразу подготовить отдельные черновики запросов КП. "
            "Отправка каждого письма всё равно потребует вашего подтверждения.",
            reply_markup=_actions_keyboard(request_id).as_markup(),
        )

    intake._run_suppliers = run_suppliers
    _INSTALLED = True
