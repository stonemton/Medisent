"""Приём запроса: одиночные изделия и многопозиционные закупочные списки."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.config import get_settings
from bot.db import repo
from bot.db.session import session_scope
from bot.handlers.common import download
from bot.pipeline import build_report, close_request, run_search
from bot.services.batch_intake import (
    ProcurementBatch,
    ProcurementItem,
    deserialise_batch,
    parse_media_batch,
    parse_text_batch,
    serialise_batch,
)
from bot.services.gemini import GeminiError, ProductRequest, get_gemini_service
from bot.services.mail import MailError, get_mail_service
from bot.services.procurement_batch import run_batch_procurement
from bot.services.registry import get_registry_service
from bot.services.report import render

logger = logging.getLogger(__name__)
router = Router(name="intake")


def _product_keyboard(request_id: int) -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Да, это оно", callback_data=f"product:yes:{request_id}"),
        InlineKeyboardButton(text="❌ Нет", callback_data=f"product:no:{request_id}"),
    )
    return builder


def _batch_keyboard(request_id: int) -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Да, искать весь список", callback_data=f"batch:yes:{request_id}"),
        InlineKeyboardButton(text="❌ Нет", callback_data=f"batch:no:{request_id}"),
    )
    return builder


async def _send_chunks(message: Message, lines: list[str], *, parse_mode: str | None = None) -> None:
    chunk = ""
    for line in lines:
        candidate = f"{chunk}\n{line}" if chunk else line
        if len(candidate) > 3800:
            await message.answer(chunk, parse_mode=parse_mode)
            chunk = line
        else:
            chunk = candidate
    if chunk:
        await message.answer(chunk, parse_mode=parse_mode)


async def _run_suppliers(
    message: Message,
    request_id: int,
    product: str,
    qty: str = "",
    requirements: list[str] | None = None,
) -> None:
    await message.answer(
        "🔎 Изделие подтверждено. Ищу производителя/держателя РУ, "
        "официальных дистрибьюторов и остальных продавцов…"
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


async def _start_pipeline(message: Message, parsed: ProductRequest, input_kind: str) -> None:
    if not parsed.recognised:
        await message.answer(texts.INTAKE_NOT_RECOGNISED)
        return
    async with session_scope() as session:
        request = await repo.create_request(
            session,
            product=parsed.product,
            raw_input=parsed.raw_input,
            input_kind=input_kind,
        )
        request_id, token = int(request.id), request.token
        await repo.attach_orphan_api_calls(session, request_id, operation_prefix="intake.")
    await message.answer(
        texts.intake_recognised(parsed.product, parsed.qty, token),
        parse_mode="HTML",
    )
    if not get_settings().search_enabled:
        await message.answer(texts.SEARCH_OFF)
        return
    await message.answer("🔎 Сначала проверяю, как изделие зарегистрировано в РФ…")
    registry = await get_registry_service().check_product(
        parsed.product,
        request_id=request_id,
        cache=True,
    )
    best = registry.best
    if best is None:
        await message.answer(
            "РУ по этому наименованию уверенно не найдено. Поиск продавцов пока не запускаю. "
            "Уточните наименование/модель или пришлите РУ."
        )
        return
    status = "действует" if best.valid is True else (
        "не действует" if best.valid is False else "статус не определён"
    )
    text = (
        f"<b>Нашёл зарегистрированное изделие</b>\n\n"
        f"Наименование по РУ: <b>{texts.esc(best.product_name or parsed.product)}</b>\n"
        f"РУ: <b>{texts.esc(best.ru_number or '—')}</b> · {status}\n"
        f"Держатель РУ / производитель: <b>{texts.esc(best.holder or 'не указан')}</b>\n\n"
        "Это то изделие, которое нужно искать у поставщиков?"
    )
    await message.answer(
        text,
        parse_mode="HTML",
        reply_markup=_product_keyboard(request_id).as_markup(),
    )


async def _start_batch(message: Message, batch: ProcurementBatch, input_kind: str) -> None:
    items = batch.recognised_items
    if len(items) < 2:
        item = items[0] if items else None
        if item is None:
            await message.answer(texts.INTAKE_NOT_RECOGNISED)
            return
        await _start_pipeline(
            message,
            ProductRequest(
                product=item.product,
                qty=f"{item.qty} {item.unit}".strip(),
                requirements=item.requirements,
                confidence=item.confidence,
            ),
            input_kind,
        )
        return

    async with session_scope() as session:
        request = await repo.create_request(
            session,
            product=batch.product_text(),
            raw_input=serialise_batch(batch),
            input_kind=f"{input_kind}-batch",
        )
        request_id, token = int(request.id), request.token
        await repo.attach_orphan_api_calls(session, request_id, operation_prefix="intake.")

    intro = [
        f"<b>{texts.esc(token)} · распознано позиций: {len(items)}</b>",
        "",
    ]
    intro.extend(texts.esc(item.line(i)) for i, item in enumerate(items, start=1))
    await _send_chunks(message, intro, parse_mode="HTML")

    if not get_settings().search_enabled:
        await message.answer(texts.SEARCH_OFF)
        return

    await message.answer(
        "🔎 Проверяю каждую позицию по реестру Росздравнадзора. "
        "После этого объединю поиск по общим каналам закупки."
    )
    service = get_registry_service()
    checks = await asyncio.gather(
        *[
            service.check_product(item.product, request_id=request_id, cache=True)
            for item in items
        ]
    )

    lines = ["<b>Проверка позиций по РУ</b>", ""]
    unresolved: list[ProcurementItem] = []
    for index, (item, registry) in enumerate(zip(items, checks, strict=True), start=1):
        best = registry.best
        if best is None:
            unresolved.append(item)
            lines.append(f"⚠️ {index}. {texts.esc(item.product)} — РУ уверенно не найдено")
            continue
        status = "действует" if best.valid is True else (
            "не действует" if best.valid is False else "статус не определён"
        )
        lines.append(
            f"✅ {index}. {texts.esc(item.product)}\n"
            f"   РУ: <b>{texts.esc(best.ru_number or '—')}</b> · {status}\n"
            f"   Держатель: {texts.esc(best.holder or 'не указан')}"
        )
    await _send_chunks(message, lines, parse_mode="HTML")

    if unresolved:
        async with session_scope() as session:
            await close_request(session, request_id)
        await message.answer(
            "Поиск поставщиков не запускаю: по части списка нет уверенной идентификации РУ. "
            "Уточните эти позиции или пришлите их РУ — так бот не закупит похожее изделие вместо нужного."
        )
        return

    await message.answer(
        "Все позиции идентифицированы. Искать производителя и поставщиков сразу по всему списку?",
        reply_markup=_batch_keyboard(request_id).as_markup(),
    )


@router.callback_query(F.data.startswith("product:"))
async def product_confirm(callback: CallbackQuery) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        return
    decision, raw_id = parts[1], parts[2]
    try:
        request_id = int(raw_id)
    except ValueError:
        return
    await callback.answer()
    async with session_scope() as session:
        request = await repo.get_request(session, request_id)
    if request is None:
        if isinstance(callback.message, Message):
            await callback.message.answer("Заявка не найдена или устарела.")
        return
    if not isinstance(callback.message, Message):
        return
    if decision != "yes":
        async with session_scope() as session:
            await close_request(session, request_id)
        await callback.message.answer(
            "Хорошо. Поиск поставщиков не запускаю. Пришлите правильное наименование/модель."
        )
        return
    await _run_suppliers(callback.message, request_id, request.product)


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
        "Сначала проверю каждую позицию отдельно, затем оставлю только поставщиков, "
        "которые реально закрывают весь список."
    )
    result = await run_batch_procurement(
        master_request_id=request_id,
        items=items,
        master_product=request.product,
    )
    if result.report is not None:
        await callback.message.answer(
            f"✅ Нашёл поставщиков с подтверждённым покрытием всех {len(items)} позиций: "
            f"{result.full_coverage_suppliers}. Показываю лучшие каналы закупки."
        )
        for chunk in render(result.report):
            await callback.message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)
        return

    if result.split_plan:
        lines = [
            "<b>Одного подтверждённого поставщика на весь список не найдено.</b>",
            "Оптимальный план разделения закупки:",
            "",
        ]
        for index, plan in enumerate(result.split_plan, start=1):
            lines.append(f"<b>{index}. {texts.esc(plan.supplier_name)}</b>")
            for product in plan.covered_items:
                lines.append(f"• {texts.esc(product)}")
        if result.errors:
            lines.extend(["", "Ошибки отдельных проходов: " + texts.esc("; ".join(result.errors))])
        await _send_chunks(callback.message, lines, parse_mode="HTML")
        await callback.message.answer(
            "Общий запрос КП специально не создаю: иначе поставщику ушли бы позиции, "
            "по которым его товар не подтверждён."
        )
        return

    error = "; ".join(result.errors) if result.errors else "подходящих общих поставщиков не найдено"
    await callback.message.answer(texts.search_failed(error))


@router.message(Command("forward"))
async def forward_file(message: Message, bot: Bot) -> None:
    settings = get_settings()
    if not settings.gmail_enabled or not settings.forward_to_email:
        await message.answer(texts.FORWARD_OFF)
        return
    source = message.reply_to_message or message
    document = source.document
    if document is None:
        await message.answer(texts.FORWARD_NO_FILE)
        return
    content = await download(bot, document.file_id)
    if content is None:
        await message.answer(texts.ERROR_GENERIC)
        return
    try:
        await get_mail_service().forward_file(
            to=settings.forward_to_email,
            filename=document.file_name or "file",
            content=content,
            mime_type=document.mime_type or "application/octet-stream",
        )
    except MailError as exc:
        logger.error("Пересылка не удалась: %s", exc)
        await message.answer(texts.ERROR_GENERIC)
        return
    await message.answer(texts.FORWARD_OK.format(email=settings.forward_to_email))


async def _intake_media(
    message: Message,
    bot: Bot,
    *,
    kind: str,
    ack: str,
    file_id: str | None,
    mime_type: str,
    fallback: Callable[[bytes], Awaitable[ProductRequest]],
) -> None:
    if not get_settings().gemini_enabled:
        await message.answer(texts.INTAKE_GEMINI_OFF)
        return
    await message.answer(ack)
    if file_id is None:
        return
    content = await download(bot, file_id)
    if content is None:
        await message.answer(texts.ERROR_GENERIC)
        return
    try:
        batch = await parse_media_batch(content, mime_type=mime_type)
        if batch.is_multi:
            await _start_batch(message, batch, kind)
            return
        item = batch.recognised_items[0] if batch.recognised_items else None
        if item is not None:
            await _start_pipeline(
                message,
                ProductRequest(
                    product=item.product,
                    qty=f"{item.qty} {item.unit}".strip(),
                    requirements=item.requirements,
                    confidence=item.confidence,
                    transcript=batch.transcript,
                ),
                kind,
            )
            return
    except GeminiError as exc:
        logger.warning("Пакетное распознавание (%s) не удалось: %s; пробую старый путь", kind, exc)

    try:
        parsed = await fallback(content)
    except GeminiError as exc:
        logger.error("Распознавание (%s) не удалось: %s", kind, exc)
        await message.answer(texts.INTAKE_NOT_RECOGNISED)
        return
    await _start_pipeline(message, parsed, kind)


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot) -> None:
    photo = message.photo[-1] if message.photo else None
    await _intake_media(
        message,
        bot,
        kind="photo",
        ack=texts.INTAKE_PHOTO,
        file_id=photo.file_id if photo else None,
        mime_type="image/jpeg",
        fallback=lambda content: get_gemini_service().parse_photo(content),
    )


@router.message(F.voice | F.audio)
async def on_voice(message: Message, bot: Bot) -> None:
    media = message.voice or message.audio
    mime = (media.mime_type if media else None) or "audio/ogg"
    await _intake_media(
        message,
        bot,
        kind="voice",
        ack=texts.INTAKE_VOICE,
        file_id=media.file_id if media else None,
        mime_type=mime,
        fallback=lambda content: get_gemini_service().parse_voice(content, mime_type=mime),
    )


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    document = message.document
    mime = (document.mime_type if document else None) or "application/pdf"
    filename = (document.file_name if document else None) or ""
    await _intake_media(
        message,
        bot,
        kind="file",
        ack=texts.INTAKE_FILE,
        file_id=document.file_id if document else None,
        mime_type=mime,
        fallback=lambda content: get_gemini_service().parse_document(
            content,
            mime_type=mime,
            filename=filename,
        ),
    )


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    await message.answer(texts.INTAKE_ACCEPTED)

    local_batch = parse_text_batch(text)
    if local_batch.is_multi:
        await _start_batch(message, local_batch, "text")
        return

    settings = get_settings()
    if settings.gemini_enabled:
        try:
            parsed = await get_gemini_service().parse_text(text)
        except GeminiError as exc:
            logger.warning("Разбор текста моделью не удался (%s) — беру как есть", exc)
            parsed = ProductRequest(product=text, raw_input=text)
    else:
        parsed = ProductRequest(product=text, raw_input=text)
    parsed.product = parsed.product.strip(" \t\r\n,.;:-")
    await _start_pipeline(message, parsed, "text")
