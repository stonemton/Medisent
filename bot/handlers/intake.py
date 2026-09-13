"""Приём запроса: текст, фото, голос, файл. Этапы 3–5 одной цепочкой."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import Message

from bot import texts
from bot.config import get_settings
from bot.db import repo
from bot.db.session import session_scope
from bot.handlers.common import download
from bot.pipeline import build_report, close_request, run_search
from bot.services.gemini import GeminiError, ProductRequest, get_gemini_service
from bot.services.mail import MailError, get_mail_service
from bot.services.report import render

logger = logging.getLogger(__name__)
router = Router(name="intake")


async def _start_pipeline(message: Message, parsed: ProductRequest, input_kind: str) -> None:
    """Общий хвост для всех видов входа: заявка → поиск → отчёт."""
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
        # Распознавание шло до заявки — его расход приписывается ей задним
        # числом, чтобы потолок на заявку и /stats видели полную цену.
        await repo.attach_orphan_api_calls(session, request_id, operation_prefix="intake.")

    await message.answer(
        texts.intake_recognised(parsed.product, parsed.qty, token), parse_mode="HTML"
    )

    settings = get_settings()
    if not settings.search_enabled:
        await message.answer(texts.SEARCH_OFF)
        return

    await message.answer(texts.SEARCH_RUNNING)
    summary = await run_search(
        request_id=request_id,
        product=parsed.product,
        requirements=parsed.requirements,
    )

    budget_note = texts.BUDGET_PER_REQUEST_EXCEEDED.format(
        limit=f"{get_settings().max_cost_per_request_usd:.2f}"
    )

    # Три разных исхода, и говорить о них надо по-разному: поиск упал (виноват
    # сервис или потолок, а не название), поиск честно ничего не дал, поиск
    # что-то дал. Раньше первые два сливались в «уточните название».
    if summary.search_failed:
        await message.answer(
            budget_note
            if summary.budget_exceeded
            else texts.search_failed("; ".join(summary.errors) or "сервис не ответил")
        )
        async with session_scope() as session:
            await close_request(session, request_id)
        return

    if summary.total_found == 0:
        await message.answer(texts.SEARCH_NOTHING)
        async with session_scope() as session:
            await close_request(session, request_id)
        return

    await message.answer(texts.search_found(summary.total_found, summary.blacklisted))
    if summary.budget_exceeded:
        await message.answer(budget_note)
    await message.answer(texts.REPORT_BUILDING)

    async with session_scope() as session:
        report = await build_report(
            session,
            request_id=request_id,
            product=parsed.product,
            qty=parsed.qty,
            requirements=parsed.requirements,
        )

    for chunk in render(report):
        await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)


@router.message(Command("forward"))
async def forward_file(message: Message, bot: Bot) -> None:
    """Переслать приложенный файл на почту без всякой обработки."""
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
    parse: Callable[[bytes], Awaitable[ProductRequest]],
) -> None:
    """Общее тело для фото, голосового и файла: подтверждение → скачать →
    распознать → конвейер. Отличаются только текст подтверждения, откуда
    брать файл и какой разборщик звать."""
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
        parsed = await parse(content)
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
        parse=lambda content: get_gemini_service().parse_photo(content),
    )


@router.message(F.voice | F.audio)
async def on_voice(message: Message, bot: Bot) -> None:
    """Голосовое на входе — это новая заявка.

    Голосовое в ответ на отчёт обрабатывает selection.py: он стоит раньше в
    цепочке роутеров и перехватывает сообщение, когда заявка ждёт выбора.
    """
    media = message.voice or message.audio
    mime = (media.mime_type if media else None) or "audio/ogg"
    await _intake_media(
        message,
        bot,
        kind="voice",
        ack=texts.INTAKE_VOICE,
        file_id=media.file_id if media else None,
        parse=lambda content: get_gemini_service().parse_voice(content, mime_type=mime),
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
        parse=lambda content: get_gemini_service().parse_document(
            content, mime_type=mime, filename=filename
        ),
    )


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    await message.answer(texts.INTAKE_ACCEPTED)

    settings = get_settings()
    if settings.gemini_enabled:
        try:
            parsed = await get_gemini_service().parse_text(text)
        except GeminiError as exc:
            logger.warning("Разбор текста моделью не удался (%s) — беру как есть", exc)
            parsed = ProductRequest(product=text, raw_input=text)
    else:
        # Без ключа Gemini текстовый запрос всё равно работает: берём строку
        # как название изделия. Это единственный вид входа, который не требует
        # модели, и терять его из-за отсутствия ключа незачем.
        parsed = ProductRequest(product=text, raw_input=text)

    # Локальный парсер удаляет количество из названия. Если после него остались
    # разделители вроде «, .», не отправляем этот мусор в поиск и в отчёт.
    parsed.product = parsed.product.strip(" \t\r\n,.;:-")

    await _start_pipeline(message, parsed, "text")
