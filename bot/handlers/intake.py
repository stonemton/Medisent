"""Приём запроса: сначала идентификация изделия по РУ, затем поиск поставщиков после подтверждения."""
from __future__ import annotations
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
from bot.services.gemini import GeminiError, ProductRequest, get_gemini_service
from bot.services.mail import MailError, get_mail_service
from bot.services.registry import get_registry_service
from bot.services.report import render
logger=logging.getLogger(__name__)
router=Router(name="intake")

def _product_keyboard(request_id:int)->InlineKeyboardBuilder:
    b=InlineKeyboardBuilder(); b.row(InlineKeyboardButton(text="✅ Да, это оно",callback_data=f"product:yes:{request_id}"),InlineKeyboardButton(text="❌ Нет",callback_data=f"product:no:{request_id}")); return b

async def _run_suppliers(message:Message,request_id:int,product:str,qty:str="",requirements:list[str]|None=None)->None:
    await message.answer("🔎 Изделие подтверждено. Ищу производителя/держателя РУ, официальных дистрибьюторов и остальных продавцов…")
    summary=await run_search(request_id=request_id,product=product,requirements=requirements or [])
    if summary.search_failed:
        await message.answer(texts.search_failed("; ".join(summary.errors) or "сервис не ответил")); return
    if summary.total_found==0:
        await message.answer(texts.SEARCH_NOTHING); return
    await message.answer(texts.search_found(summary.total_found,summary.blacklisted)); await message.answer(texts.REPORT_BUILDING)
    async with session_scope() as session: report=await build_report(session,request_id=request_id,product=product,qty=qty,requirements=requirements or [])
    for chunk in render(report): await message.answer(chunk,parse_mode="HTML",disable_web_page_preview=True)

async def _start_pipeline(message:Message,parsed:ProductRequest,input_kind:str)->None:
    if not parsed.recognised: await message.answer(texts.INTAKE_NOT_RECOGNISED); return
    async with session_scope() as session:
        request=await repo.create_request(session,product=parsed.product,raw_input=parsed.raw_input,input_kind=input_kind); request_id,token=int(request.id),request.token
        await repo.attach_orphan_api_calls(session,request_id,operation_prefix="intake.")
    await message.answer(texts.intake_recognised(parsed.product,parsed.qty,token),parse_mode="HTML")
    if not get_settings().search_enabled: await message.answer(texts.SEARCH_OFF); return
    await message.answer("🔎 Сначала проверяю, как изделие зарегистрировано в РФ…")
    registry=await get_registry_service().check_product(parsed.product,request_id=request_id,cache=True); best=registry.best
    if best is None:
        await message.answer("РУ по этому наименованию уверенно не найдено. Поиск продавцов пока не запускаю. Уточните наименование/модель или пришлите РУ."); return
    status="действует" if best.valid is True else ("не действует" if best.valid is False else "статус не определён")
    text=(f"<b>Нашёл зарегистрированное изделие</b>\n\n"
          f"Наименование по РУ: <b>{texts.esc(best.product_name or parsed.product)}</b>\n"
          f"РУ: <b>{texts.esc(best.ru_number or '—')}</b> · {status}\n"
          f"Держатель РУ / производитель: <b>{texts.esc(best.holder or 'не указан')}</b>\n\n"
          "Это то изделие, которое нужно искать у поставщиков?")
    await message.answer(text,parse_mode="HTML",reply_markup=_product_keyboard(request_id).as_markup())

@router.callback_query(F.data.startswith("product:"))
async def product_confirm(callback:CallbackQuery)->None:
    parts=(callback.data or "").split(":");
    if len(parts)!=3: return
    decision,raw_id=parts[1],parts[2]
    try: request_id=int(raw_id)
    except ValueError: return
    await callback.answer()
    async with session_scope() as session: request=await repo.get_request(session,request_id)
    if request is None:
        if isinstance(callback.message,Message): await callback.message.answer("Заявка не найдена или устарела.")
        return
    if not isinstance(callback.message,Message): return
    if decision!="yes":
        async with session_scope() as session: await close_request(session,request_id)
        await callback.message.answer("Хорошо. Поиск поставщиков не запускаю. Пришлите правильное наименование/модель."); return
    await _run_suppliers(callback.message,request_id,request.product)

@router.message(Command("forward"))
async def forward_file(message:Message,bot:Bot)->None:
    settings=get_settings()
    if not settings.gmail_enabled or not settings.forward_to_email: await message.answer(texts.FORWARD_OFF); return
    source=message.reply_to_message or message; document=source.document
    if document is None: await message.answer(texts.FORWARD_NO_FILE); return
    content=await download(bot,document.file_id)
    if content is None: await message.answer(texts.ERROR_GENERIC); return
    try: await get_mail_service().forward_file(to=settings.forward_to_email,filename=document.file_name or "file",content=content,mime_type=document.mime_type or "application/octet-stream")
    except MailError as exc: logger.error("Пересылка не удалась: %s",exc); await message.answer(texts.ERROR_GENERIC); return
    await message.answer(texts.FORWARD_OK.format(email=settings.forward_to_email))

async def _intake_media(message:Message,bot:Bot,*,kind:str,ack:str,file_id:str|None,parse:Callable[[bytes],Awaitable[ProductRequest]])->None:
    if not get_settings().gemini_enabled: await message.answer(texts.INTAKE_GEMINI_OFF); return
    await message.answer(ack)
    if file_id is None:return
    content=await download(bot,file_id)
    if content is None: await message.answer(texts.ERROR_GENERIC); return
    try: parsed=await parse(content)
    except GeminiError as exc: logger.error("Распознавание (%s) не удалось: %s",kind,exc); await message.answer(texts.INTAKE_NOT_RECOGNISED); return
    await _start_pipeline(message,parsed,kind)

@router.message(F.photo)
async def on_photo(message:Message,bot:Bot)->None:
    photo=message.photo[-1] if message.photo else None; await _intake_media(message,bot,kind="photo",ack=texts.INTAKE_PHOTO,file_id=photo.file_id if photo else None,parse=lambda c:get_gemini_service().parse_photo(c))
@router.message(F.voice|F.audio)
async def on_voice(message:Message,bot:Bot)->None:
    media=message.voice or message.audio; mime=(media.mime_type if media else None) or "audio/ogg"; await _intake_media(message,bot,kind="voice",ack=texts.INTAKE_VOICE,file_id=media.file_id if media else None,parse=lambda c:get_gemini_service().parse_voice(c,mime_type=mime))
@router.message(F.document)
async def on_document(message:Message,bot:Bot)->None:
    d=message.document; mime=(d.mime_type if d else None) or "application/pdf"; filename=(d.file_name if d else None) or ""; await _intake_media(message,bot,kind="file",ack=texts.INTAKE_FILE,file_id=d.file_id if d else None,parse=lambda c:get_gemini_service().parse_document(c,mime_type=mime,filename=filename))
@router.message(F.text&~F.text.startswith("/"))
async def on_text(message:Message)->None:
    text=(message.text or "").strip()
    if not text:return
    await message.answer(texts.INTAKE_ACCEPTED); settings=get_settings()
    if settings.gemini_enabled:
        try: parsed=await get_gemini_service().parse_text(text)
        except GeminiError as exc: logger.warning("Разбор текста моделью не удался (%s) — беру как есть",exc); parsed=ProductRequest(product=text,raw_input=text)
    else: parsed=ProductRequest(product=text,raw_input=text)
    parsed.product=parsed.product.strip(" \t\r\n,.;:-"); await _start_pipeline(message,parsed,"text")
