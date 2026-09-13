"""Команды владельца: /start, /help, /testmail, /stats, /session, /blacklist, /cancel."""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

from bot import texts
from bot.config import get_settings
from bot.db import repo
from bot.db.session import session_scope
from bot.pipeline import close_request
from bot.services.mail import MailError, get_mail_service

logger = logging.getLogger(__name__)
router = Router(name="admin")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(texts.START)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(texts.HELP)


@router.message(Command("testmail"))
async def cmd_testmail(message: Message) -> None:
    """Отправить тестовое письмо через настроенную Яндекс Почту самому себе."""
    settings = get_settings()
    if not settings.yandex_mail_enabled:
        await message.answer("Яндекс Почта не настроена в Railway.")
        return
    try:
        await get_mail_service().send(
            to=settings.yandex_email,
            token="RFQ-TEST-1",
            subject_suffix="Тест MEDISENT",
            body="Тестовое письмо MEDISENT. Если вы его получили, SMTP Яндекс Почты работает.",
        )
    except MailError as exc:
        logger.exception("Тест Яндекс Почты не удался")
        await message.answer(f"Ошибка отправки: {exc}")
        return
    await message.answer("Тестовое письмо отправлено на ваш Yandex-ящик.")


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Расходы на внешние сервисы за сегодня."""
    async with session_scope() as session:
        rows = await repo.stats_today(session)
        total = await repo.spent_today(session)

    if not rows:
        await message.answer(texts.STATS_EMPTY)
        return

    lines = [texts.STATS_HEADER]
    lines.extend(
        texts.stats_line(str(row.service), int(row.calls), float(row.cost)) for row in rows
    )
    lines.append(texts.stats_total(float(total)))
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("session"))
async def cmd_session(message: Message) -> None:
    async with session_scope() as session:
        request = await repo.get_active_request(session)
    if request is None:
        await message.answer(texts.SESSION_NONE)
        return
    await message.answer(
        texts.session_info(
            token=request.token,
            product=request.product,
            status=request.status,
            created=request.created_at.strftime("%d.%m.%Y %H:%M"),
        ),
        parse_mode="HTML",
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message) -> None:
    async with session_scope() as session:
        request = await repo.get_active_request(session)
        if request is None:
            await message.answer(texts.SESSION_NONE)
            return
        await close_request(session, int(request.id))
    await message.answer(texts.CANCELLED)


@router.message(Command("blacklist"))
async def cmd_blacklist(message: Message, command: CommandObject) -> None:
    """Показать список, добавить или снять."""
    args = (command.args or "").split(maxsplit=2)

    if not args:
        async with session_scope() as session:
            rows = await repo.list_blacklist(session)
        if not rows:
            await message.answer(texts.BLACKLIST_EMPTY + "\n\n" + texts.BLACKLIST_USAGE)
            return
        lines = [texts.BLACKLIST_HEADER]
        for row in rows:
            when = row.added_at.strftime("%d.%m.%Y")
            lines.append(
                texts.blacklist_line(int(row.supplier_id), str(row.supplier_name), row.reason, when)
            )
        lines.append("\n" + texts.BLACKLIST_USAGE)
        await message.answer("\n".join(lines), parse_mode="HTML")
        return

    action = args[0].lower()

    if action == "add":
        if len(args) < 3:
            await message.answer(texts.BLACKLIST_USAGE)
            return
        try:
            supplier_id = int(args[1])
        except ValueError:
            await message.answer(texts.BLACKLIST_USAGE)
            return
        async with session_scope() as session:
            supplier = await repo.get_supplier(session, supplier_id)
            if supplier is None:
                await message.answer(texts.supplier_not_found(supplier_id))
                return
            await repo.add_to_blacklist(session, supplier_id, args[2])
            name = supplier.name
        await message.answer(texts.blacklist_added(name))
        return

    if action == "lift":
        if len(args) < 2:
            await message.answer(texts.BLACKLIST_USAGE)
            return
        try:
            supplier_id = int(args[1])
        except ValueError:
            await message.answer(texts.BLACKLIST_USAGE)
            return
        async with session_scope() as session:
            supplier = await repo.get_supplier(session, supplier_id)
            lifted = await repo.lift_from_blacklist(session, supplier_id)
            name = supplier.name if supplier else str(supplier_id)
        await message.answer(texts.blacklist_lifted(name) if lifted else texts.BLACKLIST_NOT_LISTED)
        return

    await message.answer(texts.BLACKLIST_USAGE)
