"""Точка входа. Поднимает бота, роутеры и фоновые задачи."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot import texts
from bot.config import get_settings
from bot.db.session import dispose_engine
from bot.handlers import admin, batch_actions, diagnostics, errors, intake, selection, single_rfq_actions
from bot.logging_setup import setup_logging
from bot.middleware import OwnerOnlyMiddleware, ThrottleMiddleware
from bot.scheduler import start_background_tasks, stop_background_tasks
from bot.services import guard
from bot.services.batch_registry_policy import install_batch_registry_policy
from bot.services.batch_supplier_policy import install_batch_supplier_policy
from bot.services.direct_web_policy import install_direct_web_policy
from bot.services.firecrawl import close_firecrawl_service
from bot.services.gemini import close_gemini_service
from bot.services.http import flush_meter
from bot.services.mail import close_mail_service
from bot.services.perplexity import close_perplexity_service
from bot.services.registry import close_registry_service
from bot.services.registry_direct_policy import install_registry_direct_policy
from bot.services.registry_query_fallbacks import install_registry_query_fallbacks
from bot.services.registry_primary_policy import install_registry_primary_policy
from bot.services.single_report_actions_policy import install_single_report_actions_policy
from bot.services.unverified_registry_policy import install_unverified_registry_policy

logger = logging.getLogger(__name__)


def build_dispatcher() -> Dispatcher:
    settings = get_settings()
    dispatcher = Dispatcher()

    owner_only = OwnerOnlyMiddleware(settings.telegram_owner_id)
    throttle = ThrottleMiddleware(interval=1.0)
    for observer in (dispatcher.message, dispatcher.callback_query):
        observer.middleware(owner_only)
        observer.middleware(throttle)

    dispatcher.include_router(errors.router)
    dispatcher.include_router(diagnostics.router)
    dispatcher.include_router(admin.router)
    dispatcher.include_router(selection.router)
    dispatcher.include_router(single_rfq_actions.router)
    dispatcher.include_router(batch_actions.router)
    dispatcher.include_router(intake.router)
    return dispatcher


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    # Cheapest/most authoritative path first. Firecrawl remains a fallback.
    install_registry_direct_policy()
    install_registry_query_fallbacks()
    install_registry_primary_policy()
    # Missing RU is evidence uncertainty, not a procurement stop condition.
    install_unverified_registry_policy()
    install_batch_registry_policy()
    install_direct_web_policy()
    install_batch_supplier_policy()
    # A supplier report should immediately offer the next procurement action.
    install_single_report_actions_policy()

    warnings = settings.warn_about_missing_keys()
    logger.info("Запуск бота. Предупреждений: %s", len(warnings))

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = build_dispatcher()
    await asyncio.to_thread(guard.preload)
    tasks = start_background_tasks(bot)

    try:
        me = await bot.get_me()
        logger.info("Бот @%s готов", me.username)
        if warnings:
            await bot.send_message(settings.telegram_owner_id, texts.started_with_warnings(warnings))
        else:
            await bot.send_message(settings.telegram_owner_id, texts.START)

        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        logger.info("Останавливаюсь")
        await stop_background_tasks(tasks)
        await close_registry_service()
        await close_gemini_service()
        await close_perplexity_service()
        await close_firecrawl_service()
        await close_mail_service()
        await flush_meter()
        await dispose_engine()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger(__name__).info("Остановлен вручную")
