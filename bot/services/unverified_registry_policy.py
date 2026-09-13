"""Do not let an unverified RU block procurement.

Registry verification is evidence, not a prerequisite for requesting a quote.
When no confident RU is found, Medisent may continue supplier discovery using
the exact product/brand/manufacturer clues from intake. RFQs must explicitly
ask the supplier to provide the current RU number/copy (or state that RU is not
required). Nothing in this policy invents an RU.
"""
from __future__ import annotations

from aiogram.types import Message

from bot import texts
from bot.services.batch_intake import ProcurementBatch, ProcurementItem
from bot.services.gemini import ProductRequest
from bot.services.registry import get_registry_service

_INSTALLED = False
_ORIGINAL_START_PIPELINE = None


async def _continue_single_without_ru(message: Message, parsed: ProductRequest, input_kind: str) -> bool:
    """Return True when this function handled a single unverified-RU request."""
    from bot.db import repo
    from bot.db.session import session_scope
    from bot.handlers import intake

    if not parsed.recognised or not intake.get_settings().search_enabled:
        return False

    # We deliberately perform the same registry check before deciding to bypass
    # the stop. Existing cache/direct-ELK policies still apply.
    registry = await get_registry_service().check_product(parsed.product, cache=False)
    if registry.best is not None:
        return False

    async with session_scope() as session:
        request = await repo.create_request(
            session,
            product=parsed.product,
            raw_input=parsed.raw_input,
            input_kind=input_kind,
        )
        request_id, token = int(request.id), request.token
        await repo.attach_orphan_api_calls(session, request_id, operation_prefix="intake.")

    await message.answer(texts.intake_recognised(parsed.product, parsed.qty, token), parse_mode="HTML")
    await message.answer(
        "⚠️ РУ автоматически не подтверждено. Это не блокирует закупку.\n\n"
        "Продолжаю поиск производителя, официальных каналов и поставщиков по точному "
        "наименованию/бренду. В запрос КП добавлю требование указать номер и предоставить "
        "копию действующего РУ на предлагаемое изделие; если РУ для товара не требуется — "
        "попросим поставщика прямо это указать."
    )
    requirements = list(parsed.requirements or [])
    requirements.append(
        "Обязательно указать номер действующего регистрационного удостоверения (РУ) на предлагаемое изделие "
        "и предоставить его копию/ссылку. Если государственная регистрация для товара не требуется, "
        "письменно указать это в ответе."
    )
    await intake._run_suppliers(
        message,
        request_id,
        parsed.product,
        parsed.qty or "",
        requirements,
    )
    return True


def install_unverified_registry_policy() -> None:
    """Patch single intake; batch behavior is handled by batch_registry_policy."""
    global _INSTALLED, _ORIGINAL_START_PIPELINE
    if _INSTALLED:
        return
    from bot.handlers import intake

    _ORIGINAL_START_PIPELINE = intake._start_pipeline
    original = _ORIGINAL_START_PIPELINE

    async def wrapped(message: Message, parsed: ProductRequest, input_kind: str) -> None:
        if await _continue_single_without_ru(message, parsed, input_kind):
            return
        await original(message, parsed, input_kind)

    intake._start_pipeline = wrapped
    _INSTALLED = True
