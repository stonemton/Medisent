"""Do not let an unverified RU block procurement.

Registry verification is evidence, not a prerequisite for requesting a quote.
For single-item intake this replaces the old hard stop while preserving the
normal confirmation flow whenever a verified registry candidate exists.
"""
from __future__ import annotations

from aiogram.types import Message

from bot import texts
from bot.services.gemini import ProductRequest
from bot.services.registry import get_registry_service

_INSTALLED = False
_ORIGINAL_START_PIPELINE = None


def install_unverified_registry_policy() -> None:
    global _INSTALLED, _ORIGINAL_START_PIPELINE
    if _INSTALLED:
        return

    from bot.db import repo
    from bot.db.session import session_scope
    from bot.handlers import intake

    _ORIGINAL_START_PIPELINE = intake._start_pipeline

    async def wrapped(message: Message, parsed: ProductRequest, input_kind: str) -> None:
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
        if not intake.get_settings().search_enabled:
            await message.answer(texts.SEARCH_OFF)
            return

        await message.answer("🔎 Сначала проверяю, как изделие зарегистрировано в РФ…")
        registry = await get_registry_service().check_product(
            parsed.product,
            request_id=request_id,
            cache=False,
        )
        best = registry.best

        if best is None:
            await message.answer(
                "⚠️ РУ автоматически не подтверждено. Это не блокирует закупку.\n\n"
                "Продолжаю поиск производителя, официальных каналов и поставщиков по точному "
                "наименованию/бренду. В запрос КП обязательно попрошу указать номер и предоставить "
                "копию/официальную ссылку на действующее РУ. Если для товара РУ не требуется — "
                "поставщик должен письменно это указать."
            )
            requirements = list(parsed.requirements or [])
            requirements.append(
                "Указать номер действующего регистрационного удостоверения (РУ) на предлагаемое изделие "
                "и предоставить копию/официальную ссылку. Если государственная регистрация для товара "
                "не требуется, письменно указать это в ответе."
            )
            await intake._run_suppliers(
                message,
                request_id,
                parsed.product,
                parsed.qty or "",
                requirements,
            )
            return

        agent_meta = intake._agent_decision(best)
        needs_review = bool(agent_meta.get("needs_review"))
        try:
            confidence = float(agent_meta.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if needs_review and confidence < 0.70:
            reason = str(agent_meta.get("reasoning") or "между найденными РУ остаётся существенная неоднозначность")
            await message.answer(
                "⚠️ Нашёл несколько похожих регистрационных вариантов, но не хочу выбирать наугад.\n\n"
                f"{texts.esc(reason)}\n\n"
                "Здесь уточнение действительно влияет на идентичность товара. Пришлите точное "
                "наименование/модель/артикул либо номер РУ — после этого продолжу.",
                parse_mode="HTML",
            )
            return

        status = "действует" if best.valid is True else (
            "не действует" if best.valid is False else "статус не определён"
        )
        parts = [
            "<b>Нашёл зарегистрированное изделие</b>", "",
            f"Наименование по РУ: <b>{texts.esc(best.product_name or parsed.product)}</b>",
            f"РУ: <b>{texts.esc(best.ru_number or '—')}</b> · {status}",
            f"Держатель РУ / производитель: <b>{texts.esc(best.holder or 'не указан')}</b>",
        ]
        alternatives = intake._alternate_ru_lines(best)
        if alternatives:
            parts.extend(["", *alternatives])
        if agent_meta.get("reasoning"):
            parts.extend(["", "🧠 " + texts.esc(str(agent_meta.get("reasoning")))])
        parts.extend([
            "",
            "Основным считаю РУ выше. Изделия под другим РУ показываю только как альтернативные совпадения — не смешиваю производителей автоматически.",
            "",
            "Это то изделие, которое нужно искать у поставщиков?",
        ])
        await message.answer(
            "\n".join(parts),
            parse_mode="HTML",
            reply_markup=intake._product_keyboard(request_id).as_markup(),
        )

    intake._start_pipeline = wrapped
    _INSTALLED = True
