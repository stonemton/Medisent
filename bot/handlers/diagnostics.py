"""Диагностика внешних AI-сервисов без вывода секретов.

/visiondiag проверяет не только наличие GEMINI_API_KEY, но и реальный вызов
той же vision-модели, которую использует пакетный intake. Это помогает отличить
ошибку ключа/квоты/модели от проблемы конкретной фотографии Telegram.
"""
from __future__ import annotations

import io
import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from PIL import Image, ImageDraw

from bot.config import get_settings
from bot.services.batch_intake import VISION_MODEL
from bot.services.gemini import GeminiError, Part, get_gemini_service

logger = logging.getLogger(__name__)
router = Router(name="diagnostics")

_DIAG_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "ok": {"type": "BOOLEAN"},
        "seen_text": {"type": "STRING"},
    },
    "required": ["ok", "seen_text"],
}


def _probe_png() -> bytes:
    image = Image.new("RGB", (720, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 700, 220), outline="black", width=3)
    draw.text((60, 95), "MEDISENT VISION TEST 12345", fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@router.message(Command("visiondiag"))
async def visiondiag(message: Message) -> None:
    settings = get_settings()
    lines = [
        "<b>Диагностика Gemini Vision</b>",
        f"GEMINI_API_KEY: {'задан' if settings.gemini_enabled else 'НЕ ЗАДАН'}",
        f"Vision model: <code>{VISION_MODEL}</code>",
        f"Report model: <code>{settings.llm_report_model}</code>",
    ]
    if not settings.gemini_enabled:
        lines.append("\n❌ В Railway отсутствует GEMINI_API_KEY.")
        await message.answer("\n".join(lines))
        return

    probe = _probe_png()
    lines.append(f"Probe image: {len(probe)} bytes, image/png")
    await message.answer("\n".join(lines) + "\n\n⏳ Выполняю реальный multimodal-вызов…")

    try:
        result = await get_gemini_service().generate_json(
            parts=[
                Part(text="Прочитай текст на тестовом изображении. Верни ok=true и seen_text."),
                Part(data=probe, mime_type="image/png"),
            ],
            system_instruction="Это техническая проверка vision API. Не выдумывай текст.",
            schema=_DIAG_SCHEMA,
            model=VISION_MODEL,
            operation="diagnostics.vision",
            temperature=0.0,
        )
    except GeminiError as exc:
        error = str(exc).replace(settings.gemini_api_key, "***") if settings.gemini_api_key else str(exc)
        logger.exception("Gemini vision diagnostic failed")
        await message.answer(
            "❌ <b>Gemini Vision не работает.</b>\n"
            f"Модель: <code>{VISION_MODEL}</code>\n"
            f"Ошибка: <code>{error[:1500]}</code>"
        )
        return
    except Exception as exc:  # safety net: диагностическая команда должна показать класс ошибки
        logger.exception("Unexpected vision diagnostic failure")
        await message.answer(
            "❌ <b>Неожиданная ошибка vision-пайплайна.</b>\n"
            f"Тип: <code>{type(exc).__name__}</code>\n"
            f"Ошибка: <code>{str(exc)[:1500]}</code>"
        )
        return

    await message.answer(
        "✅ <b>Gemini Vision отвечает.</b>\n"
        f"ok: <code>{result.get('ok')}</code>\n"
        f"seen_text: <code>{str(result.get('seen_text') or '')[:500]}</code>\n\n"
        "Если эта проверка проходит, проблема уже не в ключе Gemini, а в обработке конкретного фото/ответа модели."
    )
