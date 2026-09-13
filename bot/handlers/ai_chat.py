"""Режим свободного общения владельца с языковой моделью."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.types import Message

from bot.config import get_settings
from bot.services.chat import get_chat_service
from bot.services.gemini import GeminiError

logger = logging.getLogger(__name__)
router = Router(name="ai_chat")

# Бот однопользовательский (OwnerOnlyMiddleware), поэтому достаточно хранить
# короткий контекст в памяти процесса. После перезапуска Railway чат начинается заново.
_enabled_chats: set[int] = set()
_history: dict[int, list[dict[str, str]]] = defaultdict(list)
_MAX_HISTORY_MESSAGES = 16


class ChatModeFilter(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return message.chat.id in _enabled_chats


def _system_prompt() -> str:
    path = Path(get_settings().prompts_dir) / "chat.md"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "Ты рабочий ассистент Medisent. Отвечай по-русски, кратко и по делу. "
            "Не утверждай, что выполнил действие во внешней системе, если ты только отвечаешь текстом."
        )


async def _answer_with_model(message: Message, text: str) -> None:
    chat_id = message.chat.id
    if not get_settings().gemini_enabled:
        await message.answer(
            "AI-чат пока не подключён: нужен RELAYMODELS_API_KEY или GEMINI_API_KEY в Railway."
        )
        return

    try:
        answer = await get_chat_service().reply(
            system_instruction=_system_prompt(),
            history=_history[chat_id],
            user_text=text,
        )
    except GeminiError as exc:
        logger.error("AI-чат: модель не ответила: %s", exc)
        await message.answer(f"⚠️ Не удалось получить ответ модели: {exc}")
        return

    _history[chat_id].append({"role": "user", "content": text})
    _history[chat_id].append({"role": "assistant", "content": answer})
    _history[chat_id] = _history[chat_id][-_MAX_HISTORY_MESSAGES:]

    # Telegram ограничивает одно сообщение примерно 4096 символами.
    while answer:
        chunk = answer[:3900]
        answer = answer[3900:]
        await message.answer(chunk, parse_mode=None)


@router.message(Command("chat"))
async def enable_chat(message: Message) -> None:
    _enabled_chats.add(message.chat.id)
    await message.answer(
        "🧠 AI-чат включён. Теперь обычные текстовые сообщения идут языковой модели.\n\n"
        "Чтобы снова отправлять текст как закупочный запрос: /procurement\n"
        "Очистить контекст разговора: /newchat"
    )


@router.message(Command("procurement"))
async def disable_chat(message: Message) -> None:
    _enabled_chats.discard(message.chat.id)
    await message.answer(
        "📦 Режим закупки включён. Обычный текст снова будет восприниматься как заявка на поиск товара."
    )


@router.message(Command("newchat"))
async def new_chat(message: Message) -> None:
    _history.pop(message.chat.id, None)
    _enabled_chats.add(message.chat.id)
    await message.answer("🧹 Контекст очищен. AI-чат включён — можно начинать новый разговор.")


@router.message(Command("ai"))
async def one_shot_ai(message: Message, command: CommandObject) -> None:
    text = (command.args or "").strip()
    if not text:
        await message.answer(
            "Напиши вопрос после команды, например:\n/ai кто держатель РУ у этого изделия?\n\n"
            "Для постоянного диалога используй /chat."
        )
        return
    await _answer_with_model(message, text)


@router.message(ChatModeFilter(), F.text & ~F.text.startswith("/"))
async def chat_message(message: Message) -> None:
    text = (message.text or "").strip()
    if text:
        await _answer_with_model(message, text)
