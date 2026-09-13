"""Свободный диалог владельца с LLM внутри Telegram.

Сервис не выполняет закупочные действия сам: он отвечает текстом. Рабочие
сценарии (поиск поставщиков, письма, КП) остаются в существующих хендлерах.
"""

from __future__ import annotations

from typing import Any

from bot.config import get_settings
from bot.services.gemini import GOOGLE_API_BASE, GeminiError, usage_from_payload
from bot.services.http import ApiClient


class ChatService:
    """Текстовый чат через уже настроенный RelayModels или прямой Gemini."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._relay = settings.relaymodels_enabled
        if self._relay:
            self._client = ApiClient(
                "relaymodels",
                base_url=settings.relaymodels_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.relaymodels_api_key}"},
                timeout_read=120.0,
            )
        else:
            self._client = ApiClient(
                "gemini",
                base_url=GOOGLE_API_BASE,
                headers={"Content-Type": "application/json"},
                timeout_read=120.0,
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def reply(
        self,
        *,
        system_instruction: str,
        history: list[dict[str, str]],
        user_text: str,
    ) -> str:
        settings = self._settings
        if not settings.gemini_enabled:
            raise GeminiError("RELAYMODELS_API_KEY/GEMINI_API_KEY не задан")

        if self._relay:
            model = settings.llm_agent_model
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_instruction},
                *history,
                {"role": "user", "content": user_text},
            ]
            result = await self._client.post(
                "/chat/completions",
                operation=f"chat:{model}",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.35,
                },
                price=lambda payload: usage_from_payload(payload, model),
            )
            if not result.ok:
                if result.budget_exceeded:
                    raise GeminiError("исчерпан дневной/заявочный лимит расходов")
                raise GeminiError(result.error or "чат-модель не ответила")
            text = _openai_text(result.json or {})
        else:
            model = settings.llm_report_model
            contents = [
                {
                    "role": "model" if item.get("role") == "assistant" else "user",
                    "parts": [{"text": item.get("content", "")}],
                }
                for item in history
                if item.get("content")
            ]
            contents.append({"role": "user", "parts": [{"text": user_text}]})
            result = await self._client.post(
                f"/models/{model}:generateContent",
                operation=f"chat:{model}",
                params={"key": settings.gemini_api_key},
                json={
                    "contents": contents,
                    "systemInstruction": {"parts": [{"text": system_instruction}]},
                    "generationConfig": {"temperature": 0.35},
                },
                price=lambda payload: usage_from_payload(payload, model),
            )
            if not result.ok:
                if result.budget_exceeded:
                    raise GeminiError("исчерпан дневной/заявочный лимит расходов")
                raise GeminiError(result.error or "чат-модель не ответила")
            text = _google_text(result.json or {})

        if not text:
            raise GeminiError("модель вернула пустой ответ")
        return text


def _openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        ).strip()
    return ""


def _google_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, dict)
    ).strip()


_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service


async def close_chat_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None
