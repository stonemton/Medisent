"""Gemini: мультимодальный вход и строгий JSON на выходе.

Изображения и аудио модель берёт нативно, отдельный Whisper не нужен.

В бесплатном режиме простые текстовые заявки, ранжирование кандидатов и
черновики писем обрабатываются локально и не расходуют суточную квоту Gemini.
Gemini остаётся для фото, голоса, файлов и других действительно сложных задач.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bot.config import get_settings
from bot.logging_setup import log_extra
from bot.services import pricing
from bot.services.http import ApiClient, Usage
from bot.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    """Вызов не удался или ответ не разобрался."""


@dataclass(slots=True)
class Part:
    """Кусок мультимодального запроса: текст или файл."""

    text: str | None = None
    mime_type: str | None = None
    data: bytes | None = None

    def to_api(self) -> dict[str, Any]:
        if self.text is not None:
            return {"text": self.text}
        if self.data is None or self.mime_type is None:
            raise ValueError("часть без текста должна иметь mime_type и данные")
        return {
            "inline_data": {
                "mime_type": self.mime_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            }
        }


@dataclass(slots=True)
class ProductRequest:
    """Результат разбора входа. Одна форма для текста, фото, голоса и файла."""

    product: str
    qty: str = "не указано"
    requirements: list[str] = field(default_factory=list)
    raw_input: str = ""
    confidence: float = 1.0
    transcript: str | None = None

    @property
    def recognised(self) -> bool:
        return bool(self.product.strip()) and self.product.strip().lower() not in (
            "не определено",
            "unknown",
            "null",
            "-",
        )


PRODUCT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "product": {"type": "STRING"},
        "qty": {"type": "STRING"},
        "requirements": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "NUMBER"},
        "transcript": {"type": "STRING"},
    },
    "required": ["product", "qty", "requirements", "confidence"],
}


_QTY_RE = re.compile(
    r"(?P<qty>\d[\d\s]*(?:[.,]\d+)?)\s*(?P<unit>шт\.?|штук(?:а|и)?|уп(?:ак(?:овк[аи])?)?\.?|компл(?:ект(?:а|ов)?)?\.?|короб(?:ка|ки|ок)?)\b",
    re.IGNORECASE,
)


def _parse_text_locally(text: str) -> ProductRequest:
    """Разобрать обычную текстовую заявку без LLM и без расхода Gemini."""
    raw = " ".join((text or "").strip().split())
    if not raw:
        return ProductRequest(product="", raw_input=text)

    match = _QTY_RE.search(raw)
    qty = match.group(0).strip() if match else "не указано"
    product = raw
    if match:
        product = (raw[: match.start()] + " " + raw[match.end() :]).strip(" ,;:-")

    product = re.sub(
        r"^(?:нуж(?:ен|на|но|ны)|требуется|ищем|закупаем|купить|запрос(?:ить)?(?:\s+кп)?(?:\s+на)?)\s+",
        "",
        product,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")
    return ProductRequest(product=product or raw, qty=qty, raw_input=text, confidence=1.0)


def _local_rank(payload: dict[str, Any]) -> dict[str, Any]:
    """Детерминированное ранжирование без LLM.

    Оцениваются только факты, уже собранные конвейером: наличие на сайте,
    состояние РУ, контакты и опубликованная цена. Никаких выдуманных признаков.
    """
    ranked: list[tuple[int, int, dict[str, Any], list[str]]] = []
    for index, candidate in enumerate(payload.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        score = 0
        reasons: list[str] = []
        concerns: list[str] = []

        site_claims = candidate.get("site_claims")
        if site_claims is True:
            score += 50
            reasons.append("на сайте найдено подтверждение товара")
        elif site_claims is False:
            score -= 20
            concerns.append("на сайте товар не подтверждён")
        else:
            concerns.append("сайт не дал подтверждения товара")

        registry = candidate.get("registry") or {}
        if isinstance(registry, dict):
            state = str(registry.get("state") or "")
            ru_valid = registry.get("ru_valid")
            if state == "found" and ru_valid is True:
                score += 30
                reasons.append("найдено действующее РУ на изделие")
            elif state == "found":
                score += 10
                reasons.append("РУ найдено, статус требует внимания")
            elif state == "unavailable":
                concerns.append("реестр РУ был недоступен")
            elif state == "not_found":
                score -= 10
                concerns.append("РУ не найдено")

        if candidate.get("email"):
            score += 15
            reasons.append("есть e-mail")
        if candidate.get("phone"):
            score += 5
            reasons.append("есть телефон")
        if candidate.get("site_price") is not None:
            score += 10
            reasons.append("есть опубликованная цена")

        flags = candidate.get("unrega_flags") or []
        if isinstance(flags, list) and flags:
            score -= min(15, 5 * len(flags))
            concerns.append("есть информационные письма, требуется ручная проверка")

        reason = "; ".join(reasons) if reasons else "данных для преимущества немного"
        ranked.append((score, index, {"id": candidate.get("id"), "reason": reason}, concerns))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    rows = []
    for rank, (_, _, base, concerns) in enumerate(ranked, start=1):
        rows.append({"id": base["id"], "rank": rank, "reason": base["reason"], "concerns": concerns})

    missing: list[str] = []
    if any((c.get("registry") or {}).get("state") == "unavailable" for c in payload.get("candidates") or [] if isinstance(c, dict)):
        missing.append("проверка РУ недоступна")
    return {
        "ranked": rows,
        "summary": "Рейтинг рассчитан локально без Gemini по подтверждённым данным сайта, РУ, контактам и цене.",
        "missing_data": missing,
    }


def _local_email(payload: dict[str, Any]) -> dict[str, Any]:
    product = str(payload.get("product") or "изделие").strip()
    qty = str(payload.get("qty") or "не указано").strip()
    supplier = payload.get("supplier") or {}
    supplier_name = str(supplier.get("name") or "").strip() if isinstance(supplier, dict) else ""
    requirements = payload.get("requirements") or []
    token = str(payload.get("token") or "").strip()

    greeting = f"Добрый день, коллеги из {supplier_name}!" if supplier_name else "Добрый день!"
    lines = [
        greeting,
        "",
        f"Просим предоставить коммерческое предложение на: {product}.",
    ]
    if qty and qty != "не указано":
        lines.append(f"Количество: {qty}.")
    if isinstance(requirements, list) and requirements:
        lines.append("Требования: " + "; ".join(str(x) for x in requirements if str(x).strip()) + ".")
    lines.extend(
        [
            "",
            "Просим указать цену, срок поставки, наличие, производителя и номер регистрационного удостоверения (при наличии).",
            "Также просим приложить карточку предприятия или реквизиты для оформления заказа.",
        ]
    )
    if token:
        lines.extend(["", f"Внутренний номер запроса: {token}."])
    lines.extend(["", "Заранее благодарим за ответ."])
    return {
        "subject_suffix": f"Запрос КП — {product}",
        "body": "\n".join(lines),
        "questions": [],
    }


class GeminiService:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = ApiClient(
            "gemini",
            base_url=API_BASE,
            headers={"Content-Type": "application/json"},
            timeout_read=120.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_json(
        self,
        *,
        parts: list[Part],
        system_instruction: str,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        request_id: int | None = None,
        operation: str = "generate",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        settings = self._settings
        if not settings.gemini_enabled:
            raise GeminiError("GEMINI_API_KEY не задан")

        model_name = model or settings.llm_report_model
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [p.to_api() for p in parts]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        if schema is not None:
            body["generationConfig"]["responseSchema"] = schema

        result = await self._client.post(
            f"/models/{model_name}:generateContent",
            operation=f"{operation}:{model_name}",
            request_id=request_id,
            params={"key": settings.gemini_api_key},
            json=body,
            price=lambda payload: usage_from_payload(payload, model_name),
        )
        if not result.ok:
            if result.budget_exceeded:
                raise GeminiError("исчерпан потолок расходов на заявку")
            raise GeminiError(result.error or "вызов Gemini не удался")

        payload = result.json or {}
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = payload.get("promptFeedback", {}).get("blockReason")
            raise GeminiError(f"модель не вернула ответ (blockReason={reason})")

        text_parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(part.get("text", "") for part in text_parts).strip()
        if not raw_text:
            raise GeminiError("модель вернула пустой текст")

        parsed = parse_llm_json(raw_text)
        if parsed is None:
            logger.error("Gemini вернул не JSON: %s", raw_text[:300], extra=log_extra(request_id))
            raise GeminiError("ответ не разобрался как JSON")
        if not isinstance(parsed, dict):
            raise GeminiError("ожидался объект JSON")
        return parsed

    def load_instruction(self, name: str) -> str:
        path = Path(self._settings.prompts_dir) / f"{name}.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise GeminiError(f"нет файла инструкции {path}") from exc

    async def parse_text(self, text: str, *, request_id: int | None = None) -> ProductRequest:
        # Обычный текст не тратит Gemini. Это основной бесплатный путь.
        return _parse_text_locally(text)

    async def parse_photo(
        self, image: bytes, mime_type: str = "image/jpeg", *, request_id: int | None = None
    ) -> ProductRequest:
        parsed = await self.generate_json(
            parts=[
                Part(text="На фото — медицинское изделие или его упаковка. Определи, что это."),
                Part(mime_type=mime_type, data=image),
            ],
            system_instruction=self.load_instruction("product"),
            schema=PRODUCT_SCHEMA,
            request_id=request_id,
            operation="intake.photo",
        )
        return _to_product_request(parsed, raw_input="[фото]")

    async def parse_voice(
        self, audio: bytes, mime_type: str = "audio/ogg", *, request_id: int | None = None
    ) -> ProductRequest:
        parsed = await self.generate_json(
            parts=[
                Part(text="Это голосовое сообщение закупщика. Расшифруй и определи изделие."),
                Part(mime_type=mime_type, data=audio),
            ],
            system_instruction=self.load_instruction("product"),
            schema=PRODUCT_SCHEMA,
            request_id=request_id,
            operation="intake.voice",
        )
        return _to_product_request(parsed, raw_input="[голосовое]")

    async def parse_document(
        self,
        content: bytes,
        mime_type: str,
        *,
        filename: str = "",
        request_id: int | None = None,
    ) -> ProductRequest:
        parsed = await self.generate_json(
            parts=[
                Part(text=f"Файл «{filename}» с описанием изделия. Определи, что нужно купить."),
                Part(mime_type=mime_type, data=content),
            ],
            system_instruction=self.load_instruction("product"),
            schema=PRODUCT_SCHEMA,
            request_id=request_id,
            operation="intake.file",
        )
        return _to_product_request(parsed, raw_input=f"[файл {filename}]")

    async def transcribe(
        self, audio: bytes, mime_type: str = "audio/ogg", *, request_id: int | None = None
    ) -> str:
        parsed = await self.generate_json(
            parts=[
                Part(text="Расшифруй это голосовое сообщение дословно, по-русски."),
                Part(mime_type=mime_type, data=audio),
            ],
            system_instruction=self.load_instruction("transcribe"),
            schema={
                "type": "OBJECT",
                "properties": {"transcript": {"type": "STRING"}},
                "required": ["transcript"],
            },
            request_id=request_id,
            operation="transcribe",
        )
        return str(parsed.get("transcript", "")).strip()

    async def run_prompt_file(
        self,
        prompt_name: str,
        payload: dict[str, Any],
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        request_id: int | None = None,
        operation: str | None = None,
        untrusted: bool = True,
    ) -> dict[str, Any]:
        # Два частых шага выполняются локально: это сохраняет бесплатную
        # суточную квоту для фото, голоса и документов.
        if prompt_name == "report":
            return _local_rank(payload)
        if prompt_name == "email":
            return _local_email(payload)

        instruction = self.load_instruction(prompt_name)
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        if untrusted:
            from bot.services import guard

            body = guard.wrap_untrusted(body, source="поиск и сайты поставщиков")

        return await self.generate_json(
            parts=[Part(text=body)],
            system_instruction=instruction,
            schema=schema,
            model=model,
            request_id=request_id,
            operation=operation or f"prompt.{prompt_name}",
        )


def usage_from_payload(payload: Any, model_name: str) -> Usage:
    meta = (payload or {}).get("usageMetadata", {}) if isinstance(payload, dict) else {}
    tokens_in = int(meta.get("promptTokenCount", 0) or 0)
    tokens_out = int(meta.get("candidatesTokenCount", 0) or 0)
    cached = meta.get("cachedContentTokenCount")
    return Usage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=int(cached) if cached is not None else None,
        cost_usd=pricing.llm_cost(model_name, tokens_in, tokens_out),
    )


def _to_product_request(parsed: dict[str, Any], *, raw_input: str) -> ProductRequest:
    requirements = parsed.get("requirements") or []
    if not isinstance(requirements, list):
        requirements = [str(requirements)]
    transcript = parsed.get("transcript")
    return ProductRequest(
        product=str(parsed.get("product", "")).strip(),
        qty=str(parsed.get("qty", "не указано")).strip() or "не указано",
        requirements=[str(item) for item in requirements if str(item).strip()],
        raw_input=str(transcript).strip() if transcript else raw_input,
        confidence=float(parsed.get("confidence", 1.0) or 0.0),
        transcript=str(transcript).strip() if transcript else None,
    )


_service: GeminiService | None = None


def get_gemini_service() -> GeminiService:
    global _service
    if _service is None:
        _service = GeminiService()
    return _service


async def close_gemini_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None
