"""LLM-сервис для мультимодального ввода и строгого JSON.

Если задан RELAYMODELS_API_KEY, используется RelayModels через OpenAI-compatible
/chat/completions. Иначе сохраняется прямой Google Gemini API через GEMINI_API_KEY.
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

GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    """Вызов не удался или ответ не разобрался."""


@dataclass(slots=True)
class Part:
    """Кусок мультимодального запроса: текст или файл."""

    text: str | None = None
    mime_type: str | None = None
    data: bytes | None = None

    def to_google(self) -> dict[str, Any]:
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

    def to_openai(self) -> dict[str, Any]:
        if self.text is not None:
            return {"type": "text", "text": self.text}
        if self.data is None or self.mime_type is None:
            raise ValueError("часть без текста должна иметь mime_type и данные")
        encoded = base64.b64encode(self.data).decode("ascii")
        if self.mime_type.startswith("image/") or self.mime_type == "application/pdf":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.mime_type};base64,{encoded}"},
            }
        raise GeminiError(
            f"RelayModels: тип {self.mime_type} нельзя передать через chat/completions"
        )


@dataclass(slots=True)
class ProductRequest:
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
    ranked: list[tuple[int, int, dict[str, Any], list[str]]] = []
    for index, candidate in enumerate(payload.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        score = 0
        reasons: list[str] = []
        concerns: list[str] = []

        site_claims = candidate.get("site_claims")
        if site_claims is True:
            score += 30
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
            site_match = registry.get("site_match")
            match_basis = str(registry.get("match_basis") or "")
            if state == "found" and ru_valid is True:
                score += 20
                reasons.append("найдено действующее РУ на изделие")
            elif state == "found":
                score += 5
                reasons.append("РУ найдено, статус требует внимания")
            elif state == "unavailable":
                concerns.append("реестр РУ был недоступен")
            elif state == "not_found":
                score -= 15
                concerns.append("РУ не найдено")

            if site_match is True:
                score += 60
                reasons.append("товар поставщика подтверждённо связан с найденным РУ")
                if "бренд/модель" in match_basis.lower():
                    score += 40
                    reasons.append("совпадает отличительный бренд/модель")
                elif "номер ру" in match_basis.lower() or "ру " in match_basis.lower():
                    score += 20
                    reasons.append("на странице поставщика указан тот же номер РУ")
            elif site_match is False:
                score -= 60
                concerns.append("на странице поставщика есть противоречие с найденным РУ")
            elif state == "found":
                concerns.append("связь товара поставщика с найденным РУ не подтверждена")

        if candidate.get("email"):
            score += 10
            reasons.append("есть e-mail")
        if candidate.get("phone"):
            score += 5
            reasons.append("есть телефон")
        if candidate.get("site_price") is not None:
            score += 5
            reasons.append("есть опубликованная цена")

        flags = candidate.get("unrega_flags") or []
        if isinstance(flags, list) and flags:
            score -= min(15, 5 * len(flags))
            concerns.append("есть информационные письма, требуется ручная проверка")

        reason = "; ".join(reasons) if reasons else "данных для преимущества немного"
        ranked.append((score, index, {"id": candidate.get("id"), "reason": reason}, concerns))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    rows = [
        {"id": base["id"], "rank": rank, "reason": base["reason"], "concerns": concerns}
        for rank, (_, _, base, concerns) in enumerate(ranked, start=1)
    ]
    missing: list[str] = []
    if any(
        (c.get("registry") or {}).get("state") == "unavailable"
        for c in payload.get("candidates") or []
        if isinstance(c, dict)
    ):
        missing.append("проверка РУ недоступна")
    return {
        "ranked": rows,
        "summary": (
            "Рейтинг рассчитан локально без LLM: приоритет у подтверждённой связи "
            "товара поставщика с РУ и совпадения бренда/модели; затем учитываются "
            "наличие, контакты и цена."
        ),
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
    lines = [greeting, "", f"Просим предоставить коммерческое предложение на: {product}."]
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


def _openai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Преобразовать старую Gemini-схему с TYPE в обычный JSON Schema."""
    type_map = {
        "OBJECT": "object",
        "ARRAY": "array",
        "STRING": "string",
        "NUMBER": "number",
        "INTEGER": "integer",
        "BOOLEAN": "boolean",
    }

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if key == "type" and isinstance(item, str):
                    out[key] = type_map.get(item.upper(), item.lower())
                else:
                    out[key] = convert(item)
            if out.get("type") == "object":
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    result = convert(schema)
    if not isinstance(result, dict):
        raise GeminiError("некорректная JSON schema")
    return result


def _assistant_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks).strip()
    return ""


class GeminiService:
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
            raise GeminiError("RELAYMODELS_API_KEY/GEMINI_API_KEY не задан")

        model_name = model or settings.llm_report_model
        if self._relay:
            content = [part.to_openai() for part in parts]
            body: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content},
                ],
                "temperature": temperature,
            }
            if schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "medisent_response",
                        "schema": _openai_schema(schema),
                    },
                }
            else:
                body["response_format"] = {"type": "json_object"}

            result = await self._client.post(
                "/chat/completions",
                operation=f"{operation}:{model_name}",
                request_id=request_id,
                json=body,
                price=lambda payload: usage_from_payload(payload, model_name),
            )
            if not result.ok:
                if result.budget_exceeded:
                    raise GeminiError("исчерпан потолок расходов на заявку")
                raise GeminiError(result.error or "вызов RelayModels не удался")
            payload = result.json or {}
            raw_text = _assistant_text(payload)
        else:
            body = {
                "contents": [{"role": "user", "parts": [p.to_google() for p in parts]}],
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
            logger.error("LLM вернул не JSON: %s", raw_text[:300], extra=log_extra(request_id))
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

    async def _relay_transcribe(
        self,
        audio: bytes,
        mime_type: str,
        *,
        request_id: int | None = None,
    ) -> str:
        result = await self._client.post(
            "/audio/transcriptions",
            operation=f"transcribe:{self._settings.relaymodels_transcribe_model}",
            request_id=request_id,
            files={"file": ("voice.ogg", audio, mime_type)},
            data={"model": self._settings.relaymodels_transcribe_model},
        )
        if not result.ok:
            raise GeminiError(result.error or "RelayModels не распознал аудио")
        payload = result.json or {}
        return str(payload.get("text") or payload.get("transcript") or "").strip()

    async def parse_voice(
        self, audio: bytes, mime_type: str = "audio/ogg", *, request_id: int | None = None
    ) -> ProductRequest:
        if self._relay:
            transcript = await self._relay_transcribe(audio, mime_type, request_id=request_id)
            result = _parse_text_locally(transcript)
            result.transcript = transcript
            result.raw_input = transcript
            return result
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
        if self._relay:
            return await self._relay_transcribe(audio, mime_type, request_id=request_id)
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
    if not isinstance(payload, dict):
        return Usage()
    usage = payload.get("usage") or {}
    if isinstance(usage, dict) and usage:
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        cached = usage.get("cached_tokens")
        return Usage(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=int(cached) if cached is not None else None,
            cost_usd=pricing.llm_cost(model_name, tokens_in, tokens_out),
        )
    meta = payload.get("usageMetadata", {})
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
