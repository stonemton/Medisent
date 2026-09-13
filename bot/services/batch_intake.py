"""Распознавание многопозиционных закупочных заявок."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from bot.logging_setup import log_extra
from bot.services.gemini import GeminiError, Part, get_gemini_service

logger = logging.getLogger(__name__)

# Для vision не используем общий alias из окружения. В сентябре 2026 стабильная
# production-модель Gemini 3.8 Flash нативно принимает Image/PDF и поддерживает
# structured outputs. Это устраняет зависимость от устаревших alias вида
# gemini-flash-latest, которые могут вести на модель без нужной multimodal-схемы.
VISION_MODEL = "gemini-3.8-flash"


@dataclass(slots=True)
class ProcurementItem:
    product: str
    qty: str = "не указано"
    unit: str = "шт."
    requirements: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def recognised(self) -> bool:
        return bool(self.product.strip()) and self.product.strip().lower() not in {
            "не определено", "unknown", "null", "-"
        }

    def line(self, index: int | None = None) -> str:
        prefix = f"{index}. " if index is not None else ""
        amount = self.qty.strip() or "не указано"
        unit = self.unit.strip()
        tail = f" — {amount}{(' ' + unit) if unit and amount != 'не указано' else ''}"
        return f"{prefix}{self.product.strip()}{tail}"


@dataclass(slots=True)
class ProcurementBatch:
    items: list[ProcurementItem] = field(default_factory=list)
    confidence: float = 1.0
    transcript: str | None = None

    @property
    def recognised_items(self) -> list[ProcurementItem]:
        return [item for item in self.items if item.recognised]

    @property
    def is_multi(self) -> bool:
        return len(self.recognised_items) > 1

    def product_text(self) -> str:
        return "\n".join(item.line(i) for i, item in enumerate(self.recognised_items, start=1))


BATCH_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "product": {"type": "STRING"},
                    "qty": {"type": "STRING"},
                    "unit": {"type": "STRING"},
                    "requirements": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["product", "qty", "unit", "requirements", "confidence"],
            },
        },
        "confidence": {"type": "NUMBER"},
        "transcript": {"type": "STRING"},
    },
    "required": ["items", "confidence"],
}

BATCH_INSTRUCTION = """Ты профессиональный закупщик медицинских изделий и читаешь фотографию/скриншот таблицы.
Твоя задача — НЕ угадывать одно изделие, а извлечь ВСЕ товарные строки таблицы.

Для каждой строки товара верни отдельный объект:
- product: точное читаемое наименование, включая Анти-A/Анти-B/Анти-D/Анти-Kell, Super/Супер, объем флакона, концентрацию, артикул или исполнение, если они видны;
- qty: количество из той же строки;
- unit: единица измерения;
- requirements: остальные явно видимые требования;
- confidence: уверенность 0..1.

Критично: количество часто находится в крайнем правом столбце. Сопоставляй его со строкой по горизонтали.
Не считай заголовки, номер строки, номер колонки и итог отдельным товаром. Не объединяй разные реагенты в одну позицию.
Если часть текста читается плохо, всё равно верни все различимые позиции и понизь confidence. Не отвечай общей фразой «не удалось определить изделие».
"""

VISION_USER_PROMPT = """Прочитай изображение как таблицу закупки. Сначала определи границы строк, затем для каждой строки прочитай левый столбец с наименованием и правый столбец с количеством. Верни все строки товара. Проверь количество строк дважды перед ответом."""

_QTY_TAIL_RE = re.compile(
    r"(?P<qty>\d[\d\s]*(?:[.,]\d+)?)\s*(?P<unit>фл\.?|флак\.?|флакон(?:а|ов)?|шт\.?|штук(?:а|и)?|уп\.?|упаков(?:ка|ки|ок)|компл\.?|комплект(?:а|ов)?)\s*$",
    re.IGNORECASE,
)
_NUMBERING_RE = re.compile(r"^\s*(?:\d+[.)]|[-•])\s*")


def parse_text_batch(text: str) -> ProcurementBatch:
    raw_lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    items: list[ProcurementItem] = []
    for raw in raw_lines:
        line = _NUMBERING_RE.sub("", raw).strip(" \t|;,")
        match = _QTY_TAIL_RE.search(line)
        if not match:
            continue
        product = line[: match.start()].strip(" \t-–—|;,:.")
        if not product:
            continue
        qty = " ".join(match.group("qty").split()).replace(",", ".")
        unit = match.group("unit").strip().rstrip(".")
        items.append(ProcurementItem(product=product, qty=qty, unit=unit))
    return ProcurementBatch(items=items, confidence=1.0)


def _from_payload(payload: dict[str, object]) -> ProcurementBatch:
    items: list[ProcurementItem] = []
    rows = payload.get("items")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            product = str(row.get("product") or "").strip()
            if not product:
                continue
            requirements = row.get("requirements")
            try:
                confidence = float(row.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            items.append(
                ProcurementItem(
                    product=product,
                    qty=str(row.get("qty") or "не указано").strip() or "не указано",
                    unit=str(row.get("unit") or "шт.").strip() or "шт.",
                    requirements=[str(x).strip() for x in requirements if str(x).strip()]
                    if isinstance(requirements, list)
                    else [],
                    confidence=confidence,
                )
            )
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return ProcurementBatch(
        items=items,
        confidence=confidence,
        transcript=str(payload.get("transcript") or "").strip() or None,
    )


async def _vision_attempt(
    content: bytes,
    *,
    mime_type: str,
    request_id: int | None,
    strict_schema: bool,
) -> ProcurementBatch:
    service = get_gemini_service()
    parsed = await service.generate_json(
        parts=[Part(text=VISION_USER_PROMPT), Part(data=content, mime_type=mime_type)],
        system_instruction=BATCH_INSTRUCTION,
        schema=BATCH_SCHEMA if strict_schema else None,
        model=VISION_MODEL,
        request_id=request_id,
        operation="intake.batch.strict" if strict_schema else "intake.batch.retry",
        temperature=0.0,
    )
    batch = _from_payload(parsed)
    if not batch.recognised_items:
        raise GeminiError("модель вернула JSON без товарных позиций")
    return batch


async def parse_media_batch(
    content: bytes,
    *,
    mime_type: str,
    request_id: int | None = None,
) -> ProcurementBatch:
    """Два независимых vision-прохода Gemini 3.8 Flash.

    Первый — со строгой схемой. Второй — без responseSchema, если первый упал.
    Если оба не сработали, наружу уходит подробная безопасная причина для логов.
    """
    errors: list[str] = []
    for strict in (True, False):
        try:
            batch = await _vision_attempt(
                content,
                mime_type=mime_type,
                request_id=request_id,
                strict_schema=strict,
            )
            logger.info(
                "Vision batch: распознано %s позиций, model=%s strict=%s",
                len(batch.recognised_items),
                VISION_MODEL,
                strict,
                extra=log_extra(request_id),
            )
            return batch
        except GeminiError as exc:
            errors.append(str(exc))
            logger.warning(
                "Vision batch failed model=%s strict=%s: %s",
                VISION_MODEL,
                strict,
                exc,
                extra=log_extra(request_id),
            )
    raise GeminiError("; ".join(errors) or "vision не вернул позиции")


def serialise_batch(batch: ProcurementBatch) -> str:
    payload = {
        "v": 1,
        "items": [
            {
                "product": item.product,
                "qty": item.qty,
                "unit": item.unit,
                "requirements": item.requirements,
                "confidence": item.confidence,
            }
            for item in batch.recognised_items
        ],
        "confidence": batch.confidence,
        "transcript": batch.transcript,
    }
    return "BATCH_V1:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def deserialise_batch(raw: str | None) -> ProcurementBatch | None:
    if not raw or not raw.startswith("BATCH_V1:"):
        return None
    try:
        payload = json.loads(raw[len("BATCH_V1:") :])
    except (json.JSONDecodeError, TypeError):
        return None
    return _from_payload(payload) if isinstance(payload, dict) else None
