"""Распознавание многопозиционных закупочных заявок."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass, field

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from bot.logging_setup import log_extra
from bot.services.gemini import GeminiError, Part, get_gemini_service

logger = logging.getLogger(__name__)

VISION_MODEL = "gemini-3.8-flash"
MAX_VISION_EDGE = 3000
MIN_VISION_WIDTH = 1800


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
Твоя задача — извлечь ВСЕ товарные строки таблицы, а не выбрать одно изделие.

На входе может быть:
1) исходный скриншот целиком;
2) улучшенная увеличенная версия;
3) несколько перекрывающихся фрагментов того же изображения.
Это всё ОДИН документ. Не дублируй одинаковые строки из разных фрагментов.

Для каждой строки товара верни отдельный объект:
- product: точное читаемое наименование, включая бренд/модель/Анти-A/Анти-B/Анти-D/Анти-Kell, Super/Супер, объём, концентрацию, артикул или исполнение, если они видны;
- qty: количество из той же строки;
- unit: единица измерения;
- requirements: остальные явно видимые требования;
- confidence: уверенность 0..1.

Критично: количество часто находится в крайнем правом столбце. Сопоставляй его со строкой по горизонтали.
Игнорируй интерфейс Telegram, фон чата, время сообщения, номера колонок/строк, заголовки и итоги.
Если таблица занимает только небольшой участок скриншота, ищи именно её и используй увеличенные фрагменты.
Если часть текста читается плохо, всё равно верни все различимые позиции и понизь confidence.
"""

VISION_USER_PROMPT = """Прочитай приложенные изображения как одну закупочную таблицу.
Сначала найди саму таблицу (она может занимать небольшую часть скриншота), затем посчитай товарные строки.
Пройди каждую строку слева направо: наименование -> единица -> количество.
Используй увеличенные фрагменты для мелкого текста. Перед ответом проверь, что не потерял строки и не создал дубликаты."""

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
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            product = str(row.get("product") or "").strip()
            if not product:
                continue
            qty = str(row.get("qty") or "не указано").strip() or "не указано"
            unit = str(row.get("unit") or "шт.").strip() or "шт."
            key = (re.sub(r"\s+", " ", product.lower()), qty.lower(), unit.lower())
            if key in seen:
                continue
            seen.add(key)
            requirements = row.get("requirements")
            try:
                confidence = float(row.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            items.append(
                ProcurementItem(
                    product=product,
                    qty=qty,
                    unit=unit,
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


def _jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _prepare_image_parts(content: bytes) -> tuple[list[Part], str]:
    """Подготовить оригинал + увеличенную версию + перекрывающиеся тайлы.

    Это не OCR: мы только улучшаем читаемость мелкого текста перед multimodal-моделью.
    Особенно полезно для скриншотов Telegram, где таблица занимает 20–40% кадра.
    """
    digest = hashlib.sha256(content).hexdigest()[:10]
    try:
        with Image.open(io.BytesIO(content)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as exc:
        logger.warning("Vision preprocess: Pillow не открыл изображение %s: %s", digest, exc)
        return [Part(data=content, mime_type="image/jpeg")], f"sha={digest} decode=failed bytes={len(content)}"

    width, height = image.size
    scale = max(1.0, MIN_VISION_WIDTH / max(width, 1))
    if max(width, height) * scale > MAX_VISION_EDGE:
        scale = MAX_VISION_EDGE / max(width, height)
    if scale > 1.05:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    enhanced = ImageOps.autocontrast(image)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.65)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))

    parts: list[Part] = [
        Part(text="Исходное изображение:"),
        Part(data=content, mime_type="image/jpeg"),
        Part(text="Увеличенная и усиленная версия того же изображения:"),
        Part(data=_jpeg_bytes(enhanced), mime_type="image/jpeg"),
    ]

    ew, eh = enhanced.size
    # 2x2 тайлы с 12% перекрытием: мелкая таблица в углу становится отдельным крупным входом.
    overlap_x = int(ew * 0.12)
    overlap_y = int(eh * 0.12)
    x_mid, y_mid = ew // 2, eh // 2
    boxes = [
        (0, 0, min(ew, x_mid + overlap_x), min(eh, y_mid + overlap_y)),
        (max(0, x_mid - overlap_x), 0, ew, min(eh, y_mid + overlap_y)),
        (0, max(0, y_mid - overlap_y), min(ew, x_mid + overlap_x), eh),
        (max(0, x_mid - overlap_x), max(0, y_mid - overlap_y), ew, eh),
    ]
    for idx, box in enumerate(boxes, start=1):
        tile = enhanced.crop(box)
        tw, th = tile.size
        tile_scale = min(2.0, MAX_VISION_EDGE / max(tw, th, 1))
        if tile_scale > 1.1:
            tile = tile.resize((int(tw * tile_scale), int(th * tile_scale)), Image.Resampling.LANCZOS)
        parts.extend([
            Part(text=f"Фрагмент {idx}/4 того же изображения:"),
            Part(data=_jpeg_bytes(tile, quality=90), mime_type="image/jpeg"),
        ])

    diag = (
        f"sha={digest} bytes={len(content)} original={width}x{height} "
        f"enhanced={ew}x{eh} parts={1 + 1 + len(boxes)}"
    )
    return parts, diag


async def _vision_attempt(
    content: bytes,
    *,
    mime_type: str,
    request_id: int | None,
    strict_schema: bool,
    preprocess: bool,
) -> ProcurementBatch:
    service = get_gemini_service()
    if preprocess and mime_type.startswith("image/"):
        media_parts, diag = _prepare_image_parts(content)
        parts = [Part(text=VISION_USER_PROMPT), *media_parts]
    else:
        diag = f"bytes={len(content)} mime={mime_type} raw"
        parts = [Part(text=VISION_USER_PROMPT), Part(data=content, mime_type=mime_type)]

    logger.info(
        "Vision attempt model=%s strict=%s preprocess=%s %s",
        VISION_MODEL,
        strict_schema,
        preprocess,
        diag,
        extra=log_extra(request_id),
    )
    parsed = await service.generate_json(
        parts=parts,
        system_instruction=BATCH_INSTRUCTION,
        schema=BATCH_SCHEMA if strict_schema else None,
        model=VISION_MODEL,
        request_id=request_id,
        operation=(
            "intake.batch.preprocessed.strict" if preprocess and strict_schema
            else "intake.batch.preprocessed.retry" if preprocess
            else "intake.batch.raw"
        ),
        temperature=0.0,
    )
    batch = _from_payload(parsed)
    if not batch.recognised_items:
        raise GeminiError("модель вернула JSON без товарных позиций")
    logger.info(
        "Vision result: %s items: %s",
        len(batch.recognised_items),
        " | ".join(item.line()[:180] for item in batch.recognised_items),
        extra=log_extra(request_id),
    )
    return batch


async def parse_media_batch(
    content: bytes,
    *,
    mime_type: str,
    request_id: int | None = None,
) -> ProcurementBatch:
    """Устойчивый multimodal-разбор: raw -> enhanced/tiles -> relaxed schema."""
    digest = hashlib.sha256(content).hexdigest()[:10]
    logger.info(
        "Vision input received: sha=%s bytes=%s mime=%s model=%s",
        digest,
        len(content),
        mime_type,
        VISION_MODEL,
        extra=log_extra(request_id),
    )
    errors: list[str] = []
    attempts = (
        (True, False),   # быстрый оригинал
        (True, True),    # улучшенный + тайлы + строгий JSON
        (False, True),   # улучшенный + тайлы + свободный JSON
    )
    for strict, preprocess in attempts:
        try:
            return await _vision_attempt(
                content,
                mime_type=mime_type,
                request_id=request_id,
                strict_schema=strict,
                preprocess=preprocess,
            )
        except GeminiError as exc:
            label = f"strict={strict},preprocess={preprocess}"
            errors.append(f"{label}: {exc}")
            logger.warning(
                "Vision batch failed %s model=%s sha=%s: %s",
                label,
                VISION_MODEL,
                digest,
                exc,
                extra=log_extra(request_id),
            )
    raise GeminiError(" | ".join(errors) or "vision не вернул позиции")


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
