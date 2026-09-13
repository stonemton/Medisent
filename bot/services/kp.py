"""Коммерческое предложение: извлечение закупочных цен и сборка PDF.

Порядок:
1. модель вытаскивает из письма поставщика позиции и закупочные цены;
2. владелец видит закупочную цену и расчётную продажную цену;
3. только после подтверждения собирается PDF с продажной ценой.

Базовый коэффициент продажи MEDISENT — 1.8. Закупочная цена при этом не
теряется: она остаётся в Extraction и в quote_requests для аналитики.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import sys
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from bot.config import get_settings
from bot.logging_setup import log_extra
from bot.services import guard
from bot.services.gemini import GeminiError, Part, get_gemini_service

logger = logging.getLogger(__name__)

ORDER_OF_MAGNITUDE_FACTOR = Decimal(20)
VALID_UNTIL_DAYS = 14
BASE_SALES_COEFFICIENT = Decimal("1.8")
MONEY_QUANT = Decimal("0.01")

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "note": {"type": "STRING"},
                    "qty": {"type": "NUMBER"},
                    "unit": {"type": "STRING"},
                    "price": {"type": "NUMBER"},
                    "vat_included": {"type": "BOOLEAN", "nullable": True},
                    "min_qty": {"type": "NUMBER", "nullable": True},
                    "prepayment_pct": {"type": "NUMBER", "nullable": True},
                    "caveat": {"type": "STRING"},
                },
                "required": ["name", "qty", "price"],
            },
        },
        "currency": {"type": "STRING"},
        "lead_time": {"type": "STRING"},
        "payment_terms": {"type": "STRING"},
        "valid_until": {"type": "STRING"},
        "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["items", "currency"],
}


def sales_price(price: Decimal, coefficient: Decimal = BASE_SALES_COEFFICIENT) -> Decimal:
    """Продажная цена с денежным округлением до копеек."""
    return (price * coefficient).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


@dataclass(slots=True)
class ExtractedItem:
    name: str
    qty: Decimal
    price: Decimal
    unit: str = "шт."
    note: str = ""
    vat_included: bool | None = None
    min_qty: Decimal | None = None
    prepayment_pct: Decimal | None = None
    caveat: str = ""

    @property
    def total(self) -> Decimal:
        """Закупочная сумма позиции."""
        return (self.qty * self.price).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)

    @property
    def sale_price(self) -> Decimal:
        return sales_price(self.price)

    @property
    def sale_total(self) -> Decimal:
        return (self.qty * self.sale_price).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)

    @property
    def caveats(self) -> list[str]:
        out: list[str] = []
        if self.vat_included is False:
            out.append("без НДС")
        elif self.vat_included is True:
            out.append("с НДС")
        if self.min_qty:
            out.append(f"от {self.min_qty:g} шт.")
        if self.prepayment_pct:
            out.append(f"предоплата {self.prepayment_pct:g}%")
        if self.caveat:
            out.append(self.caveat)
        return out


@dataclass(slots=True)
class Extraction:
    items: list[ExtractedItem] = field(default_factory=list)
    currency: str = "RUB"
    lead_time: str = ""
    payment_terms: str = ""
    valid_until: str = ""
    notes: list[str] = field(default_factory=list)
    failed: bool = False
    error: str = ""

    @property
    def total(self) -> Decimal:
        """Закупочная сумма всего ответа поставщика."""
        return sum((item.total for item in self.items), Decimal(0))

    @property
    def sale_total(self) -> Decimal:
        """Расчётная сумма продажи по базовому коэффициенту."""
        return sum((item.sale_total for item in self.items), Decimal(0))

    def suspicious_items(self) -> list[ExtractedItem]:
        prices = sorted(item.price for item in self.items if item.price > 0)
        if len(prices) < 3:
            return []
        median = prices[len(prices) // 2]
        if median <= 0:
            return []
        return [
            item
            for item in self.items
            if item.price > median * ORDER_OF_MAGNITUDE_FACTOR
            or item.price * ORDER_OF_MAGNITUDE_FACTOR < median
        ]


def _decimal(value: Any, default: Decimal | None = None) -> Decimal | None:
    if value is None or value == "":
        return default
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return default


async def extract_from_letter(
    letter_text: str,
    *,
    attachments_text: str = "",
    request_id: int | None = None,
) -> Extraction:
    combined = letter_text
    if attachments_text:
        combined += f"\n\n--- из вложений ---\n{attachments_text}"
    safe = await guard.sanitise_for_model(combined, source="письмо поставщика")
    try:
        gemini = get_gemini_service()
        parsed = await gemini.generate_json(
            parts=[Part(text=safe)],
            system_instruction=gemini.load_instruction("extract"),
            schema=EXTRACTION_SCHEMA,
            model=get_settings().llm_email_model,
            request_id=request_id,
            operation="kp.extract",
        )
    except GeminiError as exc:
        logger.error("Разбор письма не удался: %s", exc, extra=log_extra(request_id))
        return Extraction(failed=True, error=str(exc))

    items: list[ExtractedItem] = []
    for row in parsed.get("items") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        price = _decimal(row.get("price"))
        qty = _decimal(row.get("qty"), Decimal(1)) or Decimal(1)
        if not name or price is None or price <= 0:
            continue
        items.append(
            ExtractedItem(
                name=name,
                qty=qty,
                price=price,
                unit=str(row.get("unit") or "шт."),
                note=str(row.get("note") or ""),
                vat_included=row.get("vat_included"),
                min_qty=_decimal(row.get("min_qty")),
                prepayment_pct=_decimal(row.get("prepayment_pct")),
                caveat=str(row.get("caveat") or ""),
            )
        )
    notes = parsed.get("notes") or []
    return Extraction(
        items=items,
        currency=str(parsed.get("currency") or "RUB").upper(),
        lead_time=str(parsed.get("lead_time") or ""),
        payment_terms=str(parsed.get("payment_terms") or ""),
        valid_until=str(parsed.get("valid_until") or ""),
        notes=[str(n) for n in notes if str(n).strip()] if isinstance(notes, list) else [],
    )


def read_pdf_attachment(content: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber не установлен — вложение PDF пропущено")
        return ""
    chunks: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(content)
        handle.flush()
        try:
            with pdfplumber.open(handle.name) as pdf:
                for number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.append(f"[страница {number}]\n{text}")
                    for table in page.extract_tables() or []:
                        rows = [
                            " | ".join(cell or "" for cell in row)
                            for row in table
                            if any(cell for cell in row)
                        ]
                        if rows:
                            chunks.append(f"[таблица, страница {number}]\n" + "\n".join(rows))
        except Exception as exc:
            logger.warning("PDF не разобрался: %s", exc)
            return ""
    return "\n\n".join(chunks)


def build_kp_json(
    extraction: Extraction,
    *,
    number: str,
    client_name: str,
    intro: str = "",
) -> tuple[dict[str, Any], str | None]:
    """Собрать КП с продажными ценами; закупочные остаются только внутри системы."""
    today = dt.date.today()
    warning: str | None = None
    valid_until = extraction.valid_until.strip()
    if not valid_until:
        default = today + dt.timedelta(days=VALID_UNTIL_DAYS)
        valid_until = default.strftime("%d.%m.%Y")
        warning = valid_until

    terms: list[str] = []
    if extraction.lead_time:
        terms.append(f"Срок поставки — {extraction.lead_time}.")
    if extraction.payment_terms:
        terms.append(f"Порядок оплаты — {extraction.payment_terms}.")
    terms.extend(extraction.notes)

    payload: dict[str, Any] = {
        "number": number,
        "date": today.strftime("%d.%m.%Y"),
        "valid_until": valid_until,
        "title": "Коммерческое предложение",
        "client": {"name": client_name},
        "currency": extraction.currency,
        "items": [
            {
                "name": item.name,
                "note": "; ".join(filter(None, [item.note, *item.caveats])),
                "qty": float(item.qty),
                "unit": item.unit,
                "price": float(item.sale_price),
            }
            for item in extraction.items
        ],
        "terms": terms,
    }
    if intro:
        payload["intro"] = intro
    return payload, warning


def check_assets() -> list[str]:
    settings = get_settings()
    missing: list[str] = []
    for name in ("logo.png", "stamp.png", "signature.png"):
        if not (Path(settings.kp_builder_dir) / "assets" / name).exists():
            missing.append(name)
    return missing


async def build_pdf(
    data: dict[str, Any],
    *,
    out_path: Path,
    final: bool = False,
    request_id: int | None = None,
) -> tuple[bool, str]:
    settings = get_settings()
    kp_dir = Path(settings.kp_builder_dir)
    script = kp_dir / "scripts" / "build_kp.py"
    if not script.exists():
        return False, f"нет скрипта сборки {script}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        data_path = Path(handle.name)
    args = [sys.executable, str(script), "--data", str(data_path), "--out", str(out_path)]
    if not final:
        args.append("--no-stamp")
    logger.info(
        "Сборка КП: %s",
        "финальная с печатью" if final else "черновик без печати",
        extra=log_extra(request_id),
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(kp_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=180)
        output = stdout.decode("utf-8", errors="replace")
        success = process.returncode == 0 and await asyncio.to_thread(out_path.exists)
        if not success:
            logger.error("Сборка КП не удалась: %s", output[-1000:], extra=log_extra(request_id))
        return success, output
    except TimeoutError:
        return False, "сборка КП не уложилась в 180 секунд"
    finally:
        await asyncio.to_thread(data_path.unlink, True)


def extraction_to_payload(extraction: Extraction) -> dict[str, Any]:
    """Закупочные цены сохраняются в одобрении; продажные всегда пересчитываются по политике."""
    return {
        "currency": extraction.currency,
        "lead_time": extraction.lead_time,
        "payment_terms": extraction.payment_terms,
        "valid_until": extraction.valid_until,
        "notes": list(extraction.notes),
        "items": [
            {
                "name": item.name,
                "qty": str(item.qty),
                "price": str(item.price),
                "unit": item.unit,
                "note": item.note,
                "vat_included": item.vat_included,
                "min_qty": str(item.min_qty) if item.min_qty is not None else None,
                "prepayment_pct": str(item.prepayment_pct) if item.prepayment_pct is not None else None,
                "caveat": item.caveat,
            }
            for item in extraction.items
        ],
    }


def extraction_from_payload(payload: dict[str, Any]) -> Extraction:
    items = [
        ExtractedItem(
            name=str(row.get("name", "")),
            qty=Decimal(str(row.get("qty", "1"))),
            price=Decimal(str(row.get("price", "0"))),
            unit=str(row.get("unit", "шт.")),
            note=str(row.get("note", "")),
            vat_included=row.get("vat_included"),
            min_qty=Decimal(str(row["min_qty"])) if row.get("min_qty") else None,
            prepayment_pct=Decimal(str(row["prepayment_pct"])) if row.get("prepayment_pct") else None,
            caveat=str(row.get("caveat", "")),
        )
        for row in payload.get("items", [])
    ]
    return Extraction(
        items=items,
        currency=str(payload.get("currency", "RUB")),
        lead_time=str(payload.get("lead_time", "")),
        payment_terms=str(payload.get("payment_terms", "")),
        valid_until=str(payload.get("valid_until", "")),
        notes=[str(n) for n in payload.get("notes", [])],
    )
