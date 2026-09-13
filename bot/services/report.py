"""Коммерческий отчёт по кандидатам-поставщикам.

После подтверждения изделия РУ используется только для идентификации держателя/
производителя. Поставщики ранжируются по коммерческой роли:
1) держатель РУ / производитель;
2) официальный дистрибьютор / дилер с доказательством;
3) прочие продавцы.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bot import texts
from bot.db.models import RegistryState
from bot.logging_setup import log_extra
from bot.services.gemini import GeminiError, get_gemini_service

logger = logging.getLogger(__name__)
TELEGRAM_LIMIT = 4096

REPORT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "ranked": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "rank": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                    "concerns": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["id", "rank", "reason"],
            },
        },
        "summary": {"type": "STRING"},
        "missing_data": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["ranked", "summary"],
}


@dataclass(slots=True)
class CandidateView:
    candidate_id: int
    supplier_id: int
    supplier_name: str
    domain: str | None
    email: str | None
    phone: str | None
    site_claims: bool | None
    site_url: str | None
    site_price: Decimal | None
    ru_number: str | None
    ru_holder: str | None
    ru_valid: bool | None
    ru_registry: str | None
    registry_state: str
    ru_site_match: bool | None = None
    ru_match_basis: str | None = None
    supplier_role: str = "candidate"
    role_evidence_url: str | None = None
    role_evidence: str | None = None
    unrega_flags: list[str] = field(default_factory=list)
    injection_suspected: bool = False
    rank: int = 0
    reason: str = ""
    concerns: list[str] = field(default_factory=list)


FORBIDDEN_CONFLATION = (
    re.compile(r"поставщик\w*\s+(?:\S+\s+){0,2}провер\w+(?:\s+в\s+росздравнадзор\w*)?", re.IGNORECASE),
    re.compile(r"аккредитован\w*\s+(?:в\s+)?росздравнадзор\w*", re.IGNORECASE),
    re.compile(r"провер\w+\s+(?:в\s+)?росздравнадзор\w*", re.IGNORECASE),
    re.compile(r"поставщик\w*\s+(?:\S+\s+){0,2}зарегистрирован\w*", re.IGNORECASE),
)


def find_conflation(text: str) -> str | None:
    for pattern in FORBIDDEN_CONFLATION:
        match = pattern.search(text or "")
        if match:
            return match.group(0)
    return None


def scrub_conflation(text: str, *, where: str, request_id: int | None = None) -> str:
    found = find_conflation(text)
    if not found:
        return text
    logger.warning(
        "Модель попыталась слить реестр изделия и поставщика в «%s» (%s)",
        found,
        where,
        extra=log_extra(request_id),
    )
    cleaned = text
    for pattern in FORBIDDEN_CONFLATION:
        cleaned = pattern.sub(
            "[формулировка убрана: РУ относится к изделию, а не к продавцу]", cleaned
        )
    return cleaned


@dataclass(slots=True)
class Report:
    request_token: str
    product: str
    candidates: list[CandidateView]
    summary: str = ""
    missing_data: list[str] = field(default_factory=list)
    llm_failed: bool = False
    unrega_state: str = ""


def _normalise_org(value: str | None) -> set[str]:
    if not value:
        return set()
    stop = {
        "ооо", "ао", "пао", "оао", "зао", "общество", "ограниченной",
        "ответственностью", "акционерное", "компания", "россия", "рф",
    }
    return {
        x for x in re.findall(r"[a-zа-яё0-9]{4,}", value.lower())
        if x not in stop
    }


def _is_holder_or_manufacturer(view: CandidateView) -> bool:
    if view.supplier_role == "manufacturer":
        return True
    holder_terms = _normalise_org(view.ru_holder)
    if not holder_terms:
        return False
    haystack = f"{view.supplier_name} {view.domain or ''}".lower()
    return any(term in haystack for term in holder_terms)


def _verified_official_distributor(view: CandidateView) -> bool:
    return bool(
        view.supplier_role == "official_distributor"
        and view.role_evidence_url
        and view.role_evidence
    )


def _role_priority(view: CandidateView) -> int:
    if _is_holder_or_manufacturer(view):
        return 1
    if _verified_official_distributor(view):
        return 2
    return 3


def _role_line(view: CandidateView) -> str:
    priority = _role_priority(view)
    if priority == 1:
        return "1 · держатель РУ / производитель"
    if priority == 2:
        return "2 · официальный дистрибьютор / дилер"
    return "3 · прочий поставщик"


def _site_line(view: CandidateView) -> str:
    if view.site_claims is True:
        return "наличие заявлено на сайте"
    if view.site_claims is False:
        return "на сайте указано отсутствие / под заказ"
    return "наличие не подтверждено"


def build_payload(
    candidates: list[CandidateView],
    *,
    product: str,
    qty: str,
    requirements: list[str],
    criteria: list[dict[str, Any]],
    unrega_state: str = "",
) -> dict[str, Any]:
    return {
        "product": product,
        "qty": qty,
        "requirements": requirements,
        "criteria": criteria,
        "unrega_state": unrega_state,
        "ranking_rule": "Сначала группа 1, затем группа 2, затем группа 3. Внутри группы ранжируй по товару, наличию, цене, контактам и критериям.",
        "candidates": [
            {
                "id": c.candidate_id,
                "supplier": c.supplier_name,
                "domain": c.domain,
                "email": c.email,
                "phone": c.phone,
                "site_claims": c.site_claims,
                "site_url": c.site_url,
                "site_price": float(c.site_price) if c.site_price is not None else None,
                "role_priority": _role_priority(c),
                "role": _role_line(c),
                "role_evidence_url": c.role_evidence_url,
                "role_evidence": c.role_evidence,
                "unrega_flags": c.unrega_flags,
            }
            for c in candidates
        ],
    }


async def rank_candidates(
    candidates: list[CandidateView],
    *,
    product: str,
    qty: str,
    requirements: list[str],
    criteria: list[dict[str, Any]],
    request_id: int | None = None,
    unrega_state: str = "",
) -> tuple[list[CandidateView], str, list[str], bool]:
    if not candidates:
        return [], "", [], False

    payload = build_payload(
        candidates,
        product=product,
        qty=qty,
        requirements=requirements,
        criteria=criteria,
        unrega_state=unrega_state,
    )
    llm_failed = False
    parsed: dict[str, Any] = {}
    try:
        parsed = await get_gemini_service().run_prompt_file(
            "report",
            payload,
            schema=REPORT_SCHEMA,
            request_id=request_id,
            operation="report.rank",
            untrusted=True,
        )
    except GeminiError as exc:
        logger.error("Ранжирование не удалось: %s", exc, extra=log_extra(request_id))
        llm_failed = True

    by_id = {c.candidate_id: c for c in candidates}
    ranked_rows = parsed.get("ranked") or [] if isinstance(parsed, dict) else []
    seen: set[int] = set()
    for row in ranked_rows:
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("id")
        matched = by_id.get(_as_id(candidate_id)) if isinstance(candidate_id, int | str) else None
        if matched is None:
            continue
        matched.rank = int(row.get("rank") or 0)
        matched.reason = scrub_conflation(
            str(row.get("reason") or "").strip(), where="reason", request_id=request_id
        )
        concerns = row.get("concerns") or []
        matched.concerns = (
            [
                scrub_conflation(str(c).strip(), where="concerns", request_id=request_id)
                for c in concerns
                if str(c).strip()
            ]
            if isinstance(concerns, list)
            else []
        )
        seen.add(matched.candidate_id)

    tail_rank = max((c.rank for c in candidates if c.rank), default=0)
    for view in candidates:
        if view.candidate_id not in seen:
            tail_rank += 1
            view.rank = tail_rank

    # Роль является жёстким верхнеуровневым приоритетом и не может быть
    # переопределена моделью. Ранг модели работает только внутри одной группы.
    ordered = sorted(
        candidates,
        key=lambda c: (_role_priority(c), c.rank or 10_000, c.candidate_id),
    )
    for index, view in enumerate(ordered, start=1):
        view.rank = index

    missing = parsed.get("missing_data") or [] if isinstance(parsed, dict) else []
    summary = parsed.get("summary") or "" if isinstance(parsed, dict) else ""
    return (
        ordered,
        scrub_conflation(str(summary).strip(), where="summary", request_id=request_id),
        [str(m) for m in missing if str(m).strip()] if isinstance(missing, list) else [],
        llm_failed,
    )


def _as_id(value: int | str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def render(report: Report) -> list[str]:
    esc = texts.esc
    header = texts.REPORT_HEADER.format(token=esc(report.request_token), product=esc(report.product))
    blocks: list[str] = [header]
    if report.llm_failed:
        blocks.append("⚠️ Ранжирование моделью не сработало; применён локальный порядок.\n")
    if report.unrega_state == RegistryState.UNAVAILABLE:
        blocks.append(texts.UNREGA_UNAVAILABLE + "\n")

    for index, view in enumerate(report.candidates, start=1):
        lines = [f"<b>{index}. {esc(view.supplier_name)}</b>"]
        lines.append(f"   Роль: {esc(_role_line(view))}")
        if view.domain:
            lines.append(f"   {esc(view.domain)}")
        if view.site_url:
            href = html.escape(view.site_url, quote=True)
            lines.append(f'   🔗 <a href="{href}">страница с упоминанием товара</a>')
        if _verified_official_distributor(view):
            evidence_href = html.escape(view.role_evidence_url or "", quote=True)
            evidence_text = esc(view.role_evidence or "официальный статус подтверждён")
            lines.append(
                f'   ✅ Официальность: <a href="{evidence_href}">{evidence_text}</a>'
            )
        lines.append(f"   Наличие: {esc(_site_line(view))}")
        if view.site_price is not None:
            lines.append(f"   Цена на сайте: {view.site_price:,.0f} ₽".replace(",", " "))
        contacts = " · ".join(esc(c) for c in (view.email, view.phone) if c)
        lines.append(f"   Контакты: {contacts}" if contacts else "   Контакты: не найдены")
        if view.unrega_flags:
            lines.append(f"   ⚠️ Информационные письма: {len(view.unrega_flags)}")
        if view.injection_suspected:
            lines.append(f"   {texts.INJECTION_SUSPECTED}")
        if view.reason:
            lines.append(f"   <i>{esc(view.reason)}</i>")
        for concern in view.concerns:
            lines.append(f"   — {esc(concern)}")
        blocks.append("\n".join(lines) + "\n")

    if report.summary:
        blocks.append(f"<b>Вывод:</b> {esc(report.summary)}\n")
    if report.missing_data:
        blocks.append(
            "<b>Не хватило данных:</b> " + "; ".join(esc(m) for m in report.missing_data) + "\n"
        )
    blocks.append(texts.REPORT_FOOTER)
    return _pack(blocks)


def _pack(blocks: list[str]) -> list[str]:
    messages: list[str] = []
    current = ""
    for block in blocks:
        if len(current) + len(block) + 1 > TELEGRAM_LIMIT:
            if current:
                messages.append(current.rstrip())
            while len(block) > TELEGRAM_LIMIT:
                cut = block.rfind("\n", 0, TELEGRAM_LIMIT)
                cut = cut if cut > 0 else TELEGRAM_LIMIT
                messages.append(block[:cut].rstrip())
                block = block[cut:]
            current = block
        else:
            current += block + "\n"
    if current.strip():
        messages.append(current.rstrip())
    return messages
