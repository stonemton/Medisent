"""Prepare supplier RFQ drafts without sending them.

This service is intentionally side-effect-limited: it may create an approval
record, but it never sends email. Sending still requires the existing explicit
`mail:yes:<approval_id>` confirmation handled by selection.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.config import get_settings
from bot.db import repo
from bot.db.models import ApprovalKind, QuoteStatus
from bot.db.session import session_scope
from bot.services.gemini import GeminiError, get_gemini_service

EMAIL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "subject_suffix": {"type": "STRING"},
        "body": {"type": "STRING"},
        "questions": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["subject_suffix", "body"],
}


@dataclass(slots=True)
class RfqDraft:
    supplier_id: int
    supplier_name: str
    email: str
    approval_id: int
    body: str
    subject_suffix: str


class RfqDraftError(RuntimeError):
    pass


async def prepare_rfq_draft(*, request_id: int, product: str, supplier_id: int) -> RfqDraft:
    settings = get_settings()
    if not settings.gmail_enabled:
        raise RfqDraftError("почта не настроена")

    async with session_scope() as session:
        selectable = await repo.is_selectable_candidate(session, request_id, supplier_id)
        supplier = await repo.get_supplier(session, supplier_id) if selectable else None
        request = await repo.get_request(session, request_id)
        already = await repo.find_quote(session, request_id, supplier_id)

    if not selectable:
        raise RfqDraftError("поставщик отсутствует среди кандидатов заявки")
    if supplier is None or request is None:
        raise RfqDraftError("не удалось загрузить заявку или поставщика")
    if already is not None and already.status in QuoteStatus.DELIVERED:
        raise RfqDraftError("запрос этому поставщику уже отправлен")
    if not supplier.email:
        raise RfqDraftError("у поставщика не найден e-mail")

    try:
        drafted = await get_gemini_service().run_prompt_file(
            "email",
            {
                "token": request.token,
                "product": product,
                "qty": "см. список позиций" if "\n" in product else "не указано",
                "requirements": [],
                "supplier": {"name": supplier.name, "email": supplier.email},
                "site_claims": None,
                "site_url": None,
                "sender_name": settings.gmail_sender,
            },
            schema=EMAIL_SCHEMA,
            model=settings.llm_email_model,
            request_id=request_id,
            operation="email.draft",
        )
    except GeminiError as exc:
        raise RfqDraftError(f"модель не составила письмо: {exc}") from exc

    body = str(drafted.get("body") or "").strip()
    suffix = str(drafted.get("subject_suffix") or "Запрос КП — медицинские изделия").strip()
    if not body:
        raise RfqDraftError("модель вернула пустой текст письма")

    async with session_scope() as session:
        approval = await repo.create_approval(
            session,
            kind=ApprovalKind.EMAIL,
            request_id=request_id,
            supplier_id=supplier_id,
            payload={
                "to": supplier.email,
                "supplier_name": supplier.name,
                "token": request.token,
                "subject_suffix": suffix,
                "body": body,
            },
        )
        approval_id = int(approval.id)

    return RfqDraft(
        supplier_id=supplier_id,
        supplier_name=supplier.name,
        email=supplier.email,
        approval_id=approval_id,
        body=body,
        subject_suffix=suffix,
    )
