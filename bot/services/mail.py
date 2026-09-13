"""Яндекс Почта через IMAP.

MEDISENT читает входящие и сохраняет подготовленные письма в стандартную
папку Drafts. SMTP намеренно не используется: пользователь проверяет черновик
в Яндекс.Почте и отправляет его вручную.
"""

from __future__ import annotations

import asyncio
import base64
import imaplib
import logging
import re
import ssl
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import formataddr, make_msgid, parseaddr
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from bot.config import get_settings
from bot.db import repo
from bot.db.models import QuoteRequest
from bot.logging_setup import log_extra

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\b(RFQ-\d{4}-\d+)\b", re.IGNORECASE)
MESSAGE_ID_RE = re.compile(r"<[^<>\s]+>")


class MailError(RuntimeError):
    pass


@dataclass(slots=True)
class ReplyHeaders:
    gmail_id: str = ""
    thread_id: str | None = None
    subject: str = ""
    from_email: str = ""
    from_name: str = ""
    message_ids: list[str] = field(default_factory=list)
    body: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MatchResult:
    quote: QuoteRequest | None
    method: str


def extract_token(subject: str) -> str | None:
    match = TOKEN_RE.search(subject or "")
    return match.group(1).upper() if match else None


def parse_message_ids(*header_values: str | None) -> list[str]:
    found: list[str] = []
    for value in header_values:
        if not value:
            continue
        for match in MESSAGE_ID_RE.finditer(value):
            candidate = match.group(0)
            if candidate not in found:
                found.append(candidate)
    return found


async def match_quote(session: AsyncSession, headers: ReplyHeaders) -> MatchResult:
    if headers.message_ids:
        quote = await repo.find_quote_by_message_id(session, headers.message_ids)
        if quote is not None:
            return MatchResult(quote, "message_id")
    if headers.thread_id:
        quote = await repo.find_quote_by_thread(session, headers.thread_id)
        if quote is not None:
            return MatchResult(quote, "thread")
    token = extract_token(headers.subject)
    if token:
        quote = await repo.find_quote_by_token(session, token)
        if quote is not None:
            return MatchResult(quote, "token")
    if headers.from_email:
        quote = await repo.find_quote_by_sender(session, headers.from_email)
        if quote is not None:
            return MatchResult(quote, "sender")
    return MatchResult(None, "none")


def new_message_id(sender: str) -> str:
    return make_msgid(domain=sender.split("@")[-1] if "@" in sender else None)


def build_message(
    *,
    sender: str,
    sender_name: str,
    to: str,
    token: str,
    subject_suffix: str,
    body: str,
    message_id: str | None = None,
) -> tuple[str, str]:
    message = EmailMessage()
    message["Subject"] = f"[{token}] {subject_suffix}"
    message["From"] = formataddr((sender_name, sender)) if sender_name else sender
    message["To"] = to
    message_id = message_id or new_message_id(sender)
    message["Message-ID"] = message_id
    message.set_content(body)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    return raw, message_id


def _decode_raw_message(raw_b64: str) -> bytes:
    padding = "=" * (-len(raw_b64) % 4)
    return base64.urlsafe_b64decode(raw_b64 + padding)


def _plain_body(message: Message) -> str:
    if message.is_multipart():
        body = message.get_body(preferencelist=("plain", "html"))
        if body is None:
            return ""
        try:
            return body.get_content()
        except Exception:
            payload = body.get_payload(decode=True) or b""
            return payload.decode(body.get_content_charset() or "utf-8", errors="replace")
    try:
        return message.get_content()
    except Exception:
        payload = message.get_payload(decode=True) or b""
        return payload.decode(message.get_content_charset() or "utf-8", errors="replace")


def _attachment_parts(message: Message) -> list[Message]:
    result: list[Message] = []
    if not message.is_multipart():
        return result
    for part in message.walk():
        if part.is_multipart():
            continue
        if part.get_filename() or part.get_content_disposition() == "attachment":
            result.append(part)
    return result


def _parse_rfc822(raw: bytes, uid: str) -> ReplyHeaders:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    name, address = parseaddr(str(message.get("From", "")))
    attachments: list[dict[str, Any]] = []
    for index, part in enumerate(_attachment_parts(message)):
        payload = part.get_payload(decode=True) or b""
        attachments.append(
            {
                "filename": part.get_filename() or f"attachment-{index + 1}",
                "attachment_id": str(index),
                "mime_type": part.get_content_type(),
                "size": len(payload),
            }
        )
    return ReplyHeaders(
        gmail_id=uid,
        thread_id=None,
        subject=str(message.get("Subject", "")),
        from_email=address.lower(),
        from_name=name,
        message_ids=parse_message_ids(
            str(message.get("In-Reply-To", "")),
            str(message.get("References", "")),
        ),
        body=_plain_body(message),
        attachments=attachments,
    )


class MailService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def aclose(self) -> None:
        return None

    def _require_enabled(self) -> None:
        if not self._settings.yandex_mail_enabled:
            raise MailError("Яндекс Почта не настроена")

    def _imap_login(self) -> imaplib.IMAP4_SSL:
        self._require_enabled()
        try:
            client = imaplib.IMAP4_SSL(
                self._settings.imap_host,
                self._settings.imap_port,
                ssl_context=ssl.create_default_context(),
                timeout=30,
            )
            client.login(self._settings.yandex_email, self._settings.yandex_app_password)
            return client
        except MailError:
            raise
        except Exception as exc:
            raise MailError(f"не удалось подключиться к Яндекс IMAP: {exc}") from exc

    def _imap_connect(self) -> imaplib.IMAP4_SSL:
        client = self._imap_login()
        try:
            status, _ = client.select("INBOX", readonly=True)
            if status != "OK":
                client.logout()
                raise MailError("не удалось открыть INBOX Яндекс Почты")
            return client
        except Exception:
            try:
                client.logout()
            except Exception:
                pass
            raise

    def _find_drafts_mailbox(self, client: imaplib.IMAP4_SSL) -> str:
        """Найти серверную папку с атрибутом \\Drafts; для Yandex fallback = Drafts."""
        try:
            status, rows = client.list()
            if status == "OK" and rows:
                for row in rows:
                    if not row or b"\\Drafts" not in row:
                        continue
                    text = row.decode("utf-8", errors="replace")
                    # Последний элемент LIST — имя папки, обычно quoted-string.
                    match = re.search(r'\s"([^\"]+)"$', text)
                    if match:
                        return match.group(1)
                    return text.rsplit(" ", 1)[-1].strip('"')
        except Exception:
            logger.debug("Не удалось определить папку Drafts по SPECIAL-USE", exc_info=True)
        return "Drafts"

    def _append_draft_bytes(self, raw: bytes) -> str:
        client = self._imap_login()
        try:
            mailbox = self._find_drafts_mailbox(client)
            status, data = client.append(mailbox, "(\\Draft)", None, raw)
            if status != "OK":
                details = " ".join(
                    item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
                    for item in (data or [])
                )
                raise MailError(f"не удалось сохранить черновик в {mailbox}: {details or status}")
            return mailbox
        except MailError:
            raise
        except Exception as exc:
            raise MailError(f"не удалось сохранить черновик в Яндекс.Почте: {exc}") from exc
        finally:
            try:
                client.logout()
            except Exception:
                pass

    def _all_uids_sync(self) -> list[str]:
        client = self._imap_connect()
        try:
            status, data = client.uid("search", None, "ALL")
            if status != "OK" or not data:
                return []
            return [item.decode("ascii") for item in data[0].split() if item]
        finally:
            try:
                client.logout()
            except Exception:
                pass

    def _fetch_raw_sync(self, uid: str) -> bytes | None:
        client = self._imap_connect()
        try:
            status, data = client.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not data:
                return None
            for item in data:
                if isinstance(item, tuple) and len(item) >= 2:
                    return bytes(item[1])
            return None
        finally:
            try:
                client.logout()
            except Exception:
                pass

    async def find_sent_by_message_id(
        self, message_id: str, *, request_id: int | None = None
    ) -> str | None:
        return None

    async def send(
        self,
        *,
        to: str,
        token: str,
        subject_suffix: str,
        body: str,
        request_id: int | None = None,
        message_id: str | None = None,
    ) -> tuple[str, str]:
        """Сформировать письмо и сохранить его как черновик в Яндекс.Почте."""
        raw_b64, message_id = build_message(
            sender=self._settings.yandex_email,
            sender_name="Medisent",
            to=to,
            token=token,
            subject_suffix=subject_suffix,
            body=body,
            message_id=message_id,
        )
        mailbox = await asyncio.to_thread(self._append_draft_bytes, _decode_raw_message(raw_b64))
        logger.info(
            "Черновик сохранён в Яндекс.Почте (%s) для %s, msgid=%s",
            mailbox,
            to,
            message_id,
            extra=log_extra(request_id),
        )
        return message_id, message_id

    async def forward_file(
        self,
        *,
        to: str,
        filename: str,
        content: bytes,
        mime_type: str,
        request_id: int | None = None,
    ) -> None:
        """Сохранить письмо с вложением как черновик вместо автоматической отправки."""
        message = EmailMessage()
        message["Subject"] = f"Файл из Telegram: {filename}"
        message["From"] = self._settings.yandex_email
        message["To"] = to
        message["Message-ID"] = new_message_id(self._settings.yandex_email)
        message.set_content("Файл подготовлен ботом MEDISENT.")
        maintype, _, subtype = mime_type.partition("/")
        message.add_attachment(
            content,
            maintype=maintype or "application",
            subtype=subtype or "octet-stream",
            filename=filename,
        )
        mailbox = await asyncio.to_thread(self._append_draft_bytes, message.as_bytes())
        logger.info(
            "Черновик с файлом %s сохранён в %s для %s",
            filename,
            mailbox,
            to,
            extra=log_extra(request_id),
        )

    async def current_history_id(self, request_id: int | None = None) -> str | None:
        uids = await asyncio.to_thread(self._all_uids_sync)
        return max(uids, key=int) if uids else "0"

    async def new_message_ids(
        self, start_history_id: str, *, request_id: int | None = None
    ) -> tuple[list[str], str | None]:
        uids = await asyncio.to_thread(self._all_uids_sync)
        if not uids:
            return [], "0"
        latest = max(uids, key=int)
        try:
            start = int(start_history_id)
        except (TypeError, ValueError):
            start = int(latest)
        if start > int(latest):
            return [], latest
        return [uid for uid in uids if int(uid) > start], latest

    async def get_message(
        self, message_id: str, *, request_id: int | None = None, full: bool = True
    ) -> ReplyHeaders | None:
        raw = await asyncio.to_thread(self._fetch_raw_sync, message_id)
        if raw is None:
            logger.warning("Не удалось прочитать письмо UID=%s", message_id)
            return None
        parsed = _parse_rfc822(raw, message_id)
        if not full:
            parsed.body = ""
            parsed.attachments = []
        return parsed

    async def download_attachment(
        self, message_id: str, attachment_id: str, *, request_id: int | None = None
    ) -> bytes | None:
        raw = await asyncio.to_thread(self._fetch_raw_sync, message_id)
        if raw is None:
            return None
        message = BytesParser(policy=policy.default).parsebytes(raw)
        parts = _attachment_parts(message)
        try:
            part = parts[int(attachment_id)]
        except (ValueError, IndexError):
            return None
        return part.get_payload(decode=True) or b""


_service: MailService | None = None


def get_mail_service() -> MailService:
    global _service
    if _service is None:
        _service = MailService()
    return _service


async def close_mail_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None
