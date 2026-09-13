"""Direct HTTP first for supplier pages; Firecrawl is only a fallback.

This policy wraps ``FirecrawlService.scrape`` without changing its public API.
For ordinary HTML pages we fetch the page directly through the project's
metered ``ApiClient`` and extract the same basic commercial signals. Only when
direct HTTP fails, returns unusable content, or does not contain a product
signal do we call the original Firecrawl implementation.
"""
from __future__ import annotations

import html
import ipaddress
import logging
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from bot.logging_setup import log_extra
from bot.services import guard
from bot.services.contacts import extract_email
from bot.services.firecrawl import (
    PHONE_RE,
    FirecrawlService,
    ScrapeResult,
    _detect_stock,
    _extract_price,
    _has_product_signal,
)
from bot.services.http import ApiClient

logger = logging.getLogger(__name__)
_ORIGINAL_SCRAPE: Any = None


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "template"}:
            self._hidden_depth += 1
        elif tag.lower() in {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "template"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n", value)
        return value.strip()


def _safe_public_url(url: str) -> bool:
    """Reject obviously unsafe/non-web targets before following external URLs."""
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _html_to_text(body: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(body)
        return parser.text()
    except Exception:
        # Even malformed supplier HTML can still provide useful plain text.
        return re.sub(r"<[^>]+>", " ", body)


async def _direct_fetch(
    service: FirecrawlService,
    url: str,
    product: str,
    *,
    request_id: int | None,
) -> ScrapeResult | None:
    if not _safe_public_url(url):
        return None
    client = getattr(service, "_direct_web_client", None)
    if client is None:
        client = ApiClient(
            "direct_web",
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MedisentBot/1.0; +medical procurement)",
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.8,text/plain;q=0.7,*/*;q=0.5",
                "Accept-Language": "ru,en;q=0.7",
            },
            timeout_read=20.0,
            max_retries=1,
        )
        setattr(service, "_direct_web_client", client)

    result = await client.get(
        url if "://" in url else "https://" + url,
        operation="supplier.fetch",
        request_id=request_id,
        expect_json=False,
    )
    if not result.ok or not result.text.strip():
        logger.info(
            "Direct web: %s не прочитан (%s), оставляю Firecrawl fallback",
            url,
            result.error or result.status_code,
            extra=log_extra(request_id),
        )
        return None

    content_type = str(result.headers.get("content-type") or "").lower()
    raw = result.text[:1_500_000]
    text = _html_to_text(raw) if "html" in content_type or "<html" in raw[:1000].lower() else raw
    text = text[:250_000].strip()
    if len(text) < 80:
        return None

    screening = await guard.screen_third_party_async(text, source=f"сайт {url}")
    direct = ScrapeResult(
        url=url,
        ok=True,
        claims_stock=_detect_stock(text, product),
        price=_extract_price(text),
        email=extract_email(text),
        phone=(m.group(0) if (m := PHONE_RE.search(text)) else None),
        markdown=text,
        injection_suspected=screening.suspicious,
        evidence_url=url,
    )
    if _has_product_signal(product, text):
        logger.info(
            "Direct web: %s прочитан без Firecrawl, товарный сигнал найден",
            url,
            extra=log_extra(request_id),
        )
        return direct

    logger.info(
        "Direct web: %s доступен, но товарного сигнала нет — пробую Firecrawl/site search",
        url,
        extra=log_extra(request_id),
    )
    return direct


def install_direct_web_policy() -> None:
    global _ORIGINAL_SCRAPE
    if _ORIGINAL_SCRAPE is not None:
        return
    _ORIGINAL_SCRAPE = FirecrawlService.scrape
    original = _ORIGINAL_SCRAPE

    async def wrapped(
        self: FirecrawlService,
        url: str,
        product: str,
        *,
        request_id: int | None = None,
    ) -> ScrapeResult:
        direct = await _direct_fetch(self, url, product, request_id=request_id)
        if direct is not None and direct.ok and _has_product_signal(product, direct.markdown):
            return direct
        fallback = await original(self, url, product, request_id=request_id)
        if fallback.ok:
            return fallback
        # A directly readable page is still more useful than a failed paid fallback.
        return direct if direct is not None else fallback

    FirecrawlService.scrape = wrapped  # type: ignore[method-assign]
