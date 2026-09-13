"""Скрейп и точечный поиск по сайтам поставщиков через Firecrawl.

Только сайты поставщиков. К gateway реестра elk Firecrawl не применяется —
там свой JSON-API, и это отдельное жёсткое правило проекта.

Со страницы берутся наличие, цена, e-mail, телефон и исходный markdown. Если
первичная ссылка ведёт на главную/категорию и товара там нет, ``search_site``
ищет внутри конкретного домена наиболее релевантную страницу товара.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

from bot.config import get_settings
from bot.logging_setup import log_extra
from bot.services import guard, pricing
from bot.services.contacts import extract_email
from bot.services.http import ApiClient

logger = logging.getLogger(__name__)

API_URL = "https://api.firecrawl.dev/v1/scrape"
SEARCH_API_URL = "https://api.firecrawl.dev/v1/search"

PHONE_RE = re.compile(r"(?:\+7|8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
PRICE_RE = re.compile(
    r"(\d{1,3}(?:[\s ]\d{3})+|\d{4,9})(?:[.,](\d{1,2}))?\s*(?:руб|₽|r\.|rub)",
    re.IGNORECASE,
)
IN_STOCK_MARKERS = ("в наличии", "есть в наличии", "на складе", "готово к отгрузке", "in stock")
OUT_OF_STOCK_MARKERS = (
    "нет в наличии", "под заказ", "распродано", "снят с производства",
    "временно отсутствует", "out of stock",
)
_GENERIC_SEARCH_TERMS = {
    "степлер", "кожный", "одноразовый", "одноразовая", "стерильный", "стерильная",
    "изделие", "медицинский", "медицинское", "набор", "система", "инструмент",
    "аппарат", "устройство", "скоба", "скобы", "упаковка",
}


@dataclass(slots=True)
class ScrapeResult:
    url: str
    ok: bool = False
    claims_stock: bool | None = None
    price: Decimal | None = None
    email: str | None = None
    phone: str | None = None
    markdown: str = ""
    error: str | None = None
    injection_suspected: bool = False


def _extract_price(text: str) -> Decimal | None:
    prices: list[Decimal] = []
    for match in PRICE_RE.finditer(text):
        whole = re.sub(r"[\s ]", "", match.group(1))
        fraction = match.group(2) or "0"
        try:
            value = Decimal(f"{whole}.{fraction}")
        except InvalidOperation:
            continue
        if Decimal(100) <= value <= Decimal(100_000_000):
            prices.append(value)
    return min(prices) if prices else None


def _detect_stock(text: str, product: str) -> bool | None:
    lowered = text.lower()
    keywords = [word for word in product.lower().split() if len(word) > 4][:3]
    window = lowered
    if keywords:
        positions = [lowered.find(word) for word in keywords if lowered.find(word) >= 0]
        if positions:
            start = max(0, min(positions) - 1500)
            window = lowered[start : min(positions) + 3000]
    has_in = any(marker in window for marker in IN_STOCK_MARKERS)
    has_out = any(marker in window for marker in OUT_OF_STOCK_MARKERS)
    if has_in and not has_out:
        return True
    if has_out and not has_in:
        return False
    return None


def _product_terms(product: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9._/-]{2,}", product):
        token = raw.strip(".,;:()[]{}").lower()
        if len(token) < 4 or token in _GENERIC_SEARCH_TERMS or token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def _site_domain(url: str) -> str:
    candidate = url if "://" in url else "https://" + url
    return (urlparse(candidate).hostname or "").lower().removeprefix("www.")


def _score_search_hit(product: str, text: str, url: str) -> int:
    lowered = text.lower()
    score = 0
    for term in _product_terms(product):
        if term in lowered:
            score += 4 if any(ch.isdigit() for ch in term) else 2
        if term in url.lower():
            score += 3
    if any(x in lowered for x in ("купить", "цена", "в наличии", "заказать", "запросить цену")):
        score += 2
    return score


class FirecrawlService:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = ApiClient(
            "firecrawl",
            headers={
                "Authorization": f"Bearer {settings.firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            timeout_read=120.0,
        )
        self._semaphore = asyncio.Semaphore(settings.scrape_concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def scrape(self, url: str, product: str, *, request_id: int | None = None) -> ScrapeResult:
        if not self._settings.scrape_enabled:
            return ScrapeResult(url=url, error="FIRECRAWL_API_KEY не задан")
        async with self._semaphore:
            result = await self._client.post(
                API_URL,
                operation="scrape",
                request_id=request_id,
                cost_usd=pricing.flat_cost("firecrawl"),
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True, "timeout": 60000},
            )
        if not result.ok:
            return ScrapeResult(url=url, error=result.error or "Firecrawl не ответил")
        payload = (result.json or {}).get("data") or {}
        markdown = str(payload.get("markdown") or "")
        if not markdown.strip():
            return ScrapeResult(url=url, error="страница пустая")
        screening = await guard.screen_third_party_async(markdown, source=f"сайт {url}")
        return ScrapeResult(
            url=url,
            ok=True,
            claims_stock=_detect_stock(markdown, product),
            price=_extract_price(markdown),
            email=extract_email(markdown),
            phone=(m.group(0) if (m := PHONE_RE.search(markdown)) else None),
            markdown=markdown,
            injection_suspected=screening.suspicious,
        )

    async def search_site(
        self,
        site_url: str,
        product: str,
        *,
        request_id: int | None = None,
        limit: int = 5,
    ) -> ScrapeResult | None:
        """Найти внутри одного домена наиболее релевантную страницу товара.

        Используется только как fallback, когда исходная ссылка поставщика не содержит
        точного товара. Это позволяет находить карточки, спрятанные глубже каталога.
        """
        if not self._settings.scrape_enabled:
            return None
        domain = _site_domain(site_url)
        if not domain:
            return None
        exact = [t for t in _product_terms(product) if any(ch.isdigit() for ch in t)]
        anchor = " ".join(f'"{t}"' for t in exact[:2]) or f'"{product}"'
        query = f"site:{domain} {anchor}"
        async with self._semaphore:
            result = await self._client.post(
                SEARCH_API_URL,
                operation="site_search",
                request_id=request_id,
                cost_usd=pricing.flat_cost("firecrawl"),
                json={
                    "query": query,
                    "limit": max(1, min(limit, 8)),
                    "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
                },
            )
        if not result.ok:
            logger.info("Внутренний поиск %s не удался: %s", domain, result.error, extra=log_extra(request_id))
            return None
        rows = (result.json or {}).get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("web") or rows.get("results") or []
        if not isinstance(rows, list):
            return None
        best_url = ""
        best_text = ""
        best_score = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not url or _site_domain(url) != domain:
                continue
            text = "\n".join(str(row.get(k) or "") for k in ("title", "description", "markdown", "content"))
            score = _score_search_hit(product, text, url)
            if score > best_score:
                best_score, best_url, best_text = score, url, text
        if not best_url or best_score <= 0:
            return None
        # Search API может вернуть уже извлечённый markdown, но повторный scrape даёт
        # единый формат, контакты, цену, наличие и защиту от prompt injection.
        scraped = await self.scrape(best_url, product, request_id=request_id)
        if scraped.ok:
            return scraped
        if best_text.strip():
            screening = await guard.screen_third_party_async(best_text, source=f"поиск по сайту {best_url}")
            return ScrapeResult(
                url=best_url,
                ok=True,
                claims_stock=_detect_stock(best_text, product),
                price=_extract_price(best_text),
                email=extract_email(best_text),
                phone=(m.group(0) if (m := PHONE_RE.search(best_text)) else None),
                markdown=best_text,
                injection_suspected=screening.suspicious,
            )
        return None

    async def scrape_many(self, urls: list[str], product: str, *, request_id: int | None = None) -> list[ScrapeResult]:
        if not urls:
            return []
        results = await asyncio.gather(
            *(self.scrape(url, product, request_id=request_id) for url in urls),
            return_exceptions=True,
        )
        out: list[ScrapeResult] = []
        for url, item in zip(urls, results, strict=True):
            if isinstance(item, BaseException):
                logger.warning("Скрейп %s упал: %s", url, item, extra=log_extra(request_id))
                out.append(ScrapeResult(url=url, error=str(item)))
            else:
                out.append(item)
        return out


_service: FirecrawlService | None = None


def get_firecrawl_service() -> FirecrawlService:
    global _service
    if _service is None:
        _service = FirecrawlService()
    return _service


async def close_firecrawl_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None
