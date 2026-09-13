"""Оценка стоимости вызовов внешних сервисов.

Цифры — ориентиры для внутреннего лимитера. Реальный расход и баланс
проверяются у провайдера API.
"""

from __future__ import annotations

from decimal import Decimal

# Цена за миллион токенов, USD.
LLM_PRICES: dict[str, tuple[Decimal, Decimal]] = {
    # Google direct / старые алиасы
    "gemini-flash-latest": (Decimal("0.30"), Decimal("2.50")),
    "gemini-flash-lite-latest": (Decimal("0.10"), Decimal("0.40")),
    "gemini-pro-latest": (Decimal("1.25"), Decimal("10.00")),
    # RelayModels: в каталоге у Gemini 3.8 Flash единая цена input/output.
    "gemini-3.8-flash": (Decimal("0.075"), Decimal("0.075")),
}
LLM_PRICE_FALLBACK = (Decimal("0.30"), Decimal("2.50"))

# Фиксированная цена за вызов, USD.
FLAT_PRICES: dict[str, Decimal] = {
    "perplexity": Decimal("0.006"),
    "firecrawl": Decimal("0.001"),
    "gmail": Decimal("0"),
    "registry": Decimal("0"),
    "browseract": Decimal("0.02"),
}


def llm_cost(model: str, tokens_in: int, tokens_out: int) -> Decimal:
    """Стоимость одного вызова модели по числу токенов."""
    price_in, price_out = LLM_PRICES.get(model, LLM_PRICE_FALLBACK)
    million = Decimal(1_000_000)
    return (
        price_in * Decimal(tokens_in) / million + price_out * Decimal(tokens_out) / million
    ).quantize(Decimal("0.000001"))


def flat_cost(service: str, calls: int = 1) -> Decimal:
    """Стоимость сервиса с фиксированной ценой за вызов."""
    return (FLAT_PRICES.get(service, Decimal("0")) * Decimal(calls)).quantize(Decimal("0.000001"))
