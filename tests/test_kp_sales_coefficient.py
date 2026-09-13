from decimal import Decimal

from bot.services.kp import BASE_SALES_COEFFICIENT, ExtractedItem, Extraction, build_kp_json


def test_sales_coefficient_is_18() -> None:
    assert BASE_SALES_COEFFICIENT == Decimal("1.8")
    item = ExtractedItem(name="Товар", qty=Decimal("2"), price=Decimal("100"))
    assert item.sale_price == Decimal("180.00")
    assert item.sale_total == Decimal("360.00")


def test_kp_uses_sale_price_not_purchase_price() -> None:
    extraction = Extraction(items=[ExtractedItem(name="Товар", qty=Decimal("1"), price=Decimal("100"))])
    payload, _ = build_kp_json(extraction, number="КП-1", client_name="Клиент")
    assert payload["items"][0]["price"] == 180.0
