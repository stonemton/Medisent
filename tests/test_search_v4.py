from bot.services.perplexity import (
    FoundSupplier,
    _initial_queries,
    _product_identifiers,
    _refinement_query,
    is_supplier_domain,
)


def test_product_identifiers_extracts_model_and_article() -> None:
    product = "Степлер кожный ALFASTEP QPWB-35N, артикул 0301-01M"
    assert _product_identifiers(product) == ["QPWB-35N", "0301-01M"]


def test_initial_queries_use_ru_holder_and_identifier() -> None:
    queries = _initial_queries(
        "Степлер кожный ALFASTEP QPWB-35N",
        requirements=["35 скоб"],
        ru_number="РЗН 2023/19450",
        holder='ООО "Альфастеп"',
    )
    names = [name for name, _ in queries]
    joined = "\n".join(query for _, query in queries)

    assert "identifier" in names
    assert "ru" in names
    assert "holder" in names
    assert "QPWB-35N" in joined
    assert "РЗН 2023/19450" in joined
    assert "Альфастеп" in joined


def test_refinement_query_asks_for_new_domains() -> None:
    query = _refinement_query(
        "Степлер ALFASTEP QPWB-35N",
        known=[FoundSupplier(name="Test", site="https://supplier.example/product")],
        ru_number="РЗН 2023/19450",
        holder="Альфастеп",
    )
    assert "supplier.example" in query
    assert "Не повторяй" in query
    assert "QPWB-35N" in query
    assert "РЗН 2023/19450" in query


def test_non_supplier_domains_remain_filtered() -> None:
    assert not is_supplier_domain("https://zakupki.gov.ru/example")
    assert not is_supplier_domain("https://www.ozon.ru/product/123")
    assert is_supplier_domain("https://medical-supplier.example/product")
