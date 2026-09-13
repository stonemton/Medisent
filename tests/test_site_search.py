from bot.services.firecrawl import _product_terms, _score_search_hit, _site_domain


def test_product_terms_keep_brand_and_model() -> None:
    terms = _product_terms("Степлер кожный ALFASTEP QPWB-35N")
    assert "alfastep" in terms
    assert "qpwb-35n" in terms
    assert "степлер" not in terms


def test_exact_model_scores_above_generic_brand_page() -> None:
    product = "Степлер кожный ALFASTEP QPWB-35N"
    brand_page = _score_search_hit(product, "ALFASTEP медицинские изделия", "https://example.ru/brands/alfastep")
    product_page = _score_search_hit(
        product,
        "ALFASTEP QPWB-35N купить, цена, в наличии",
        "https://example.ru/catalog/qpwb-35n",
    )
    assert product_page > brand_page


def test_site_domain_normalises_www() -> None:
    assert _site_domain("https://www.imsstore.ru/catalog/item") == "imsstore.ru"
