from bot.services.batch_intake import deserialise_batch, parse_text_batch, serialise_batch


def test_parse_multiline_procurement_list() -> None:
    text = """1. Цоликлон Анти-А жидкий, готовый 1x10 фл 600 фл\n2. Цоликлон Анти-В жидкий, готовый 1x10 фл 600 фл\n3. Цоликлон Анти-D жидкий, готовый 1x10 фл 600 фл"""
    batch = parse_text_batch(text)
    assert batch.is_multi
    assert len(batch.items) == 3
    assert batch.items[0].qty == "600"
    assert "Анти-А" in batch.items[0].product


def test_batch_roundtrip() -> None:
    batch = parse_text_batch("A reagent 10 фл\nB reagent 20 фл")
    restored = deserialise_batch(serialise_batch(batch))
    assert restored is not None
    assert [x.product for x in restored.items] == ["A reagent", "B reagent"]
    assert [x.qty for x in restored.items] == ["10", "20"]
