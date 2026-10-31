from citation_gate.extract import extract_quotes, uncited_sentences


def test_attribution_before_quote():
    (q,) = extract_quotes("《素问·上古天真论》云：「女子七岁，肾气盛。」")
    assert q.text == "女子七岁，肾气盛。"
    assert q.attribution == "素问·上古天真论"


def test_attribution_after_quote():
    (q,) = extract_quotes("原文是“太阳之为病，脉浮”（《伤寒论》）。")
    assert q.attribution == "伤寒论"
    (q,) = extract_quotes("“太阳之为病，脉浮”——《伤寒论》")
    assert q.attribution == "伤寒论"


def test_attribution_does_not_cross_sentences():
    _, second = extract_quotes("《伤寒论》说「甲乙丙丁」。另一句话「戊己庚辛」。")
    assert second.attribution is None


def test_nested_quotes_belong_to_parent():
    (q,) = extract_quotes("「经曰：『虚虚实实』，是其义也」")
    assert q.text == "经曰：『虚虚实实』，是其义也"


def test_unclosed_quote_is_ignored():
    assert extract_quotes("他说「没有结尾") == []


def test_uncited_sentences():
    answer = "《素问》云：「女子七岁，肾气盛。」所以补肾很重要，人人都该吃补药。真的吗？好。"
    quotes = extract_quotes(answer)
    assert uncited_sentences(answer, quotes) == ["所以补肾很重要，人人都该吃补药。"]
