from citation_gate.normalize import fold, fold_text, load_critical, load_terms, simplify


def test_traditional_and_simplified_fold_together():
    assert fold_text("女子七歲，腎氣盛，齒更髮長。") == fold_text("女子七岁 肾气盛 齿更发长")


def test_variants_beyond_opencc():
    assert fold_text("心下痞鞕") == fold_text("心下痞硬")
    assert fold_text("脈沈") == fold_text("脉沉")
    assert fold_text("黃帝內經") == fold_text("黄帝内经")


def test_punctuation_and_width_are_dropped():
    assert fold_text("「太陽病，發熱。」") == fold_text("太阳病发热")
    assert fold_text("ＡＢ１") == "ab1"


def test_offsets_point_into_original():
    s = "甲，乙。丙"
    f = fold(s)
    assert f.text == "甲乙丙"
    assert f.span(s, 1, 3) == "乙。丙"


def test_simplify_keeps_punctuation():
    assert simplify("腎氣盛，齒更髮長。") == "肾气盛，齿更发长。"


def test_critical_profiles():
    tcm = load_critical("tcm")
    assert {"不", "肾", "寒", "七"} <= tcm
    assert "曰" not in tcm
    assert load_critical("none") == frozenset()


def test_term_profiles(tmp_path):
    tcm = load_terms("tcm")
    assert {"桂枝", "麻黄", "芍药", "干姜"} <= tcm  # folded: 麻黃, 芍藥, 乾薑 in the corpus
    assert all(fold_text(t) == t for t in tcm)
    assert load_terms("none") == frozenset()
    own = tmp_path / "terms.txt"
    own.write_text("# herbs\n桂枝 74\n\n麻黃\n", encoding="utf-8")
    assert load_terms(str(own)) == {"桂枝", "麻黄"}
