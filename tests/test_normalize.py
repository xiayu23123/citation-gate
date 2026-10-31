import pytest

from citation_gate.normalize import ProfileError, fold, fold_text, load_confusable, load_critical, load_terms, simplify


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
    assert {"桂枝汤", "大承气汤", "白虎汤", "理中丸"} <= tcm
    assert all(fold_text(t) == t for t in tcm)
    assert load_terms("none") == frozenset()
    own = tmp_path / "terms.txt"
    own.write_text("# herbs\n桂枝 74\n\n麻黃\n", encoding="utf-8")
    assert load_terms(str(own)) == {"桂枝", "麻黄"}


def test_profiles_combine(tmp_path):
    tcm, shushu = load_critical("tcm"), load_critical("shushu")
    assert {"吉", "凶", "巳", "孙", "官"} <= shushu  # folded: 子孫 in the corpus
    assert "干" in shushu  # 乾 folds to 干
    assert load_critical("tcm,shushu") == tcm | shushu
    assert load_critical(" shushu , tcm ") == tcm | shushu
    assert load_critical("none,tcm") == tcm
    own = tmp_path / "own.txt"
    own.write_text("# mine\nextra: 曰\n", encoding="utf-8")
    assert load_critical(f"tcm,{own}") == tcm | {"曰"}
    assert load_terms("tcm,none") == load_terms("tcm")


def test_profile_path_with_a_comma(tmp_path):
    own = tmp_path / "a,b.txt"
    own.write_text("x: 曰\n", encoding="utf-8")
    assert load_critical(str(own)) == {"曰"}


def test_unknown_profile_is_an_error():
    with pytest.raises(ProfileError, match="nosuch"):
        load_critical("tcm,nosuch")
    with pytest.raises(ProfileError):
        load_terms("nosuch")
    with pytest.raises(ProfileError):
        load_confusable("nosuch")


def test_confusable_sets(tmp_path):
    sets = load_confusable()
    assert frozenset("己已巳") in sets and frozenset("戊戍戌") in sets
    assert frozenset("干千于") in sets  # also 乾 and 於, which fold to 干 and 于
    assert all(len(s) >= 2 and all(fold_text(c) == c for c in s) for s in sets)
    assert load_confusable("none") == frozenset()
    own = tmp_path / "sets.txt"
    own.write_text("# mine\n曰日 note\n人\n", encoding="utf-8")
    assert load_confusable(str(own)) == {frozenset("曰日")}  # a set of one is dropped
