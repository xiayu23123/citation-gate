from citation_gate.align import align


def test_exact_inside_window():
    a = align("bcd", "xxbcdxx")
    assert (a.distance, a.start, a.end, a.edits) == (0, 2, 5, ())


def test_substitution():
    a = align("bXd", "abcde")
    assert a.distance == 1
    assert [str(e) for e in a.edits] == ["c→X"]
    assert (a.start, a.end) == (1, 4)
    assert (a.edits[0].at, a.edits[0].src) == (1, 1)


def test_source_positions_are_relative_to_the_span():
    # The span is "abcdefgh". Its "c" (span index 2) is missing before quote
    # "d" (quote index 2); the quote's extra "X" (quote index 6) sits before
    # span "h" (span index 7).
    a = align("abdefgXh", "zzabcdefghzz")
    assert (a.start, a.end) == (2, 10)
    assert [(e.kind, e.at, e.src) for e in a.edits] == [("del", 2, 2), ("ins", 6, 7)]


def test_insertion_and_deletion():
    ins = align("abXc", "abc")
    assert ins.distance == 1 and ins.edits[0].kind == "ins" and ins.edits[0].quote == "X"
    dele = align("ac", "zabcz")
    assert dele.distance == 1 and dele.edits[0].kind == "del" and dele.edits[0].source == "b"


def test_dropped_char_near_end_is_reported_as_dropped():
    # "可" could also be explained as an extra quote character; we want -不.
    a = align("下之不可", "下之不可也")
    assert a.distance == 0
    a = align("外证未解可", "外证未解不可下")
    assert [str(e) for e in a.edits] == ["-不"]


def test_whole_quote_must_be_accounted_for():
    assert align("abc", "").distance == 3
