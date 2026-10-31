import json

import pytest

from citation_gate import Corpus, Policy, Status, check_answer
from citation_gate.cli import main

BOOKS = {
    "素問": "女子七歲，腎氣盛，齒更髮長。二七而天癸至，任脈通，太衝脈盛，月事以時下，故有子。"
    "陰平陽秘，精神乃治；陰陽離決，精氣乃絕。",
    "傷寒論": "太陽之為病，脈浮，頭項強痛而惡寒。太陽病，發熱，汗出，惡風，脈緩者，名為中風。"
    "太陽病，外證未解，不可下也，下之為逆。",
}


@pytest.fixture(scope="module")
def corpus():
    return Corpus.from_texts(BOOKS, aliases={"素問": ["黃帝內經"]})


def status(corpus, answer, **policy):
    report = check_answer(corpus, answer, Policy(**policy))
    (r,) = report.results
    return r.status


def test_verified_across_scripts(corpus):
    assert status(corpus, "《黄帝内经·素问》云：「女子七岁，肾气盛，齿更发长」") is Status.VERIFIED


def test_ocr_noise_is_fuzzy(corpus):
    # 時→特 is a non-critical slip
    assert status(corpus, "《素问》：「任脉通，太冲脉盛，月事以特下，故有子」") is Status.FUZZY


def test_critical_substitution_is_altered(corpus):
    assert status(corpus, "《素问》：「女子七岁，肝气盛，齿更发长」") is Status.ALTERED


def test_dropped_negation_is_altered(corpus):
    assert status(corpus, "《伤寒论》：「太阳病，外证未解，可下也，下之为逆」") is Status.ALTERED


def test_without_critical_chars_negation_slips_through(corpus):
    answer = "《伤寒论》：「太阳病，外证未解，可下也，下之为逆」"
    assert status(corpus, answer, critical=frozenset()) is Status.FUZZY


def test_misattributed(corpus):
    assert status(corpus, "《素问》曰：「太阳之为病，脉浮，头项强痛而恶寒」") is Status.MISATTRIBUTED


def test_spliced(corpus):
    assert status(corpus, "「女子七岁，肾气盛，阴阳离决，精气乃绝」") is Status.SPLICED


def test_not_found(corpus):
    assert status(corpus, "《素问》：「正气存内，邪不可干」") is Status.NOT_FOUND


def test_unknown_source_fails_unless_allowed(corpus):
    answer = "《难经》曰：「人之所以生者，气也」"
    report = check_answer(corpus, answer)
    assert report.results[0].status is Status.UNKNOWN_SOURCE
    assert report.failed(Policy())
    assert not report.failed(Policy(allow_unknown_sources=True))


def test_short_quotes_are_skipped(corpus):
    report = check_answer(corpus, "所谓「肾气」，是指……")
    assert report.results == [] and len(report.skipped) == 1


def test_require_citation(corpus):
    answer = "《素问》：「女子七岁，肾气盛」。所以女孩七岁一定要补肾。"
    assert not check_answer(corpus, answer).failed(Policy())
    assert check_answer(corpus, answer).failed(Policy(require_citation=True))


def test_cli_check(tmp_path, capsys):
    (tmp_path / "corpus").mkdir()
    for book, text in BOOKS.items():
        (tmp_path / "corpus" / f"{book}.txt").write_text(f"# {book}\n\n{text}", encoding="utf-8")
    answers = tmp_path / "answers.jsonl"
    rows = [
        {"id": "ok", "answer": "《伤寒论》：「太阳之为病，脉浮，头项强痛而恶寒」"},
        {"id": "bad", "answer": "《伤寒论》：「太阳之为病，脉沉，头项强痛而恶热」"},
    ]
    answers.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp_path / "report.json"
    code = main(["check", str(answers), "--corpus", str(tmp_path / "corpus"), "--json", str(out)])
    assert code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["failed_answers"] == 1
    assert [a["failed"] for a in data["answers"]] == [False, True]
    assert "bad: " in capsys.readouterr().err
