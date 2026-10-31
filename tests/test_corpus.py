import json
from pathlib import Path

from citation_gate import Corpus

ROOT = Path(__file__).resolve().parent.parent / "corpus"
DOMAINS = {"tcm": {"素問", "傷寒論", "金匱要略"}}


def test_books_json_applies_to_its_own_directory(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.txt").write_text("# 甲書\n\n天地玄黃", encoding="utf-8")
    (tmp_path / "b" / "y.txt").write_text("宇宙洪荒", encoding="utf-8")
    (tmp_path / "a" / "books.json").write_text(json.dumps({"aliases": {"甲書": ["甲"]}}), encoding="utf-8")
    (tmp_path / "b" / "books.json").write_text(json.dumps({"titles": {"y.txt": "乙書"}}), encoding="utf-8")
    whole = Corpus.load(tmp_path)
    assert {d.path: d.book for d in whole.docs} == {"a/x.txt": "甲書", "b/y.txt": "乙書"}
    assert whole.resolve("甲") == ["甲書"]
    assert [d.book for d in Corpus.load(tmp_path / "b").docs] == ["乙書"]


def test_bundled_corpus_is_one_directory_per_domain():
    assert not list(ROOT.glob("*.txt")), "corpus texts belong in a domain directory"
    assert {p.name for p in ROOT.iterdir() if p.is_dir()} == set(DOMAINS)
    for domain, books in DOMAINS.items():
        corpus = Corpus.load(ROOT / domain)
        assert set(corpus.books) == books
        # Every file is pinned to a revision in that domain's SOURCES.md.
        sources = (ROOT / domain / "SOURCES.md").read_text(encoding="utf-8")
        for d in corpus.docs:
            assert f"| `{d.path}` |" in sources and "oldid=" in sources


def test_domains_load_together():
    whole = Corpus.load(ROOT)
    assert set(whole.books) == set().union(*DOMAINS.values())
    assert all(d.path.split("/", 1)[0] in DOMAINS for d in whole.docs)
    assert whole.resolve("黃帝內經") == ["素問"]  # aliases from corpus/tcm/books.json
