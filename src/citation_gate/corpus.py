"""The library that quotes are checked against."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .align import Alignment, align
from .normalize import Folded, fold, fold_text

_PART_SPLIT = re.compile(r"[·・‧•．.:：/\s]+")

# Bigrams more frequent than this vote for too many places to be useful.
_MAX_POSTINGS = 5000


@dataclass
class Doc:
    path: str  # relative to the corpus root, posix separators
    book: str
    text: str
    folded: Folded = field(repr=False)

    def line_of(self, folded_index: int) -> int:
        return self.text.count("\n", 0, self.folded.offsets[folded_index]) + 1

    def excerpt(self, start: int, end: int) -> str:
        return self.folded.span(self.text, start, end)


@dataclass(frozen=True)
class Match:
    doc: Doc
    start: int  # folded indices into doc
    end: int
    distance: int
    edits: tuple

    @property
    def length(self) -> int:
        return self.end - self.start

    def similarity(self, quote_len: int) -> float:
        return 1.0 - self.distance / quote_len if quote_len else 0.0

    def location(self) -> str:
        return f"{self.doc.path}:{self.doc.line_of(self.start)}"


class Corpus:
    def __init__(self, docs: list[Doc], aliases: dict[str, list[str]] | None = None):
        self.docs = docs
        self.books: dict[str, list[Doc]] = defaultdict(list)
        for d in docs:
            self.books[d.book].append(d)
        self._keys: dict[str, set[str]] = {}
        for book in self.books:
            keys = {fold_text(book)}
            keys.update(fold_text(a) for a in (aliases or {}).get(book, []))
            self._keys[book] = {k for k in keys if k}
        self._index: dict[str, list[tuple[int, int]]] | None = None

    @classmethod
    def load(cls, root: str | Path, pattern: str = "**/*.txt") -> "Corpus":
        """Load every file matching `pattern` under `root`.

        A book's title comes from `books.json` ("titles": {file: title}), else
        from a leading "# Title" line, else from the file name. Files sharing a
        title form one book. `books.json` may also list "aliases": {title: [...]}.
        """
        root = Path(root)
        meta_path = root / "books.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        titles = meta.get("titles", {})
        docs = []
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            rel = path.relative_to(root).as_posix()
            first = text.lstrip().split("\n", 1)[0]
            title = titles.get(rel) or (first[2:].strip() if first.startswith("# ") else path.stem)
            docs.append(Doc(rel, title, text, fold(text)))
        if not docs:
            raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
        return cls(docs, meta.get("aliases"))

    @classmethod
    def from_texts(cls, books: dict[str, str], aliases: dict[str, list[str]] | None = None) -> "Corpus":
        return cls([Doc(f"{b}.txt", b, t, fold(t)) for b, t in books.items()], aliases)

    def resolve(self, attribution: str) -> list[str]:
        """Books an attribution like 《黄帝内经·素问·上古天真论》 can refer to.

        Exact title or alias matches win; otherwise a part may abbreviate a
        title (伤寒 for 伤寒论) or extend one (素问上古天真论篇). An empty list
        means the cited work is not in the corpus.
        """
        parts = [fold_text(p) for p in _PART_SPLIT.split(attribution)]
        parts = [p for p in parts if p]
        exact = [b for b, keys in self._keys.items() if any(p in keys for p in parts)]
        if exact:
            return exact
        return [
            b
            for b, keys in self._keys.items()
            if any(len(p) >= 2 and (k.startswith(p) or p.startswith(k)) for p in parts for k in keys)
        ]

    @property
    def index(self) -> dict[str, list[tuple[int, int]]]:
        if self._index is None:
            index: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for di, d in enumerate(self.docs):
                t = d.folded.text
                for p in range(len(t) - 1):
                    index[t[p : p + 2]].append((di, p))
            self._index = dict(index)
        return self._index

    def find(self, s: str) -> tuple[Doc, int] | None:
        """First exact occurrence of a folded string."""
        for d in self.docs:
            p = d.folded.text.find(s)
            if p >= 0:
                return d, p
        return None

    def search(self, quote: str, books: list[str] | None = None, candidates: int = 5) -> list[Match]:
        """Best alignments of a folded quote, closest first, optionally only
        within `books`.

        Exact occurrences are found directly (one per document). Otherwise
        bigrams vote for alignment diagonals, the top few diagonals become
        windows, and each window is aligned.
        """
        m = len(quote)
        allowed = [books is None or d.book in books for d in self.docs]
        matches = []
        for d, ok in zip(self.docs, allowed):
            p = d.folded.text.find(quote) if ok else -1
            if p >= 0:
                matches.append(Match(d, p, p + m, 0, ()))
        if matches:
            return matches

        votes: Counter[tuple[int, int]] = Counter()
        for qi in range(m - 1):
            posts = self.index.get(quote[qi : qi + 2])
            if not posts or len(posts) > _MAX_POSTINGS:
                continue
            for di, p in posts:
                if allowed[di]:
                    votes[(di, p - qi)] += 1

        # An insertion or deletion shifts the diagonal for the rest of the
        # quote, splitting one true location's votes across neighbours. Score
        # each diagonal by the votes within ±pad, or a phrase repeated across
        # the corpus (宜大承气汤) outvotes the place the quote came from.
        pad = m // 4 + 2
        band: dict[tuple[int, int], int] = {}
        by_doc: dict[int, list[int]] = defaultdict(list)
        for di, diag in votes:
            by_doc[di].append(diag)
        for di, diags in by_doc.items():
            diags.sort()
            lo = hi = 0
            total = 0
            for diag in diags:
                while hi < len(diags) and diags[hi] <= diag + pad:
                    total += votes[(di, diags[hi])]
                    hi += 1
                while diags[lo] < diag - pad:
                    total -= votes[(di, diags[lo])]
                    lo += 1
                band[(di, diag)] = total

        chosen: list[tuple[int, int]] = []
        for di, diag in sorted(band, key=lambda k: (-band[k], -votes[k])):
            if any(di == cd and abs(diag - cg) <= pad for cd, cg in chosen):
                continue
            chosen.append((di, diag))
            if len(chosen) == candidates:
                break

        for di, diag in chosen:
            d = self.docs[di]
            lo = max(0, diag - pad)
            window = d.folded.text[lo : diag + m + pad]
            a: Alignment = align(quote, window)
            matches.append(Match(d, lo + a.start, lo + a.end, a.distance, a.edits))
        matches.sort(key=lambda x: x.distance)
        return matches
