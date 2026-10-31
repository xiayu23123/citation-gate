"""Decide, quote by quote, whether an answer's citations hold up."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .corpus import Corpus, Match
from .extract import Quote, extract_quotes, uncited_sentences
from .normalize import fold_text, load_critical


class Status(str, Enum):
    VERIFIED = "verified"  # verbatim in the cited work (or anywhere, if none cited)
    FUZZY = "fuzzy"  # near-verbatim; the differences touch no critical character
    ALTERED = "altered"  # near-verbatim, but a critical character differs
    MISATTRIBUTED = "misattributed"  # in the corpus, but not in the cited work
    SPLICED = "spliced"  # stitched from fragments of different passages
    UNKNOWN_SOURCE = "unknown_source"  # cites a work the corpus does not have
    NOT_FOUND = "not_found"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Policy:
    min_chars: int = 4  # shorter quotations are emphasis, not citations
    fuzzy_threshold: float = 0.85
    critical: frozenset[str] = field(default_factory=lambda: load_critical("tcm"))
    detect_splices: bool = True
    allow_unknown_sources: bool = False
    require_citation: bool = False

    def fails(self, status: Status) -> bool:
        if status in (Status.VERIFIED, Status.FUZZY):
            return False
        if status is Status.UNKNOWN_SOURCE:
            return not self.allow_unknown_sources
        return True


@dataclass(frozen=True)
class Segment:
    start: int  # in the folded quote
    end: int
    doc_path: str
    pos: int  # folded index in the doc
    line: int


@dataclass
class QuoteResult:
    quote: Quote
    folded: str
    status: Status
    match: Match | None = None
    cited_books: list[str] | None = None  # None: no attribution given
    segments: list[Segment] = field(default_factory=list)

    @property
    def similarity(self) -> float:
        return self.match.similarity(len(self.folded)) if self.match else 0.0

    def note(self) -> str:
        if self.status is Status.SPLICED:
            return " + ".join(f"{s.doc_path}:{s.line}" for s in self.segments)
        if not self.match:
            return ""
        edits = " ".join(str(e) for e in self.match.edits)
        where = f"{self.match.doc.book} {self.match.location()}"
        return f"{where} {edits}".strip()


@dataclass
class AnswerReport:
    id: str
    results: list[QuoteResult]
    skipped: list[Quote]
    uncited: list[str]

    def failures(self, policy: Policy) -> list[QuoteResult]:
        return [r for r in self.results if policy.fails(r.status)]

    def failed(self, policy: Policy) -> bool:
        return bool(self.failures(policy)) or (policy.require_citation and bool(self.uncited))


def _classify(m: Match, quote_len: int, policy: Policy) -> Status | None:
    if m.distance == 0:
        return Status.VERIFIED
    if m.similarity(quote_len) < policy.fuzzy_threshold:
        return None
    if any(e.chars() & policy.critical for e in m.edits):
        return Status.ALTERED
    return Status.FUZZY


_RANK = {Status.VERIFIED: 0, Status.FUZZY: 1, Status.ALTERED: 2}


def _best(matches: list[Match], quote_len: int, policy: Policy) -> tuple[Status, Match] | None:
    graded = [(s, m) for m in matches if (s := _classify(m, quote_len, policy))]
    if not graded:
        return None
    return min(graded, key=lambda sm: (_RANK[sm[0]], sm[1].distance))


def _splice(corpus: Corpus, q: str, min_seg: int = 4) -> list[Segment]:
    """Greedy cover of the quote by maximal exact fragments of the corpus."""
    segs: list[Segment] = []
    i = 0
    while i <= len(q) - min_seg:
        hit = corpus.find(q[i : i + min_seg])
        if not hit:
            i += 1
            continue
        k = min_seg
        while i + k < len(q) and (longer := corpus.find(q[i : i + k + 1])):
            hit, k = longer, k + 1
        doc, pos = hit
        segs.append(Segment(i, i + k, doc.path, pos, doc.line_of(pos)))
        i += k
    return segs


def _is_splice(segs: list[Segment], quote_len: int, slack: int = 3) -> bool:
    """Fragments cover most of the quote and do not sit in one passage.

    Fragments that line up in the source are one noisy passage, not a splice.
    """
    if len(segs) < 2 or sum(s.end - s.start for s in segs) < 0.8 * quote_len:
        return False
    return any(
        a.doc_path != b.doc_path or abs((b.pos - a.pos) - (b.start - a.start)) > slack
        for a, b in zip(segs, segs[1:])
    )


def check_quote(corpus: Corpus, quote: Quote, policy: Policy) -> QuoteResult | None:
    q = fold_text(quote.text)
    if len(q) < policy.min_chars:
        return None
    cited = corpus.resolve(quote.attribution) if quote.attribution else None

    if cited == []:
        found = _best(corpus.search(q), len(q), policy)
        return QuoteResult(quote, q, Status.UNKNOWN_SOURCE, found[1] if found else None, cited)

    scoped = corpus.search(q, books=cited)
    best = _best(scoped, len(q), policy)
    if best:
        return QuoteResult(quote, q, best[0], best[1], cited)

    everywhere = corpus.search(q) if cited else scoped
    elsewhere = _best(everywhere, len(q), policy)
    if elsewhere and elsewhere[0] is not Status.ALTERED:
        return QuoteResult(quote, q, Status.MISATTRIBUTED, elsewhere[1], cited)

    if policy.detect_splices:
        segs = _splice(corpus, q)
        if _is_splice(segs, len(q)):
            return QuoteResult(quote, q, Status.SPLICED, None, cited, segs)

    closest = everywhere[0] if everywhere else None
    return QuoteResult(quote, q, Status.NOT_FOUND, closest, cited)


def check_answer(corpus: Corpus, answer: str, policy: Policy | None = None, id: str = "") -> AnswerReport:
    policy = policy or Policy()
    quotes = extract_quotes(answer)
    results, skipped = [], []
    for quote in quotes:
        r = check_quote(corpus, quote, policy)
        if r:
            results.append(r)
        else:
            skipped.append(quote)
    return AnswerReport(id, results, skipped, uncited_sentences(answer, quotes))
