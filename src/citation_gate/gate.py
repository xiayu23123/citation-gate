"""Decide, quote by quote, whether an answer's citations hold up."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .corpus import Corpus, Match
from .extract import Quote, extract_quotes, uncited_sentences
from .align import Edit
from .normalize import fold_text, load_confusable, load_critical, load_terms


class Status(str, Enum):
    VERIFIED = "verified"  # verbatim in the cited work (or anywhere, if none cited)
    FUZZY = "fuzzy"  # near-verbatim; the differences touch no critical character or term
    NEEDS_REVIEW = "needs_review"  # near-verbatim; the only critical differences are confusable characters
    ALTERED = "altered"  # near-verbatim, but a critical character or term differs
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
    critical_terms: frozenset[str] = field(default_factory=lambda: load_terms("tcm"))
    # Sets of characters that are confused for one another (己已巳); a change
    # within one set is needs_review, not altered.
    confusable: frozenset[frozenset[str]] = field(default_factory=load_confusable)
    detect_splices: bool = True
    allow_unknown_sources: bool = False
    review_as_pass: bool = False  # needs_review warns instead of failing
    require_citation: bool = False

    def fails(self, status: Status) -> bool:
        if status in (Status.VERIFIED, Status.FUZZY):
            return False
        if status is Status.UNKNOWN_SOURCE:
            return not self.allow_unknown_sources
        if status is Status.NEEDS_REVIEW:
            return not self.review_as_pass
        return True

    def warns(self, status: Status) -> bool:
        """Passes only because a flag let it: a person should still look."""
        return status in (Status.UNKNOWN_SOURCE, Status.NEEDS_REVIEW) and not self.fails(status)


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

    def warnings(self, policy: Policy) -> list[QuoteResult]:
        return [r for r in self.results if policy.warns(r.status)]

    def failed(self, policy: Policy) -> bool:
        return bool(self.failures(policy)) or (policy.require_citation and bool(self.uncited))


def _in_term(text: str, pos: int, gap: bool, terms: frozenset[str], lengths: set[int]) -> bool:
    """Whether text[pos], or with `gap` the point just before it, lies inside
    an occurrence of one of `terms`. A gap must be strictly inside: 桂|枝, not |桂枝."""
    for n in lengths:
        for s in range(max(0, pos - n + 1), min(pos + 1 - gap, len(text) - n + 1)):
            if text[s : s + n] in terms:
                return True
    return False


def _critical_edits(m: Match, quote: str, policy: Policy) -> list[Edit]:
    """Edits that touch a critical character, or fall inside a critical term
    in the quote or in the source span it was aligned to.

    Both sides matter for terms: 桂枝→麻黄 breaks 桂枝 in the source and 麻黄
    in the quote, and either one alone is enough to know the prescription
    changed.
    """
    terms = policy.critical_terms
    source = m.doc.folded.text[m.start : m.end]
    lengths = {len(t) for t in terms}
    return [
        e
        for e in m.edits
        if e.chars() & policy.critical
        or (
            terms
            and (
                _in_term(quote, e.at, e.kind == "del", terms, lengths)
                or _in_term(source, e.src, e.kind == "ins", terms, lengths)
            )
        )
    ]


def _confusable(e: Edit, sets: frozenset[frozenset[str]]) -> bool:
    """A substitution between two characters of one confusable set (己→巳)."""
    return e.kind == "sub" and any({e.quote, e.source} <= s for s in sets)


def _classify(m: Match, quote: str, policy: Policy) -> Status | None:
    if m.distance == 0:
        return Status.VERIFIED
    if m.similarity(len(quote)) < policy.fuzzy_threshold:
        return None
    hits = _critical_edits(m, quote, policy)
    if not hits:
        return Status.FUZZY
    # 己→巳 and 已→己 are an OCR slip or a real change, and the string cannot
    # say which. Anything else critical (子→午, -不) is simply altered.
    if all(_confusable(e, policy.confusable) for e in hits):
        return Status.NEEDS_REVIEW
    return Status.ALTERED


_RANK = {Status.VERIFIED: 0, Status.FUZZY: 1, Status.NEEDS_REVIEW: 2, Status.ALTERED: 3}


def _best(matches: list[Match], quote: str, policy: Policy) -> tuple[Status, Match] | None:
    graded = [(s, m) for m in matches if (s := _classify(m, quote, policy))]
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
        found = _best(corpus.search(q), q, policy)
        return QuoteResult(quote, q, Status.UNKNOWN_SOURCE, found[1] if found else None, cited)

    scoped = corpus.search(q, books=cited)
    best = _best(scoped, q, policy)
    if best:
        return QuoteResult(quote, q, best[0], best[1], cited)

    everywhere = corpus.search(q) if cited else scoped
    elsewhere = _best(everywhere, q, policy)
    if elsewhere and elsewhere[0] in (Status.VERIFIED, Status.FUZZY):
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
