"""Find quoted passages, and the works they are attributed to, in an answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import fold_text

QUOTE_MARKS = {"「": "」", "『": "』", "“": "”", '"': '"'}
SENTENCE_END = "。！？!?\n"

_TITLE = re.compile(r"《([^《》]{1,60})》")
# An attribution right after the quote: 「…」（《伤寒论》） or “…”——《金匮要略》
_TRAILING = re.compile(r"^\s*(?:[（(]|[—–-]{1,2})\s*《([^《》]{1,60})》")


@dataclass(frozen=True)
class Quote:
    text: str  # between the marks
    start: int  # span in the answer, marks included
    end: int
    attribution: str | None  # inside 《》, e.g. "素问·上古天真论"


def _closing(answer: str, i: int) -> int:
    """Index of the mark closing the quote opened at i, or -1."""
    opener = answer[i]
    closer = QUOTE_MARKS[opener]
    if opener == closer:
        return answer.find(closer, i + 1)
    depth = 0
    for j in range(i + 1, len(answer)):
        if answer[j] == opener:
            depth += 1
        elif answer[j] == closer:
            if depth == 0:
                return j
            depth -= 1
    return -1


def _attribution(answer: str, start: int, end: int, floor: int) -> str | None:
    """The last 《title》 earlier in the same sentence, else one just after."""
    head = answer[floor:start]
    cut = max(head.rfind(c) for c in SENTENCE_END)
    titles = _TITLE.findall(head[cut + 1 :])
    if titles:
        return titles[-1]
    m = _TRAILING.match(answer[end:])
    return m.group(1) if m else None


def extract_quotes(answer: str) -> list[Quote]:
    """Outermost quotations, in order. Nested quotes are part of their parent."""
    quotes: list[Quote] = []
    i, floor = 0, 0
    while i < len(answer):
        if answer[i] in QUOTE_MARKS:
            j = _closing(answer, i)
            if j > i:
                attr = _attribution(answer, i, j + 1, floor)
                quotes.append(Quote(answer[i + 1 : j], i, j + 1, attr))
                floor = i = j + 1
                continue
        i += 1
    return quotes


def uncited_sentences(answer: str, quotes: list[Quote], min_chars: int = 8) -> list[str]:
    """Sentences that neither quote nor name a source.

    Sentence breaks inside quotations do not count, except one that ends the
    quotation (「……。」). Questions and sentences shorter than `min_chars`
    folded characters are ignored.
    """
    inside = [False] * len(answer)
    for q in quotes:
        for k in range(q.start, q.end):
            inside[k] = True
    spans = [(q.start, q.end) for q in quotes]
    closing = {q.end - 1 for q in quotes if q.text[-1:] and q.text[-1] in SENTENCE_END}

    out = []
    s = 0
    for k, ch in enumerate(answer + "\n"):
        if k < len(answer) and k not in closing and (inside[k] or ch not in SENTENCE_END):
            continue
        sentence = answer[s : k + 1].strip()
        cited = "《" in sentence or any(a < k + 1 and b > s for a, b in spans)
        if sentence and not cited and ch not in "？?" and len(fold_text(sentence)) >= min_chars:
            out.append(sentence)
        s = k + 1
    return out
