"""Semi-global alignment: the whole quote against any stretch of the source."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edit:
    """One difference between a quote and its source.

    kind is "sub" (quote has a different character), "ins" (quote has an extra
    character) or "del" (quote is missing a source character). `at` is the
    position in the folded quote.
    """

    kind: str
    at: int
    quote: str | None
    source: str | None

    def chars(self) -> set[str]:
        return {c for c in (self.quote, self.source) if c}

    def __str__(self) -> str:
        if self.kind == "sub":
            return f"{self.source}→{self.quote}"
        if self.kind == "ins":
            return f"+{self.quote}"
        return f"-{self.source}"


@dataclass(frozen=True)
class Alignment:
    distance: int
    start: int  # aligned span within the window
    end: int
    edits: tuple[Edit, ...]


def align(quote: str, window: str) -> Alignment:
    """Edit distance from `quote` to its best-matching substring of `window`.

    Leading and trailing window characters are free; every quote character
    must be accounted for. This is Sellers' variant of Levenshtein.
    """
    m, n = len(quote), len(window)
    rows = [[0] * (n + 1)]
    for i in range(1, m + 1):
        prev = rows[-1]
        cur = [i] + [0] * n
        q = quote[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j - 1] + (window[j - 1] != q),
                prev[j] + 1,
                cur[j - 1] + 1,
            )
        rows.append(cur)

    # Among equally good alignments, prefer the one that covers more source and
    # explains differences as source characters the quote dropped: a dropped
    # 不 should be reported as -不, not hidden as an extra quote character.
    last = rows[m]
    best = min(last)
    end = max(j for j in range(n + 1) if last[j] == best)
    edits: list[Edit] = []
    i, j = m, end
    while i > 0:
        here = rows[i][j]
        same = j > 0 and window[j - 1] == quote[i - 1]
        if same and here == rows[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif j > 0 and here == rows[i][j - 1] + 1:
            edits.append(Edit("del", i, None, window[j - 1]))
            j -= 1
        elif j > 0 and here == rows[i - 1][j - 1] + 1:
            edits.append(Edit("sub", i - 1, quote[i - 1], window[j - 1]))
            i, j = i - 1, j - 1
        else:
            edits.append(Edit("ins", i - 1, quote[i - 1], None))
            i -= 1
    edits.reverse()
    return Alignment(last[end], j, end, tuple(edits))
