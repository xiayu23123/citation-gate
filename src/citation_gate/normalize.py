"""Character folding, so a quote and its source compare equal across scripts.

Both sides of every comparison go through the same fold, so the fold only has
to be consistent, not linguistically correct: 後 and 后 both become 后 even
though they are different words in traditional text.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources


def _read_data(name: str) -> str:
    return (resources.files("citation_gate") / "data" / name).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    table: dict[str, str] = {}
    for name in ("TSCharacters.txt", "variants.tsv"):
        for line in _read_data(name).splitlines():
            if not line or line.startswith("#"):
                continue
            src, dst = line.split("\t")[:2]
            table[src] = dst.split()[0]
    return table


def _keep(ch: str) -> bool:
    # Letters and digits only: punctuation, spacing and symbols carry no
    # citation content, and classical texts are punctuated by later editors.
    return unicodedata.category(ch)[0] in "LN"


@dataclass(frozen=True)
class Folded:
    text: str
    offsets: tuple[int, ...]  # offsets[i] is the index in the original of text[i]

    def span(self, original: str, start: int, end: int) -> str:
        """The original text behind folded[start:end]."""
        if start >= end:
            return ""
        return original[self.offsets[start] : self.offsets[end - 1] + 1]


def fold(s: str) -> Folded:
    table = _table()
    out: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKC", ch):
            if not _keep(c):
                continue
            for f in table.get(c, c).lower():
                out.append(f)
                offsets.append(i)
    return Folded("".join(out), tuple(offsets))


def fold_text(s: str) -> str:
    return fold(s).text


def simplify(s: str) -> str:
    """Character-level traditional-to-simplified rendering that keeps punctuation."""
    table = _table()
    return "".join(table.get(c, c) for c in s)


def load_critical(spec: str) -> frozenset[str]:
    """`tcm` for the bundled profile, `none` for no critical characters,
    anything else is a path to a file in the same format."""
    if spec == "none":
        return frozenset()
    if spec == "tcm":
        text = _read_data("critical-tcm.txt")
    else:
        with open(spec, encoding="utf-8") as f:
            text = f.read()
    chars: set[str] = set()
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        chars.update(fold_text(line.split(":", 1)[-1]))
    return frozenset(chars)
