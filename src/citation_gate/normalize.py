"""Character folding, so a quote and its source compare equal across scripts.

Both sides of every comparison go through the same fold, so the fold only has
to be consistent, not linguistically correct: 後 and 后 both become 后 even
though they are different words in traditional text.
"""

from __future__ import annotations

import os
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


class ProfileError(ValueError):
    """A profile spec names neither a bundled profile nor a file."""


# Bundled profiles by name, and the data files each is made of.
CRITICAL_PROFILES = {"tcm": ("critical-tcm.txt",), "shushu": ("critical-shushu.txt",)}
TERM_PROFILES = {
    "tcm": ("terms-tcm-herbs.txt", "terms-tcm-formulas.txt"),
    "shushu": ("terms-shushu-hexagrams.txt", "terms-shushu-names.txt"),
}


def _profile(spec: str, bundled: dict[str, tuple[str, ...]]) -> list[str]:
    """Non-comment lines of a profile spec: a comma-separated list whose items
    are bundled profile names (`tcm`, `shushu`), `none` (contributes
    nothing) or paths to files in the same format. `tcm,shushu` is the union
    of both. A spec that is an existing file is read as one path, commas and all."""
    parts = [spec] if os.path.isfile(spec) else [p.strip() for p in spec.split(",") if p.strip()]
    texts = []
    for part in parts:
        if part == "none":
            continue
        if part in bundled:
            texts += [_read_data(name) for name in bundled[part]]
        elif os.path.isfile(part):
            with open(part, encoding="utf-8") as f:
                texts.append(f.read())
        else:
            names = ", ".join(sorted(bundled))
            raise ProfileError(f"unknown profile {part!r}: expected {names}, none, or a file")
    return [line for text in texts for line in text.splitlines() if line.strip() and not line.startswith("#")]


def load_critical(spec: str) -> frozenset[str]:
    """Critical characters for a profile spec such as `tcm`, `shushu`,
    `tcm,shushu`, `none` or a file path (see `_profile`). Lines are
    "class: characters"; the class is only documentation."""
    chars: set[str] = set()
    for line in _profile(spec, CRITICAL_PROFILES):
        chars.update(fold_text(line.split(":", 1)[-1]))
    return frozenset(chars)


def load_terms(spec: str) -> frozenset[str]:
    """Critical terms, in folded form, for a profile spec: `tcm` (herb and
    formula names), `shushu` (hexagram and other names), a combination such
    as `tcm,shushu`, `none`, or a path to a file with one term per line (only
    the first whitespace-separated field is read)."""
    lines = _profile(spec, TERM_PROFILES)
    terms = {fold_text(line.split()[0]) for line in lines}
    return frozenset(t for t in terms if t)


def load_confusable(spec: str = "default") -> frozenset[frozenset[str]]:
    """Sets of characters that OCR and copyists confuse with one another
    (己已巳, 戊戍戌 …), in folded form: `default` for the bundled
    `confusable.txt`, `none` for none, or a path to a file with one set per
    line (only the first whitespace-separated field is read)."""
    if spec == "none":
        return frozenset()
    if spec == "default":
        text = _read_data("confusable.txt")
    elif os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            text = f.read()
    else:
        raise ProfileError(f"unknown confusable sets {spec!r}: expected default, none, or a file")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    sets = (frozenset(fold_text(line.split()[0])) for line in lines)
    return frozenset(s for s in sets if len(s) >= 2)
