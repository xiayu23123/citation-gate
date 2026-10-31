"""Extract herb and formula names from the corpus.

The bundled critical-term profile (src/citation_gate/data/terms-tcm-*.txt)
is built here from the corpus alone. It never looks at the name lists in
scripts/build_bench.py, so the benchmark's swaps are not scored against a
copy of themselves.

Herbs. Every formula in 伤寒论 and 金匮要略 lists its ingredients on the line
just before its preparation line (上五味，以水七升…… or 右四味……):

    ::桂枝三兩（去皮）　芍藥三兩　甘草二兩（炙）　生薑三兩（切）　大棗十二枚（擘）
    ::上五味，㕮咀三味，以水七升，微火煮取三升……

Bracketed notes are dropped, the line is split at spaces and 、，；, and each
name is what precedes its dose (三兩, 各十八銖, 半升, 等分, 如雞子大), or the
whole piece if it has none. Processing notes (去皮, 炮, 熬) are cut off.

A name is kept if it appears in at least two recipes: that drops one-off
parsing accidents, and also real herbs used only once. Those single-recipe
names are written to scripts/terms-single-recipe.txt, and the benchmark swaps
among them to see what the gate does with herbs its list does not know.

Formulas. A formula's name heads its recipe, alone on its line or before a
colon, sometimes after the book it was taken from:

    :;桂枝湯方
    《千金》三黃湯：治中風手足拘急……

Every name ending in 湯, 散 or 丸 found in such a heading is kept; a heading
is not a parsing guess, so there is no frequency cut-off. Formulas without a
heading in this corpus (吳茱萸湯 has none) are missing.

Usage: python scripts/extract_terms.py [--corpus corpus/tcm]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from citation_gate.normalize import fold_text

NUM = "一二三四五六七八九十百千半"
UNIT = "兩斤銖株枚箇個升合分錢斗尺把具莖片丸條握撮匕"
QTY = rf"(?:大者)?(?:[{NUM}]+[{UNIT}]+)+[{NUM}]*|如[^{NUM}]{{1,4}}大(?:[{NUM}]+[{UNIT}]+)*"
DOSE = re.compile(rf"(?P<name>[㐀-鿿]+?)(?:各?(?:{QTY})|各?等分)")
PREP = re.compile(r"^[上右](?P<n>[一二三四五六七八九十]+)?(?:味|各)")
NOTE = re.compile(r"（[^（）]*）|\([^()]*\)")
SPLIT = re.compile(r"[\s　，、；。]+")
# A name ends where a processing or size note starts (牡丹去心, 甘草如中指節);
# a piece that starts with one is only a note (炮, 去皮, 熬焦).
NOTE_START = re.compile(r"[去炮熬炙燒研切擘洗碎炒搗如].*$")
# Instructions and editorial remarks that sit among the ingredients
# (取燒作灰, 一方三兩, 八月採), in folded form.
NOT_NAME = re.compile(r"[以煮煎取服饮用加减和为方法有采即者]")
CN_NUM = {c: i for i, c in enumerate("一二三四五六七八九十", 1)} | {
    f"十{c}": 10 + i for i, c in enumerate("一二三四五六七八九", 1)
}
MIN_RECIPES = 2
# A recipe heading: optional source (《千金》) or leading clause (治傷寒令愈不復，),
# the name, optional 方, (附加減法) or *, then the end of the line or a colon.
HEADING = re.compile(
    r"^[:;]*(?:《[^》]{1,8}》)?(?:[^，。：；、《》（）\s]*[，、])?"
    r"(?P<name>[㐀-鿿]{1,20}?[湯散丸])方?(?:（[^）]*）)?\*?(?:：.*)?$"
)

ROOT = Path(__file__).resolve().parent.parent
HERBS_OUT = ROOT / "src/citation_gate/data/terms-tcm-herbs.txt"
FORMULAS_OUT = ROOT / "src/citation_gate/data/terms-tcm-formulas.txt"
SINGLE = ROOT / "scripts/terms-single-recipe.txt"


def recipes(corpus: Path) -> list[tuple[str, int, str, int | None]]:
    """(file, line number, ingredient line, number of herbs the recipe states)."""
    out = []
    for path in sorted(corpus.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            prep = PREP.match(line.lstrip(":;"))
            if not prep:
                continue
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j < 0 or lines[j].startswith("#") or PREP.match(lines[j].lstrip(":;")):
                continue
            stated = CN_NUM.get(prep["n"]) if prep["n"] else None
            out.append((path.name, j + 1, lines[j].lstrip(":;"), stated))
    return out


def names(line: str) -> list[str]:
    found = []
    for piece in SPLIT.split(NOTE.sub("", line)):
        if re.fullmatch(QTY, piece):
            continue
        for name in [m["name"] for m in DOSE.finditer(piece)] or [piece]:
            if NOTE_START.match(name) or NOT_NAME.search(fold_text(name)):
                continue
            name = fold_text(NOTE_START.sub("", name))
            if len(name) >= 2:
                found.append(name)
    return found


def formulas(corpus: Path) -> Counter[str]:
    """Formula names in recipe headings, with the number of headings each has."""
    counts: Counter[str] = Counter()
    for path in sorted(corpus.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = None if line.startswith("#") else HEADING.match(line.strip())
            if m:
                counts[fold_text(m["name"])] += 1
    return counts


def write(path: Path, head: list[str], rows: list[str]) -> None:
    path.write_text("\n".join(head + rows) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/tcm")
    args = ap.parse_args()

    counts: Counter[str] = Counter()
    first_seen: dict[str, str] = {}
    found = recipes(Path(args.corpus))
    agree = 0
    for file, n, line, stated in found:
        here = names(line)
        agree += stated == len(here)
        for name in set(here):
            counts[name] += 1
            first_seen.setdefault(name, f"{file}:{n}")
    heads = formulas(Path(args.corpus))
    for name in [*counts, *heads]:
        assert fold_text(name) == name, name  # the gate folds terms again on load

    kept = sorted((n for n, c in counts.items() if c >= MIN_RECIPES), key=lambda n: (-counts[n], n))
    single = sorted(n for n, c in counts.items() if c < MIN_RECIPES)
    about = [
        "# Critical terms for traditional Chinese medicine texts, in folded form.",
        "# A fuzzy match is rejected if any edit falls inside one of these names,",
        "# in the quote or in the source. Generated by scripts/extract_terms.py",
        "# from the bundled corpus; do not edit by hand. Only the first field of a",
        "# line is read.",
    ]
    write(
        HERBS_OUT,
        about
        + [
            f"# Herb names found in the ingredient lines of at least {MIN_RECIPES} of its {len(found)}",
            '# recipes; lines are "term recipes".',
        ],
        [f"{n} {counts[n]}" for n in kept],
    )
    write(
        FORMULAS_OUT,
        about + ['# Formula names from its recipe headings; lines are "term headings".'],
        [f"{n} {c}" for n, c in sorted(heads.items(), key=lambda nc: (-nc[1], nc[0]))],
    )
    write(
        SINGLE,
        [
            "# Herb names found in only one recipe of the corpus, so not in",
            "# src/citation_gate/data/terms-tcm-herbs.txt. Generated by scripts/extract_terms.py;",
            "# scripts/build_bench.py swaps among these to test herbs the list does not know.",
            '# Lines are "name file:line".',
        ],
        [f"{n} {first_seen[n]}" for n in single],
    )
    print(
        f"{len(found)} recipes ({agree} parse to the number of herbs they state), {len(counts)} herb names: "
        f"{len(kept)} kept, {len(single)} single-recipe; {len(heads)} formula names from headings"
    )


if __name__ == "__main__":
    main()
