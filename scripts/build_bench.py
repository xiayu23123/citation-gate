"""Build the labelled benchmark from the corpus, deterministically.

Every item is one answer containing one quote under test. Labels come from how
the item was built, never from running the gate. Construction checks use plain
substring tests on folded text, not the gate's matcher.

  genuine        verbatim passage, simplified or traditional         should pass
  noise          genuine + 1-2 OCR-style slips on non-critical chars should pass
  altered        genuine + one critical change (肾→肝, 寒→热, -不)  should fail
  herb_swap      genuine with one herb or formula name swapped       should fail
  misattributed  genuine passage credited to another book in corpus  should fail
  spliced        two real fragments from different places, joined   should fail
  fabricated     classical-looking text from a character Markov model should fail
  paraphrase     hand-written modern paraphrase presented as a quote should fail
  herb_swap_long     herb_swap in a quote of 30-60 characters       should fail
  herb_swap_offlist  herb_swap among herbs the herb list lacks       should fail
  formula_swap       one formula name swapped, 30-60 characters      should fail
  formula_swap_bare  formula_swap among names with no listed herb    should fail
                     and no critical character in them

herb_swap and formula_swap draw from HERBS and FORMULAS, hand-written lists.
The gate's critical terms are extracted from the corpus by
scripts/extract_terms.py, which never reads either; herb_swap_offlist swaps
only herbs that extraction left out of the herb list, so it shows what the
gate does with names its list lacks.

Usage: python scripts/build_bench.py [--corpus corpus/tcm] [--out bench]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from citation_gate import Corpus
from citation_gate.normalize import fold_text, load_critical, load_terms, simplify
from extract_terms import HERBS_OUT, SINGLE

SPLITS = {"dev": 1, "test": 2}
COUNTS = {
    "genuine": 60,
    "noise": 50,
    "altered": 50,
    "herb_swap": 40,
    "misattributed": 30,
    "spliced": 30,
    "fabricated": 30,
}
# Added after v0.1.0. They are built after the paraphrases, from the same
# random stream, so that every earlier item stays byte-for-byte the same.
LATER = {"herb_swap_long": 40, "herb_swap_offlist": 40, "formula_swap": 40, "formula_swap_bare": 40}

TITLES = {"素問": ["素问", "黄帝内经·素问", "内经", "黄帝内经"], "傷寒論": ["伤寒论"], "金匱要略": ["金匮要略", "金匮"]}
WITH_CHAPTER = {"素問": "素问", "傷寒論": "伤寒论", "金匱要略": "金匮要略"}

ATTRIBUTED = [
    "《{a}》云：「{q}」",
    "《{a}》中说：“{q}”这句话很关键。",
    "原文是“{q}”（《{a}》）。",
    "据《{a}》记载：「{q}」，可见古人早有论述。",
    "“{q}”——《{a}》",
]
UNATTRIBUTED = ["古人说：“{q}”。", "经文有言：「{q}」"]
BEFORE = ["", "", "这个问题可以从经典中找到答案。", "我们先看原文。"]
AFTER = ["", "", "这说明了顺应自然的重要性。", "后世医家多有发挥。"]

# OCR-style confusions between visually similar characters (simplified forms).
CONFUSIONS = {
    "曰": "日", "日": "曰", "己": "已", "已": "己", "人": "入", "入": "人", "刀": "力", "力": "刀",
    "天": "夭", "苦": "若", "若": "苦", "白": "自", "自": "白", "目": "日", "甲": "申", "申": "甲",
    "候": "侯", "侯": "候", "间": "问", "问": "间", "今": "令", "令": "今", "见": "贝", "旦": "且",
    "且": "旦", "辨": "辩", "辩": "辨", "王": "玉", "玉": "王", "思": "恩", "体": "休", "折": "析",
}
HERBS = (
    "桂枝 麻黄 葛根 柴胡 黄芩 黄连 附子 干姜 人参 甘草 芍药 茯苓 白术 半夏 大枣 生姜 石膏 知母 "
    "杏仁 厚朴 枳实 细辛 当归 大黄 芒硝 栀子 阿胶 猪苓 泽泻 桃仁 黄柏 防己"
).split()
# Well-known formulas under their usual modern names, some of which this
# corpus writes differently (麻黄细辛附子汤) or not at all (苓桂术甘汤).
FORMULAS = (
    "桂枝汤 麻黄汤 葛根汤 大青龙汤 小青龙汤 小柴胡汤 大柴胡汤 白虎汤 白虎加人参汤 大承气汤 小承气汤 "
    "调胃承气汤 桃核承气汤 四逆汤 通脉四逆汤 四逆散 理中丸 真武汤 附子汤 五苓散 猪苓汤 小建中汤 炙甘草汤 "
    "半夏泻心汤 甘草泻心汤 生姜泻心汤 大陷胸汤 小陷胸汤 抵当汤 栀子豉汤 茵陈蒿汤 吴茱萸汤 黄连阿胶汤 "
    "当归四逆汤 乌梅丸 白头翁汤 麻黄附子细辛汤 苓桂术甘汤 肾气丸 当归芍药散 桂枝茯苓丸 酸枣仁汤 "
    "甘麦大枣汤 大黄牡丹皮汤 麻子仁丸 十枣汤 竹叶石膏汤"
).split()
ORGANS = "心肝脾肺肾"
NUMERALS = "一二三四五六七八九"
POLAR = dict(zip("寒热虚实阴阳表里上下左右大小生死男女补泻", "热寒实虚阳阴里表下上右左小大死生女男泻补"))
PUNCT = "，。；：！？、"
STRIP_MARKS = str.maketrans("", "", "「」『』“”\"")
CLAUSE_NO = re.compile(r"^[〇一二三四五六七八九十百]+、")
# Prose only: no dosage tables, bracketed notes or layout spacing.
PROSE = re.compile(rf"[㐀-鿿\U00020000-\U0003134F{PUNCT}]+")


def bare_names(names: list[str], herbs: frozenset[str], critical: frozenset[str]) -> list[str]:
    """Names with no listed herb and no critical character in them, which
    only formula terms can protect (白虎汤, 真武汤)."""
    return [f for f in names if not set(f) & critical and not any(h in f for h in herbs)]


class Builder:
    def __init__(self, corpus: Corpus, seed: int):
        self.corpus = corpus
        self.rng = random.Random(seed)
        self.critical = load_critical("tcm")
        # Herbs only: formula names were added to the profile after
        # herb_swap_offlist was built, and its items must not change.
        self.herbs = load_terms(str(HERBS_OUT))
        # Herbs that extraction saw in only one recipe, so not in the herb list.
        # Drop any the gate could still catch another way: those with a
        # critical character, and those inside a listed herb or containing one.
        lines = SINGLE.read_text(encoding="utf-8").splitlines()
        single = [line.split()[0] for line in lines if line and not line.startswith("#")]
        self.offlist = [
            h
            for h in single
            if 2 <= len(h) <= 4 and not set(h) & self.critical and not any(t in h or h in t for t in self.herbs)
        ]
        self.bare = bare_names(FORMULAS, self.herbs, self.critical)
        self.confusions = {
            a: b for a, b in CONFUSIONS.items() if not (set(fold_text(a + b)) & self.critical)
        }
        self.folded_all = "\x00".join(d.folded.text for d in corpus.docs)
        self.lines = []  # (doc, line number, chapter, clauses)
        for d in corpus.docs:
            chapter = None
            for n, raw in enumerate(d.text.splitlines(), 1):
                if raw.startswith("#"):
                    chapter = raw.lstrip("#").strip()
                    continue
                line = CLAUSE_NO.sub("", raw.translate(STRIP_MARKS))
                clauses = re.findall(rf"[^{PUNCT}]+[{PUNCT}]?", line)
                if PROSE.fullmatch(line) and len(fold_text(line)) >= 8:
                    self.lines.append((d, n, chapter, clauses))
        self.weights = [sum(len(c) for c in cl) for *_, cl in self.lines]
        self.markov = self._markov()

    # ---- sampling -------------------------------------------------------

    def span(self, lo: int, hi: int, book: str | None = None):
        """A run of whole clauses with lo..hi folded characters."""
        while True:
            d, n, chapter, clauses = self.rng.choices(self.lines, self.weights)[0]
            if book and d.book != book:
                continue
            i = self.rng.randrange(len(clauses))
            text = ""
            for c in clauses[i:]:
                text += c
                if len(fold_text(text)) >= lo:
                    break
            text = text.rstrip("，、；：")
            if lo <= len(fold_text(text)) <= hi:
                return d, n, chapter, text

    def attribution(self, book: str, chapter: str | None) -> str:
        if chapter and not chapter.startswith("卷") and "序" not in chapter and self.rng.random() < 0.4:
            ch = re.sub(r"第[一二三四五六七八九十百]+$", "", simplify(chapter)).removesuffix("篇")
            return f"{WITH_CHAPTER[book]}·{ch}"
        return self.rng.choice(TITLES[book])

    def answer(self, quote: str, attribution: str | None) -> str:
        template = self.rng.choice(ATTRIBUTED if attribution else UNATTRIBUTED)
        body = template.format(q=quote, a=attribution)
        return self.rng.choice(BEFORE) + body + self.rng.choice(AFTER)

    def in_corpus(self, s: str, docs=None) -> bool:
        f = fold_text(s)
        if docs is None:
            return f in self.folded_all
        return any(f in d.folded.text for d in docs)

    def longest_fragment(self, s: str) -> int:
        f, best = fold_text(s), 0
        for i in range(len(f)):
            k = best + 1
            while i + k <= len(f) and f[i : i + k] in self.folded_all:
                best, k = k, k + 1
        return best

    # ---- categories ------------------------------------------------------

    def genuine(self):
        d, n, chapter, text = self.span(8, 30)
        quote = text if self.rng.random() < 0.3 else simplify(text)
        attr = self.attribution(d.book, chapter) if self.rng.random() < 0.8 else None
        return quote, attr, f"{d.path}:{n}"

    def noise(self):
        d, n, chapter, text = self.span(8, 30)
        quote = simplify(text)
        edits = 1 if len(fold_text(quote)) < 20 or self.rng.random() < 0.5 else 2
        for _ in range(edits):
            quote = self._slip(quote)
        attr = self.attribution(d.book, chapter) if self.rng.random() < 0.8 else None
        return quote, attr, f"{d.path}:{n}"

    def _slip(self, s: str) -> str:
        spots = [i for i, c in enumerate(s) if c not in PUNCT and fold_text(c) and fold_text(c) not in self.critical]
        confusable = [i for i in spots if s[i] in self.confusions]
        r = self.rng.random()
        if confusable and r < 0.6:
            i = self.rng.choice(confusable)
            return s[:i] + self.confusions[s[i]] + s[i + 1 :]
        i = self.rng.choice(spots[1:-1] or spots)
        if r < 0.8:
            return s[:i] + s[i + 1 :]  # dropped character
        return s[:i] + s[i] + s[i:]  # doubled character

    def altered(self):
        while True:
            d, n, chapter, text = self.span(8, 30)
            s = simplify(text)
            options = []
            for i, c in enumerate(s):
                if c in ORGANS:
                    options.append((i, ORGANS[(ORGANS.index(c) + 1) % len(ORGANS)]))
                elif c in NUMERALS:
                    options.append((i, NUMERALS[(NUMERALS.index(c) + 1) % len(NUMERALS)]))
                elif c in POLAR:
                    options.append((i, POLAR[c]))
                elif c == "不":
                    options.append((i, ""))
            if not options:
                continue
            i, new = self.rng.choice(options)
            quote = s[:i] + new + s[i + 1 :]
            if not self.in_corpus(quote):
                attr = self.attribution(d.book, chapter) if self.rng.random() < 0.8 else None
                return quote, attr, f"{d.path}:{n}"

    def swap(self, names: list[str], lo: int, hi: int):
        """One name in a real passage replaced by another from `names`."""
        while True:
            d, n, chapter, text = self.span(lo, hi)
            s = simplify(text)
            # Not 四逆汤 where the passage has it only inside 当归四逆汤.
            present = [a for a in names if a in s and not any(a != b and a in b and b in s for b in names)]
            if not present:
                continue
            old = self.rng.choice(present)
            new = self.rng.choice([h for h in names if h not in s])
            quote = s.replace(old, new, 1)
            if not self.in_corpus(quote):
                attr = self.attribution(d.book, chapter) if self.rng.random() < 0.8 else None
                return quote, attr, f"{d.path}:{n}"

    def herb_swap(self):
        return self.swap(HERBS, 8, 30)

    def herb_swap_long(self):
        return self.swap(HERBS, 30, 60)

    def formula_swap(self):
        return self.swap(FORMULAS, 30, 60)

    def formula_swap_bare(self):
        return self.swap(self.bare, 30, 60)

    def near_term(self, s: str, i: int, k: int) -> bool:
        """Whether a listed herb occurs in fold(s) overlapping the folded s[i:i+k]."""
        f = fold_text(s)
        lo = len(fold_text(s[:i]))
        hi = lo + len(fold_text(s[i : i + k]))
        return any(f.find(t, max(0, lo - len(t) + 1), hi + len(t) - 1) >= 0 for t in self.herbs)

    def herb_swap_offlist(self):
        while True:
            d, n, chapter, text = self.span(8, 30)
            s = simplify(text)
            present = [h for h in self.offlist if h in s]
            if not present:
                continue
            old = self.rng.choice(present)
            new = self.rng.choice([h for h in self.offlist if h not in s])
            i = s.index(old)
            quote = s[:i] + new + s[i + len(old) :]
            if self.near_term(s, i, len(old)) or self.near_term(quote, i, len(new)) or self.in_corpus(quote):
                continue
            attr = self.attribution(d.book, chapter) if self.rng.random() < 0.8 else None
            return quote, attr, f"{d.path}:{n}"

    def misattributed(self):
        while True:
            d, n, chapter, text = self.span(8, 30)
            others = [b for b in self.corpus.books if b != d.book]
            book = self.rng.choice(others)
            if not self.in_corpus(text, self.corpus.books[book]):
                return simplify(text), self.attribution(book, None), f"{d.path}:{n}"

    def spliced(self):
        while True:
            a = self.span(5, 12)
            b = self.span(5, 12)
            if a[0] is b[0] and abs(a[1] - b[1]) < 5:
                continue
            quote = simplify(a[3].rstrip(PUNCT) + "，" + b[3])
            if not self.in_corpus(quote):
                attr = self.attribution(a[0].book, a[2]) if self.rng.random() < 0.8 else None
                return quote, attr, f"{a[0].path}:{a[1]} + {b[0].path}:{b[1]}"

    def _markov(self):
        model = defaultdict(list)
        for *_, clauses in self.lines:
            s = simplify("".join(clauses))
            for i in range(len(s) - 2):
                model[s[i : i + 2]].append(s[i + 2])
        return model

    def fabricated(self):
        while True:
            _, _, _, clauses = self.rng.choices(self.lines, self.weights)[0]
            s = simplify(self.rng.choice(clauses))[:2]
            target = self.rng.randint(10, 24)
            while len(fold_text(s)) < 1.5 * target:
                nxt = self.markov.get(s[-2:])
                if not nxt:
                    break
                s += self.rng.choice(nxt)
                if len(fold_text(s)) >= target and s[-1] in PUNCT:
                    break
            s = s.rstrip(PUNCT)
            f = len(fold_text(s))
            if f >= 10 and self.longest_fragment(s) <= f // 2:
                book = self.rng.choice(list(self.corpus.books))
                attr = self.attribution(book, None) if self.rng.random() < 0.8 else None
                return s, attr, None


def build(corpus: Corpus, split: str, seed: int, paraphrases: list[dict]) -> list[dict]:
    b = Builder(corpus, seed)
    items = []

    def add(category: str, quote: str, attr: str | None, source: str | None) -> None:
        items.append(
            {
                "category": category,
                "should_pass": category in ("genuine", "noise"),
                "answer": b.answer(quote, attr),
                "quote": quote,
                "attribution": attr,
                "source": source,
            }
        )

    def generate(counts: dict[str, int]) -> None:
        for category, count in counts.items():
            make = getattr(b, category)
            for _ in range(count):
                add(category, *make())

    generate(COUNTS)
    for p in paraphrases[seed - 1 :: 2]:
        add("paraphrase", p["quote"], p["attribution"], p["source"])
    generate(LATER)
    for i, item in enumerate(items, 1):
        item["id"] = f"{split}-{i:03d}"
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/tcm")
    ap.add_argument("--out", default="bench")
    args = ap.parse_args()
    corpus = Corpus.load(args.corpus)
    out = Path(args.out)
    paraphrases = json.loads((out / "paraphrases.json").read_text(encoding="utf-8"))
    for split, seed in SPLITS.items():
        items = build(corpus, split, seed, paraphrases)
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for item in items:
                f.write(json.dumps({"id": item.pop("id"), **item}, ensure_ascii=False) + "\n")
        print(f"{path}: {len(items)} items")


if __name__ == "__main__":
    main()
