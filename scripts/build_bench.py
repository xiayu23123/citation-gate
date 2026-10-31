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

Usage: python scripts/build_bench.py [--corpus corpus] [--out bench]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from citation_gate import Corpus
from citation_gate.normalize import fold_text, load_critical, simplify

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
ORGANS = "心肝脾肺肾"
NUMERALS = "一二三四五六七八九"
POLAR = dict(zip("寒热虚实阴阳表里上下左右大小生死男女补泻", "热寒实虚阳阴里表下上右左小大死生女男泻补"))
PUNCT = "，。；：！？、"
STRIP_MARKS = str.maketrans("", "", "「」『』“”\"")
CLAUSE_NO = re.compile(r"^[〇一二三四五六七八九十百]+、")
# Prose only: no dosage tables, bracketed notes or layout spacing.
PROSE = re.compile(rf"[㐀-鿿\U00020000-\U0003134F{PUNCT}]+")


class Builder:
    def __init__(self, corpus: Corpus, seed: int):
        self.corpus = corpus
        self.rng = random.Random(seed)
        self.critical = load_critical("tcm")
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

    def herb_swap(self):
        while True:
            d, n, chapter, text = self.span(8, 30)
            s = simplify(text)
            present = [h for h in HERBS if h in s]
            if not present:
                continue
            old = self.rng.choice(present)
            new = self.rng.choice([h for h in HERBS if h not in s])
            quote = s.replace(old, new, 1)
            if not self.in_corpus(quote):
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
    for category, count in COUNTS.items():
        make = getattr(b, category)
        for _ in range(count):
            quote, attr, source = make()
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
    for p in paraphrases[seed - 1 :: 2]:
        quote, attr = p["quote"], p["attribution"]
        items.append(
            {
                "category": "paraphrase",
                "should_pass": False,
                "answer": b.answer(quote, attr),
                "quote": quote,
                "attribution": attr,
                "source": p["source"],
            }
        )
    for i, item in enumerate(items, 1):
        item["id"] = f"{split}-{i:03d}"
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
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
