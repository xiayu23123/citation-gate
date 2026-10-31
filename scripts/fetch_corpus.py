"""Fetch the demo corpus from Chinese Wikisource and write it as plain text.

The source texts are public domain (Han dynasty and earlier). We pin each
page to the revision we fetched so the benchmark is reproducible; see
corpus/SOURCES.md, which this script regenerates.

Usage: python scripts/fetch_corpus.py [--out corpus]
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://zh.wikisource.org/w/api.php"
UA = {"User-Agent": "citation-gate/0.1 (https://github.com/xiayu23123/citation-gate)"}

# (book id, display title, wikisource page)
PAGES = [
    ("suwen-1", "素問", "黃帝內經/素問第一卷"),
    ("suwen-2", "素問", "黃帝內經/素問第二卷"),
    ("shanghan", "傷寒論", "傷寒論"),
    ("jingui", "金匱要略", "金匱要略"),
]


def fetch(page: str) -> tuple[str, int]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content|ids",
        "rvslots": "main",
        "titles": page,
        "redirects": 1,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = json.load(r)
    (p,) = data["query"]["pages"].values()
    rev = p["revisions"][0]
    return rev["slots"]["main"]["*"], rev["revid"]


def strip_templates(s: str) -> str:
    """Remove {{...}} templates, innermost first, so nesting is handled."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s


def clean(wikitext: str) -> str:
    s = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    s = re.sub(r"<ref[^>]*?/>|<ref.*?</ref>", "", s, flags=re.S)
    s = strip_templates(s)
    s = re.sub(r"-\{(.*?)\}-", r"\1", s)  # language-conversion guards
    s = re.sub(r"\[\[(?:File|Image|文件|檔案|Category|分類):[^\]]*\]\]", "", s)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"'{2,}", "", s)

    out = []
    for line in s.splitlines():
        line = line.strip().strip("　").strip()
        m = re.fullmatch(r"=+\s*(.*?)\s*=+", line)
        if m:
            out.append(f"\n## {m.group(1)}\n")
        elif line and not line.startswith(("*", "{|", "|", "!", "__")):
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = [
        "# Sources\n",
        "Plain-text extracts of public-domain texts from Chinese Wikisource",
        "(zh.wikisource.org), pinned to the revisions below. Markup, notes and",
        "templates are stripped by `scripts/fetch_corpus.py`. Wikisource's own",
        "editorial layer is CC BY-SA 4.0.\n",
        "| file | book | page | revision |",
        "|---|---|---|---|",
    ]
    for book_id, title, page in PAGES:
        text, revid = fetch(page)
        body = clean(text)
        path = out / f"{book_id}.txt"
        path.write_text(f"# {title}\n\n{body}", encoding="utf-8")
        link = f"https://zh.wikisource.org/w/index.php?oldid={revid}"
        sources.append(f"| `{path.name}` | {title} | {page} | [{revid}]({link}) |")
        print(f"{path}: {len(body):,} chars (rev {revid})")
    (out / "SOURCES.md").write_text("\n".join(sources) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
