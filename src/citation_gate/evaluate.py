"""Measure the gate itself against a labelled benchmark.

Each benchmark line is a JSON object with an `answer`, the `quote` inside it
that is under test, its `category`, and `should_pass`. A target quote the
extractor misses is scored as passed: the gate failed open.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import Corpus
from .gate import Policy, check_answer

MISSING = "missing"


def load_bench(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(corpus: Corpus, items: list[dict], policy: Policy) -> dict:
    by_category: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    errors = []
    tp = fn = fp = tn = 0
    for item in items:
        report = check_answer(corpus, item["answer"], policy, item["id"])
        hit = next((r for r in report.results if r.quote.text == item["quote"]), None)
        status = str(hit.status) if hit else MISSING
        flagged = bool(hit) and policy.fails(hit.status)
        cat = item["category"]
        by_category[cat]["n"] += 1
        by_category[cat]["flagged"] += flagged
        confusion[cat][status] += 1
        if item["should_pass"]:
            fp += flagged
            tn += not flagged
        else:
            tp += flagged
            fn += not flagged
        if flagged == item["should_pass"]:
            errors.append({"id": item["id"], "category": cat, "status": status, "quote": item["quote"]})

    recall = tp / (tp + fn) if tp + fn else 0.0
    false_alarm = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(items),
        "recall": recall,  # bad quotes caught
        "false_alarm": false_alarm,  # good quotes rejected
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": (recall + 1 - false_alarm) / 2,
        "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "by_category": {c: dict(v) for c, v in sorted(by_category.items())},
        "confusion": {c: dict(v) for c, v in sorted(confusion.items())},
        "errors": errors,
    }


def to_markdown(result: dict, title: str = "Benchmark") -> str:
    lines = [
        f"### {title}",
        "",
        f"n={result['n']} · caught {result['recall']:.1%} of bad quotes · "
        f"rejected {result['false_alarm']:.1%} of good quotes · F1 {result['f1']:.3f}",
        "",
        "| category | n | flagged | rate | statuses |",
        "|---|---|---|---|---|",
    ]
    for cat, c in result["by_category"].items():
        statuses = ", ".join(f"{s} {k}" for s, k in sorted(result["confusion"][cat].items(), key=lambda x: -x[1]))
        lines.append(f"| {cat} | {c['n']} | {c.get('flagged', 0)} | {c.get('flagged', 0) / c['n']:.0%} | {statuses} |")
    return "\n".join(lines) + "\n"
