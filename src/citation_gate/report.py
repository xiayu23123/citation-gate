"""JSON and Markdown renderings of gate results."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from .gate import AnswerReport, Policy, QuoteResult, Status

MARKER = "<!-- citation-gate -->"


def quote_json(r: QuoteResult) -> dict:
    out = {
        "text": r.quote.text,
        "attribution": r.quote.attribution,
        "cited_books": r.cited_books,
        "status": str(r.status),
        "similarity": round(r.similarity, 3),
        "source": None,
        "edits": [],
        "segments": [asdict(s) for s in r.segments],
    }
    if r.match:
        m = r.match
        out["source"] = {
            "book": m.doc.book,
            "path": m.doc.path,
            "line": m.doc.line_of(m.start),
            "excerpt": m.doc.excerpt(m.start, m.end),
        }
        out["edits"] = [str(e) for e in m.edits]
    return out


def summary(reports: list[AnswerReport], policy: Policy) -> dict:
    results = [r for a in reports for r in a.results]
    by_status = Counter(str(r.status) for r in results)
    failed_quotes = sum(policy.fails(r.status) for r in results)
    failed_answers = sum(a.failed(policy) for a in reports)
    return {
        "passed": failed_answers == 0,
        "answers": len(reports),
        "failed_answers": failed_answers,
        "quotes": len(results),
        "failed_quotes": failed_quotes,
        "warned_quotes": sum(policy.warns(r.status) for r in results),
        "uncited_sentences": sum(len(a.uncited) for a in reports),
        "by_status": {s.value: by_status.get(s.value, 0) for s in Status},
    }


def to_json(reports: list[AnswerReport], policy: Policy) -> dict:
    return {
        "summary": summary(reports, policy),
        "policy": {
            "min_chars": policy.min_chars,
            "fuzzy_threshold": policy.fuzzy_threshold,
            "critical_chars": len(policy.critical),
            "critical_terms": len(policy.critical_terms),
            "confusable_sets": len(policy.confusable),
            "detect_splices": policy.detect_splices,
            "allow_unknown_sources": policy.allow_unknown_sources,
            "review_as_pass": policy.review_as_pass,
            "require_citation": policy.require_citation,
        },
        "answers": [
            {
                "id": a.id,
                "failed": a.failed(policy),
                "quotes": [quote_json(r) for r in a.results],
                "skipped": [q.text for q in a.skipped],
                "uncited": a.uncited,
            }
            for a in reports
        ],
    }


def _cell(s: str | None, limit: int = 40) -> str:
    s = (s or "").replace("\n", " ").replace("|", "\\|").replace("<", "&lt;")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _row(a: AnswerReport, r: QuoteResult) -> str:
    source = ""
    if r.match:
        source = f"{r.match.doc.book} `{r.match.location()}`"
    detail = " ".join(str(e) for e in r.match.edits) if r.match else ""
    if r.status is Status.SPLICED:
        detail = " + ".join(f"`{s.doc_path}:{s.line}`" for s in r.segments)
    cited = "、".join(r.cited_books) if r.cited_books else (r.quote.attribution or "—")
    return (
        f"| {_cell(a.id, 24)} | **{r.status}** | {_cell(r.quote.text)} | {_cell(cited, 20)} "
        f"| {source} | {r.similarity:.2f} | {_cell(detail, 60)} |"
    )


_HEAD = "| answer | status | quote | cited | closest source | sim | detail |\n|---|---|---|---|---|---|---|"


def to_markdown(reports: list[AnswerReport], policy: Policy) -> str:
    s = summary(reports, policy)
    verdict = "PASS" if s["passed"] else "FAIL"
    lines = [
        MARKER,
        f"### citation-gate: {verdict}",
        "",
        f"{s['quotes']} quotes in {s['answers']} answers; "
        f"{s['failed_quotes']} quotes failed, in {s['failed_answers']} answers.",
        "",
        " · ".join(f"{k} {v}" for k, v in s["by_status"].items() if v) or "no quotations found",
        "",
    ]
    failing = [(a, r) for a in reports for r in a.results if policy.fails(r.status)]
    if failing:
        lines += ["#### Failed quotes", "", _HEAD] + [_row(a, r) for a, r in failing] + [""]
    warned = [(a, r) for a in reports for r in a.warnings(policy)]
    if warned:
        lines += ["#### Passed with a warning", ""]
        if any(r.status is Status.NEEDS_REVIEW for _, r in warned):
            lines += [
                "`needs_review`: the only critical differences are between characters that are easily "
                "confused (己/已/巳, 戊/戍/戌 …). That may be an OCR slip or a real change; check the source.",
                "",
            ]
        lines += [_HEAD] + [_row(a, r) for a, r in warned] + [""]
    if policy.require_citation:
        uncited = [(a, u) for a in reports for u in a.uncited]
        if uncited:
            lines += ["#### Uncited sentences", ""]
            lines += [f"- `{_cell(a.id, 24)}` {_cell(u, 80)}" for a, u in uncited] + [""]
    passing = [(a, r) for a in reports for r in a.results if not policy.fails(r.status) and not policy.warns(r.status)]
    if passing:
        lines += ["<details><summary>Passed quotes</summary>", "", _HEAD]
        lines += [_row(a, r) for a, r in passing] + ["", "</details>", ""]
    return "\n".join(lines)
