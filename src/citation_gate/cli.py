"""Command line: `citation-gate check` and `citation-gate eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, evaluate, report
from .corpus import Corpus
from .gate import Policy, check_answer
from .normalize import load_critical, load_terms


def _answers(paths: list[str], field: str, id_field: str) -> list[tuple[str, str]]:
    """(id, answer) pairs from .jsonl/.json records or whole text files ('-' is stdin)."""
    out = []
    for p in paths:
        if p == "-":
            out.append(("stdin", sys.stdin.read()))
            continue
        path = Path(p)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif path.suffix == ".json":
            data = json.loads(text)
            records = data if isinstance(data, list) else [data]
        else:
            out.append((path.name, text))
            continue
        for n, rec in enumerate(records, 1):
            out.append((str(rec.get(id_field, f"{path.name}#{n}")), rec[field]))
    return out


def _policy(args: argparse.Namespace) -> Policy:
    return Policy(
        min_chars=args.min_chars,
        fuzzy_threshold=args.fuzzy_threshold,
        critical=load_critical(args.critical),
        critical_terms=load_terms(args.critical_terms),
        detect_splices=not args.no_splices,
        allow_unknown_sources=args.allow_unknown_sources,
        require_citation=args.require_citation,
    )


def _write(path: str | None, text: str) -> None:
    if not path:
        return
    if path == "-":
        sys.stdout.write(text)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def cmd_check(args: argparse.Namespace) -> int:
    corpus = Corpus.load(args.corpus, args.glob)
    policy = _policy(args)
    reports = [check_answer(corpus, text, policy, id) for id, text in _answers(args.answers, args.field, args.id_field)]
    data = report.to_json(reports, policy)
    _write(args.json, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    _write(args.markdown, report.to_markdown(reports, policy))
    s = data["summary"]
    if not args.quiet:
        for a in reports:
            for r in a.failures(policy):
                print(f"{a.id}: {r.status} 「{r.quote.text[:30]}」 {r.note()}", file=sys.stderr)
        verdict = "PASS" if s["passed"] else "FAIL"
        print(
            f"citation-gate {verdict}: {s['failed_quotes']}/{s['quotes']} quotes failed "
            f"in {s['failed_answers']}/{s['answers']} answers",
            file=sys.stderr,
        )
    return 0 if s["passed"] else 1


def cmd_eval(args: argparse.Namespace) -> int:
    corpus = Corpus.load(args.corpus, args.glob)
    result = evaluate.run(corpus, evaluate.load_bench(args.bench), _policy(args))
    _write(args.json, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _write(args.markdown, evaluate.to_markdown(result, f"Benchmark: {Path(args.bench).name}"))
    print(
        f"recall {result['recall']:.3f}  false_alarm {result['false_alarm']:.3f}  f1 {result['f1']:.3f}",
        file=sys.stderr,
    )
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", required=True, help="directory of source texts")
    p.add_argument("--glob", default="**/*.txt", help="corpus files to load (default: **/*.txt)")
    p.add_argument("--fuzzy-threshold", type=float, default=0.85, help="minimum similarity for a near-verbatim match")
    p.add_argument("--critical", default="tcm", help="'tcm', 'none', or a file of critical characters")
    p.add_argument("--critical-terms", default="tcm", help="'tcm' (herb and formula names), 'none', or a file of critical terms, one per line")
    p.add_argument("--min-chars", type=int, default=4, help="ignore shorter quotations")
    p.add_argument("--no-splices", action="store_true", help="do not look for quotes stitched from fragments")
    p.add_argument("--allow-unknown-sources", action="store_true", help="warn instead of fail on works not in the corpus")
    p.add_argument("--require-citation", action="store_true", help="fail answers with uncited sentences")
    p.add_argument("--json", help="write the JSON report here ('-' for stdout)")
    p.add_argument("--markdown", help="write the Markdown report here ('-' for stdout)")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="citation-gate", description="Check that quoted passages exist in a source corpus.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="check answers")
    check.add_argument("answers", nargs="+", help=".jsonl/.json records, text files, or '-'")
    check.add_argument("--field", default="answer", help="answer field in JSON records")
    check.add_argument("--id-field", default="id", help="id field in JSON records")
    check.add_argument("-q", "--quiet", action="store_true")
    _common(check)
    check.set_defaults(func=cmd_check)

    ev = sub.add_parser("eval", help="score the gate on a labelled benchmark")
    ev.add_argument("bench", help="benchmark .jsonl")
    _common(ev)
    ev.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
