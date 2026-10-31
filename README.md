# citation-gate

Check that the passages an LLM answer *quotes* actually exist in your source
texts, and fail the build when they don't. Built for classical Chinese
(traditional/simplified, variant characters, OCR noise, editorial punctuation),
usable on any corpus of plain text.

> 中文简介：给 RAG 回答做「引文闸门」。回答里每一段带引号的原文，都要能在你的典籍库里逐字（或仅差 OCR 噪声）找到，并且出处要对得上；改了关键字（寒→热、肾→肝、漏掉「不」）、张冠李戴、东拼西凑、凭空编造的引文，一律拦下。可作命令行、Python 库或 GitHub Action 使用。

```text
$ citation-gate check examples/answers.jsonl --corpus corpus
q04-growth-wrong: altered 「女子七岁，肝气盛，齿更发长」 素問 suwen-1.txt:14 肾→肝
q05-winter: altered 「故藏于精者，春病温。」 素問 suwen-1.txt:101 -不
q06-taiyang: misattributed 「太阳之为病，脉浮，头项强痛而恶寒。」 傷寒論 shanghan.txt:156
q07-longevity: spliced 「上古之人，春秋皆度百岁，阴平阳秘，精神乃治。」 suwen-1.txt:5 + suwen-1.txt:75
q08-qi: unknown_source 「人之所以生者，气也。」
citation-gate FAIL: 5/10 quotes failed in 5/8 answers
```

The other five quotes in that file pass: they are written in simplified
characters against a traditional-character source, and one carries an
OCR-style slip (曰→日).

## Why

A model asked about a classical text will produce quotations that look right
and are not: a character changed, two real lines glued together, a line
from 《伤寒论》 credited to 《金匮要略》, or a modern paraphrase dressed in 「」.
Readers can't check, and the product looks authoritative precisely because
it quotes.

Exact string matching catches all of this but also rejects every quote that
differs from your copy of the source by script, variant form or an OCR error.
Fuzzy matching fixes that and introduces a worse problem: **a matcher tuned
to forgive one wrong character forgives 寒→热 and a dropped 不 just the same.**
On the benchmark below, plain fuzzy matching passes 50 of 50 meaning-changing
edits. The fix is to diff, not score: align the quote to its source and reject
the match if any difference touches a character that carries meaning.

The design comes from the citation gate in a production RAG product of mine
over medical and divination classics, where the rule is "the program owns the
facts, the model owns the wording". This repository is a clean
reimplementation of that idea for public use, not that product's code.

## What each quote gets

| status | meaning | fails? |
|---|---|---|
| `verified` | verbatim in the cited work (anywhere in the corpus if none is cited) | no |
| `fuzzy` | near-verbatim; no difference touches a critical character | no |
| `altered` | near-verbatim, but a critical character differs (`肾→肝`, `-不`) | yes |
| `misattributed` | in the corpus, but not in the work it is credited to | yes |
| `spliced` | stitched from fragments of different passages | yes |
| `not_found` | nothing close enough anywhere | yes |
| `unknown_source` | credits a work the corpus does not contain | yes, unless `--allow-unknown-sources` |

Quotations shorter than four characters are treated as emphasis and skipped.
With `--require-citation`, sentences that neither quote nor name a source also
fail the answer.

## How it works

1. **Fold** both sides identically: Unicode NFKC, OpenCC's traditional→simplified
   character table, a small table of variants common in medical classics
   (鞕→硬, 濇→涩, 欬→咳, 沈→沉 …, see `src/citation_gate/data/variants.tsv`),
   lower-case, and drop everything that is not a letter or digit. Classical
   texts were punctuated by later editors, so punctuation is not evidence.
   Folded positions map back to the original, so reports quote your source
   as written.
2. **Find quotes** in 「」『』“”"" and their attribution: the last 《…》 earlier
   in the same sentence, or one right after (`（《伤寒论》）`, `——《金匮要略》`).
   《黄帝内经·素问·上古天真论》 resolves to a book through titles and aliases.
3. **Search**: exact substring first. Otherwise character bigrams vote for
   alignment diagonals, votes within a band are pooled (an insertion shifts
   the diagonal for the rest of the quote), and the top windows are aligned
   with semi-global edit distance, i.e. the whole quote against any stretch
   of source.
4. **Judge the diff**, not just the score. Similarity must clear the threshold
   (0.85 by default, chosen on the dev split), *and* no substitution,
   insertion or deletion may touch a critical character. The bundled `tcm`
   profile lists numerals, negations, organs, the eight principles (阴阳表里
   寒热虚实), channels, five phases, treatment verbs, direction and size,
   prognosis and sex; see `src/citation_gate/data/critical-tcm.txt`. Pass
   `--critical none` or your own file for other domains.
5. **Explain failures**: attribution is checked against the book the best match
   came from; a quote that matches nowhere is covered greedily by exact
   fragments, and two or more fragments that do not line up in one passage
   are reported as a splice, with both locations.

No dependencies beyond the standard library.

## Benchmark

`bench/` holds a labelled benchmark built deterministically from the corpus by
`scripts/build_bench.py`: 296 items per split, each one answer with one quote
under test. Labels come from construction, never from running the gate.

| category | built as | should |
|---|---|---|
| genuine | a real passage, simplified (70%) or as in the source | pass |
| noise | a real passage with 1–2 OCR-style slips on non-critical characters | pass |
| altered | one critical edit: organ, numeral, polarity, or dropping 不 | fail |
| herb_swap | one herb or formula name replaced by another | fail |
| misattributed | a real passage credited to another book in the corpus | fail |
| spliced | two real fragments from different places, joined | fail |
| fabricated | classical-looking text from a character Markov model of the corpus | fail |
| paraphrase | a hand-written modern paraphrase presented as a quote | fail |

Protocol: the fuzzy threshold is chosen on `dev` only; `test` is scored once
per configuration. CI rebuilds both splits and rescores them on every push and
fails if anything differs from what is committed. Full output, including every
error, is in [bench/RESULTS.md](bench/RESULTS.md).

Test split, threshold 0.85, each row adding one mechanism:

| configuration | bad quotes caught | good quotes rejected |
|---|---|---|
| exact match only | 100.0% | 45.5% |
| + fuzzy match | 70.4% | 0.0% |
| + critical characters | 98.4% | 0.0% |
| + splice detection | 98.4% | 0.0% |

- Fuzzy matching removes every false alarm and lets through **all 50**
  `altered` quotes. Diffing against critical characters catches all 50 again
  without rejecting a single genuine or noisy quote.
- Splice detection changes no pass/fail decision here; those quotes already fail
  as `not_found`. It only says what went wrong and where both halves came from.
- **`herb_swap` is caught 92% of the time, mostly by accident.** Herbs are not
  critical characters. A two-character swap in a short quote pushes similarity
  below the threshold; in a long one it doesn't. All three misses are quotes of
  17+ characters, and the benchmark's quotes are short (8–30 characters), so
  real recall on long quotes will be lower. The next step is critical *terms*
  (herb and formula names, symptoms), not a tighter threshold: at 0.90 the gate
  already rejects 25% of good quotes on dev.

### What this benchmark does not show

- The noise is synthetic: visually confusable pairs, dropped and doubled
  characters, at most two per quote. Real OCR error is burstier.
- The corpus is small (four files, about 63,000 characters). The bigram index
  is in memory; corpora in the hundreds of millions of characters are untested.
- Books that share passages (《伤寒论》 and 《金匮要略》 do) make
  misattribution genuinely ambiguous. The gate accepts a quote found in any
  book its attribution can refer to.
- The gate checks *verifiability against your library*, not truth. A real
  quote from a book you don't have fails as `unknown_source` or `not_found`.
  Paraphrase that is not presented as a quote is out of scope, except that
  `--require-citation` flags sentences that cite nothing.

## Use

```bash
pip install git+https://github.com/xiayu23123/citation-gate
```

### Command line

```bash
citation-gate check answers.jsonl --corpus corpus/ --markdown report.md --json report.json
```

Answers can be `.jsonl` or `.json` records (`--field answer`, `--id-field id`),
plain text files, or `-` for stdin. The exit code is 0 if everything passes,
1 if any answer fails, 2 on usage errors. To score the gate on your own
labelled data, use `citation-gate eval bench.jsonl --corpus corpus/`.

### Python

```python
from citation_gate import Corpus, Policy, check_answer

corpus = Corpus.load("corpus")
report = check_answer(corpus, "《素问》云：「女子七岁，肝气盛，齿更发长」")
for r in report.results:
    print(r.status, r.note())  # altered 素問 suwen-1.txt:14 肾→肝
```

### GitHub Action

Run the gate over your model's answers on every pull request. The report goes
to the job summary and, on pull requests, to a comment that is updated in place.

```yaml
on: pull_request
permissions:
  contents: read
  pull-requests: write
jobs:
  quotes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python eval/generate_answers.py > answers.jsonl  # your own step
      - uses: xiayu23123/citation-gate@v0.1.0
        with:
          answers: answers.jsonl
          corpus: sources/
```

Inputs mirror the CLI flags (`fuzzy-threshold`, `critical`,
`allow-unknown-sources`, `require-citation`), plus `fail-on-error` and
`comment`. Outputs: `passed`, `failed-quotes`, `report`.

### Your corpus

A directory of UTF-8 `.txt` files (change with `--glob`). A file's book title
is its first line if that reads `# Title`, else its file name; files with the
same title form one book. An optional `books.json` adds aliases and titles:

```json
{
  "aliases": { "素問": ["黃帝內經", "內經"] },
  "titles": { "misc/scan-017.txt": "金匱要略" }
}
```

The bundled `corpus/` contains public-domain texts from Chinese Wikisource,
pinned to revisions in [corpus/SOURCES.md](corpus/SOURCES.md) and fetched by
`scripts/fetch_corpus.py`.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/build_bench.py && python scripts/bench.py
```

## License

Code: MIT. `src/citation_gate/data/TSCharacters.txt` is from
[OpenCC](https://github.com/BYVoid/OpenCC) 1.1.9 (Apache-2.0); see
`src/citation_gate/data/NOTICE`. Corpus texts are in the public domain;
Wikisource's editorial layer is CC BY-SA 4.0.
