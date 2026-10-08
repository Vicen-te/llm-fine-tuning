# Evaluation

`scripts/evaluate.py` benchmarks the base model, the merged fine-tune, and the
NF4 4-bit copy on the held-out eval split, and writes a report to
`evals/results/`.

## Run

```bash
make quantize    # first, so the NF4 model exists (adds the ft-nf4 row)
make evaluate
```

Outputs:
- `evals/results/eval.md` / `eval.json` — the summary table.
- `evals/results/predictions/{base,ft,ft-nf4}.jsonl` — one row per example with
  `question`, `gold`, `pred_raw`, the cleaned `pred`, `new_tokens` and whether
  the generation `stopped` on a stop token.

`evals/results-qlora/` holds the same report for the QLoRA adapter, produced by
pointing `--merged-model` at `outputs/qwen3.5-2b-sql-qlora-merged`.

Both directories are committed, so the reported numbers can be re-derived from
the repo: `make data` rebuilds the same seeded eval split and `id` is the row
index into it, which is all `eval_sql.py` needs to re-score a prediction.

## Metrics

Each prediction is cleaned (`clean_sql_output`) and scored as follows:

- **Executable accuracy** (`exec_acc`) — the predicted and gold SQL are run
  against the same in-memory SQLite database built from the row's
  `CREATE TABLE` schema, filled with five generic rows per table; a match means
  identical result sets. It rewards a semantically-correct query even when it
  is phrased differently from the gold, but the generic rows ("alpha", 1,
  2024-01-01 ...) never satisfy a `WHERE publisher = "Nintendo"` clause: only
  5.6% of the gold queries return anything, two empty results match, and the
  number is close to "the prediction executes".
- **Seeded executable accuracy** (`exec_acc_seeded`) — same comparison, but the
  rows also contain the literals from the gold query's WHERE clause
  (`literals_by_column`: `=`, `<`, `>`, `LIKE`, `BETWEEN`, `IN`, numbers with
  their neighbours), so 86.4% of the gold queries return rows and a prediction
  has to filter the same column with the same value to match. This is the
  stricter number to quote; it still cannot see joins or columns the gold never
  filters on.
- **Exact match** (`exact_match`) — `sqlglot`-normalized string equality.
  Strict: it keeps quote style and literal case, and the corpus writes
  `"Nintendo"` where the base model writes `'Nintendo'`.
- **Loose exact match** (`exact_match_loose`) — the same comparison with quote
  style, case and whitespace folded away.
- **BLEU** — `sacrebleu` over the SQL text. Surface similarity only.
- **Stop rate** — share of generations that ended on a stop token before
  `max_new_tokens`. Anything under 100% means the latency column is measuring
  the token budget, not the model.

The report also prints **gold coverage**: the share of reference queries that
themselves execute on their schema, an upper bound on achievable executable
accuracy (99% on this split), and the share of gold queries returning a
non-empty result under each row setup.

## Quantization report

`scripts/quantize_4bit.py` writes `evals/results/quantization.json`: a direct
bf16-vs-NF4 comparison (disk, latency, the three metrics) on the same split, so
the quantization trade-off is measured rather than assumed.
