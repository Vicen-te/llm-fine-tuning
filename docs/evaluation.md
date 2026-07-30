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
  `question`, `gold`, `pred_raw` and the cleaned `pred`.

`evals/results-qlora/` holds the same report for the QLoRA adapter, produced by
pointing `--merged-model` at `outputs/qwen3.5-2b-sql-qlora-merged`.

Both directories are committed, so the reported numbers can be re-derived from
the repo: `make data` rebuilds the same seeded eval split and `id` is the row
index into it, which is all `eval_sql.py` needs to re-score a prediction.

## Metrics

Each prediction is cleaned (`clean_sql_output`) and scored three ways:

- **Executable accuracy** — the predicted and gold SQL are run against the same
  in-memory SQLite database built from the row's `CREATE TABLE` schema; a match
  means identical result sets. This is the metric that matters: it rewards a
  semantically-correct query even when it is phrased differently from the gold.
- **Exact match** — `sqlglot`-normalized string equality. Strict; punishes
  harmless rewrites.
- **BLEU** — `sacrebleu` over the SQL text. Surface similarity only.

The report also prints **gold coverage**: the share of reference queries that
themselves execute on their schema, an upper bound on achievable executable
accuracy (99% on this split).

## Quantization report

`scripts/quantize_4bit.py` writes `evals/results/quantization.json`: a direct
bf16-vs-NF4 comparison (disk, latency, the three metrics) on the same split, so
the quantization trade-off is measured rather than assumed.
