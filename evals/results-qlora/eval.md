# Evaluation report

- Benchmark: `data/processed/eval.jsonl` (200 examples)
- Generation: greedy (do_sample=False), max_new_tokens=256

| Model | exec_acc % | exec_acc_seeded % | exact_match % | exact_match_loose % | BLEU | stop rate % | latency ms/ex |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 67.68 | 42.42 | 3.0 | 40.0 | 59.44 | 99.5 | 962.1 |
| `ft` | 88.38 | 71.21 | 54.0 | 72.5 | 87.0 | 100.0 | 757.6 |

Gold-coverage on this benchmark (gold queries that themselves execute): 99.0%.

Gold queries returning a non-empty result: 5.56% on the generic synthetic rows (`exec_acc`), 86.36% once the rows are seeded with the gold query's literals (`exec_acc_seeded`). `exact_match_loose` ignores quote style and case. Stop rate is the share of generations that ended on a stop token before max_new_tokens.