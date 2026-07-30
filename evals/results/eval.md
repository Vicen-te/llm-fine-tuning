# Evaluation report

- Benchmark: `data/processed/eval.jsonl` (200 examples)
- Generation: greedy (do_sample=False), max_new_tokens=256

| Model | exec_acc % | exact_match % | BLEU | latency ms/ex |
|---|---:|---:|---:|---:|
| `base` | 67.68 | 3.0 | 59.44 | 1082.4 |
| `ft` | 87.37 | 54.0 | 86.35 | 7413.2 |
| `ft-nf4` | 85.86 | 51.0 | 85.85 | 11786.0 |

Gold-coverage on this benchmark (gold queries that themselves execute): 99.0%.