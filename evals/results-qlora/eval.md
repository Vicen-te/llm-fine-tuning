# Evaluation report

- Benchmark: `data/processed/eval.jsonl` (200 examples)
- Generation: greedy (do_sample=False), max_new_tokens=256

| Model | exec_acc % | exact_match % | BLEU | latency ms/ex |
|---|---:|---:|---:|---:|
| `base` | 67.68 | 3.0 | 59.44 | 1077.5 |
| `ft` | 88.89 | 57.5 | 87.46 | 7342.8 |

Gold-coverage on this benchmark (gold queries that themselves execute): 99.0%.