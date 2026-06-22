# Training

LoRA and QLoRA fine-tuning of Qwen3.5-2B for Text-to-SQL, driven by
`scripts/train.py` and the YAML configs in `configs/`.

## Data

```bash
make data
```

Loads `b-mc2/sql-create-context`, deduplicates on the (schema, question, answer)
triple, drops schemas longer than 1500 chars, and writes a seeded 300 train /
200 eval split to `data/processed/`. Run `scripts/prepare_dataset.py --help` to
change the source, sizes, or seed.

## Run

```bash
make train          # LoRA, bf16     (configs/train_lora.yaml)
make train-qlora    # QLoRA, 4-bit   (configs/train_qlora.yaml)
```

The adapter and tokenizer land in `outputs/qwen3.5-2b-sql-lora/`. Training logs
go to TensorBoard (`report_to: tensorboard`).

## Configs

Both YAMLs share the LoRA and data blocks; the QLoRA one adds a 4-bit
quantization block and a paged optimizer. The fields that matter:

| field | meaning |
|---|---|
| `lora.r` / `lora.alpha` | LoRA rank and scaling (16 / 32) |
| `lora.target_modules` | linear layers the adapter wraps |
| `train.num_train_epochs` | 3 |
| `train.learning_rate` | 2e-4, cosine schedule |
| `train.warmup_steps` | 3 (~5% of the ~57-step run) |
| `train.loss_type` | `chunked_nll` (TRL 1.7 default, lower peak memory) |
| `quant.load_in_4bit` | QLoRA only: NF4 double-quant on the frozen base |

## Merge

```bash
make merge
```

Folds the adapter into the base weights (`peft.merge_and_unload`) and writes a
standalone bf16 model to `outputs/qwen3.5-2b-sql-merged/` — drop-in for
transformers or vLLM, with no PEFT runtime dependency.
