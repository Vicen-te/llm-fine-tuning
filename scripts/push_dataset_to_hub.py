"""Push the prepared train/eval JSONL splits to the Hugging Face Hub.

Requires `HF_TOKEN` (write scope) in the environment or `.env`. Creates the
repo if it doesn't exist, uploads both splits, and writes a dataset card
describing provenance and licensing of the underlying corpus.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, HfApi
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_ft.data import read_jsonl

console = Console()
load_dotenv()


DATASET_BODY = """\
# SQL Create Context (mini) — 300 train / 200 eval

A curated, deduplicated 500-row split of
[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
used to LoRA-fine-tune Qwen3.5-2B for Text-to-SQL.

## Schema

Each row has three string fields:

| field      | description |
|------------|------------|
| `schema`   | A `CREATE TABLE ...` statement (SQLite-flavoured) |
| `question` | Natural-language question |
| `answer`   | Gold SQL query |

## Splits

| split | rows |
|-------|------|
| train | 300  |
| eval  | 200  |

## Provenance

Filtered to schemas ≤ 1500 chars and deduplicated on the full triple. Sampled
with `seed=42` from the source train split. No examples leak between splits.

## Intended use

This split is intentionally small (500 rows) for parameter-efficient
fine-tuning on a single GPU. It is not a capability benchmark; use the
upstream Spider/BIRD datasets for that.

## Licensing

Inherits the source dataset's CC-BY-4.0 license.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-file", default="data/processed/train.jsonl")
    p.add_argument("--eval-file", default="data/processed/eval.jsonl")
    p.add_argument("--repo-id", required=True, help="e.g. YOUR_HF_USERNAME/sql-create-context-mini")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]HF_TOKEN not set (see .env.example).[/red]")
        return 2

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    train_rows = read_jsonl(args.train_file)
    eval_rows = read_jsonl(args.eval_file)
    console.log(f"train={len(train_rows)}  eval={len(eval_rows)}")

    dd = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    )
    dd.push_to_hub(args.repo_id, token=token, private=args.private)

    # Merge our metadata + prose into the card that push_to_hub auto-generated,
    # so its configs/dataset_info block (which powers the Dataset Viewer and the
    # split stats) survives instead of being overwritten by a hand-written card.
    card = DatasetCard.load(args.repo_id, repo_type="dataset", token=token)
    card.data.license = "cc-by-4.0"
    card.data.language = ["en"]
    card.data.size_categories = ["n<1K"]
    card.data.task_categories = ["text-generation", "table-question-answering"]
    card.data.tags = ["sql", "text-to-sql", "fine-tuning", "qwen3.5"]
    card.data.pretty_name = "SQL Create Context (mini)"
    card.text = DATASET_BODY
    card.push_to_hub(args.repo_id, repo_type="dataset", token=token)

    console.print(f"[green]Pushed to https://huggingface.co/datasets/{args.repo_id}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
