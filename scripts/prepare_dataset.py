"""Prepare a small Text-to-SQL training + evaluation split.

Source: `b-mc2/sql-create-context` on Hugging Face.

Each row of the source dataset has three fields:
    - question : natural-language question
    - context  : a single `CREATE TABLE ...` statement (the schema)
    - answer   : the gold SQL

We sample 300 rows for training and a disjoint 50 rows for evaluation. The eval
set is kept small on purpose: it's the "custom benchmark of ~50 examples"
referenced in the project plan, and the SQL exec-accuracy metric is the only
one that genuinely matters here — having 50 hand-runnable examples is more
useful than 5000 noisy ones.

Output:
    data/processed/train.jsonl
    data/processed/eval.jsonl

Both files use the canonical schema:
    {"schema": "...", "question": "...", "answer": "..."}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Allow running this file directly with `python scripts/prepare_dataset.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_ft.data import write_jsonl

load_dotenv()  # load HF_TOKEN / HF_USERNAME from .env if present
console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        default="b-mc2/sql-create-context",
        help="HF dataset id (must expose question/context/answer columns).",
    )
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-schema-chars",
        type=int,
        default=1500,
        help="Drop rows whose CREATE TABLE block is longer than this "
        "(keeps prompts short, eval fast).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    console.rule("[bold]Loading source dataset[/bold]")
    console.log(f"source = {args.source}")
    ds = load_dataset(args.source, split="train")

    needed = {"question", "context", "answer"}
    missing = needed - set(ds.column_names)
    if missing:
        console.print(f"[red]Source is missing columns: {missing}[/red]")
        return 2

    # Filter by schema length and dedup by (schema, question, answer).
    seen: set[tuple[str, str, str]] = set()
    cleaned: list[dict[str, str]] = []
    for row in ds:
        schema = (row["context"] or "").strip()
        question = (row["question"] or "").strip()
        answer = (row["answer"] or "").strip()
        if not schema or not question or not answer:
            continue
        if len(schema) > args.max_schema_chars:
            continue
        key = (schema, question, answer)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"schema": schema, "question": question, "answer": answer})

    console.log(f"available after dedup/filter: {len(cleaned)}")

    if len(cleaned) < args.n_train + args.n_eval:
        console.print("[red]Not enough rows after filtering.[/red]")
        return 2

    # Deterministic shuffle then split.
    import random

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)

    train_rows = cleaned[: args.n_train]
    eval_rows = cleaned[args.n_train : args.n_train + args.n_eval]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)

    # Summary report --------------------------------------------------------
    t = Table(title="Dataset splits")
    t.add_column("split", style="cyan")
    t.add_column("rows", justify="right")
    t.add_column("path")
    t.add_row("train", str(len(train_rows)), str(out_dir / "train.jsonl"))
    t.add_row("eval", str(len(eval_rows)), str(out_dir / "eval.jsonl"))
    console.print(t)

    # Pretty sample (first eval row) so a human can sanity-check.
    sample = eval_rows[0]
    console.rule("[bold]Sample eval row[/bold]")
    console.print(f"[bold]schema[/bold]:\n{sample['schema']}")
    console.print(f"[bold]question[/bold]: {sample['question']}")
    console.print(f"[bold]answer[/bold]: {sample['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
