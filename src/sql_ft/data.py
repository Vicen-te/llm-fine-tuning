"""Dataset I/O and chat-template formatting for SFT.

The on-disk format is plain JSONL with three string fields per row:
    {"schema": "CREATE TABLE ...", "question": "...", "answer": "SELECT ..."}

`build_sft_dataset` turns those rows into a HuggingFace `Dataset` whose `text`
column is the fully-formatted chat string (system + user + assistant), ready for
TRL's `SFTTrainer` in plain-text mode. We don't pre-tokenize here — SFTTrainer
handles that and packs sequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from .prompts import SQLExample, format_for_sft
from .tokens import ensure_eos_ending


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to JSONL (utf-8, one JSON per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def rows_to_examples(rows: list[dict[str, Any]]) -> list[SQLExample]:
    """Coerce dict rows into typed examples."""
    return [
        SQLExample(schema=r["schema"], question=r["question"], answer=r["answer"]) for r in rows
    ]


def build_sft_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool = False,
) -> Dataset:
    """Turn raw rows into a `Dataset` with a single `text` column ready for SFT.

    Each `text` is the full chat (system + user + assistant) rendered with the
    model's chat template. Thinking mode is off for SQL.

    The text ends on the tokenizer's EOS token, which for Qwen is the turn end
    `<|im_end|>`. The chat template emits `<|im_end|>\\n` after the assistant
    turn; the trailing newline is dropped so the last supervised token is the
    stop token itself and TRL's SFTTrainer (which appends `eos_token` to any
    text not already ending with it) does not add a second one — a model trained
    on `SQL<|im_end|>\\n<|im_end|>` learns to continue past the turn end.
    """
    examples = rows_to_examples(rows)
    eos = tokenizer.eos_token
    texts: list[str] = []
    for ex in examples:
        msgs = format_for_sft(ex)
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        texts.append(ensure_eos_ending(text, eos))
    return Dataset.from_dict({"text": texts})
