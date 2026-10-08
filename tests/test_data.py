"""Tests for sql_ft.data — JSONL I/O and dataset construction.

`build_sft_dataset` is exercised with a fake tokenizer whose chat template
mimics Qwen's (`<|im_end|>\\n` after every turn), so no Hub download is needed.
The real chat-message layout is covered by tests/test_prompts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from sql_ft.data import build_sft_dataset, read_jsonl, rows_to_examples, write_jsonl
from sql_ft.prompts import SQLExample


class FakeQwenTokenizer:
    """Renders messages the way Qwen's template does: every turn closes with
    `<|im_end|>\\n`, so the rendered text never ends on the EOS token itself."""

    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not add_generation_prompt and not enable_thinking
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


ROWS = [
    {
        "schema": "CREATE TABLE t (id INT)",
        "question": "How many?",
        "answer": "SELECT COUNT(*) FROM t",
    },
    {
        "schema": "CREATE TABLE u (name TEXT)",
        "question": "List names",
        "answer": "SELECT name FROM u",
    },
]


def test_build_sft_dataset_ends_on_eos_token():
    ds = build_sft_dataset(ROWS, FakeQwenTokenizer())
    assert ds.column_names == ["text"]
    assert len(ds) == 2
    for text, row in zip(ds["text"], ROWS, strict=True):
        assert text.endswith(row["answer"] + "<|im_end|>")
        assert text.count("<|im_start|>assistant") == 1
        assert "### Schema" in text


def test_build_sft_dataset_has_single_trailing_eos():
    # SFTTrainer appends eos_token to any text not ending with it; a stray
    # newline would leave the model learning `<|im_end|>\n<|im_end|>`.
    text = build_sft_dataset(ROWS[:1], FakeQwenTokenizer())["text"][0]
    assert not text.endswith("\n")
    assert text.endswith("<|im_end|>") and not text.endswith("<|im_end|>\n<|im_end|>")


def test_write_and_read_roundtrip(tmp_path: Path):
    rows = [
        {
            "schema": "CREATE TABLE t (id INT)",
            "question": "How many?",
            "answer": "SELECT COUNT(*) FROM t",
        },
        {
            "schema": "CREATE TABLE u (name TEXT)",
            "question": "List names",
            "answer": "SELECT name FROM u",
        },
    ]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_write_creates_parent_dirs(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "out.jsonl"
    write_jsonl(deep, [{"x": 1}])
    assert deep.exists()
    assert read_jsonl(deep) == [{"x": 1}]


def test_read_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "with_blanks.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_write_uses_utf8(tmp_path: Path):
    rows = [{"q": "café résumé naïve"}]
    path = tmp_path / "utf8.jsonl"
    write_jsonl(path, rows)
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw.strip())
    assert parsed["q"] == "café résumé naïve"


def test_rows_to_examples_returns_typed_objects():
    rows = [{"schema": "s", "question": "q", "answer": "a"}]
    exs = rows_to_examples(rows)
    assert len(exs) == 1
    assert isinstance(exs[0], SQLExample)
    assert (exs[0].schema, exs[0].question, exs[0].answer) == ("s", "q", "a")


def test_rows_to_examples_preserves_order():
    rows = [{"schema": f"s{i}", "question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    exs = rows_to_examples(rows)
    assert [e.schema for e in exs] == [f"s{i}" for i in range(5)]
