"""Tests for sql_ft.data — JSONL I/O and dataset construction.

We don't exercise `build_sft_dataset` here because it needs a real tokenizer
(downloads from the Hub). The chat-template formatting is implicitly covered
by tests/test_prompts.py instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from sql_ft.data import read_jsonl, rows_to_examples, write_jsonl
from sql_ft.prompts import SQLExample


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
