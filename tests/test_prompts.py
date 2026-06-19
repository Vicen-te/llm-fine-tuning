"""Tests for sql_ft.prompts — chat-message construction for SFT and inference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sql_ft.prompts import SYSTEM_PROMPT, SQLExample, build_messages, format_for_sft


def test_build_messages_has_system_and_user():
    msgs = build_messages("CREATE TABLE t (id INT)", "Count rows")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "CREATE TABLE t (id INT)" in msgs[1]["content"]
    assert "Count rows" in msgs[1]["content"]
    assert "### Schema" in msgs[1]["content"]
    assert "### Question" in msgs[1]["content"]
    assert msgs[1]["content"].rstrip().endswith("### SQL")


def test_build_messages_strips_whitespace():
    msgs = build_messages("  CREATE TABLE t (id INT)  \n", "  Count rows  \n")
    # Leading/trailing whitespace is stripped from the embedded values.
    assert "  CREATE TABLE" not in msgs[1]["content"]
    assert "Count rows  " not in msgs[1]["content"]


def test_format_for_sft_appends_assistant_turn():
    ex = SQLExample(
        schema="CREATE TABLE t (id INT)",
        question="Count rows",
        answer="SELECT COUNT(*) FROM t",
    )
    msgs = format_for_sft(ex)
    assert len(msgs) == 3
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"] == "SELECT COUNT(*) FROM t"


def test_format_for_sft_strips_answer_whitespace():
    ex = SQLExample(
        schema="CREATE TABLE t (id INT)",
        question="Count rows",
        answer="  SELECT COUNT(*) FROM t  \n",
    )
    msgs = format_for_sft(ex)
    assert msgs[-1]["content"] == "SELECT COUNT(*) FROM t"


def test_sql_example_is_frozen():
    ex = SQLExample(schema="s", question="q", answer="a")
    with pytest.raises(FrozenInstanceError):
        ex.schema = "other"  # type: ignore[misc]
