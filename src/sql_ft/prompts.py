"""Prompt construction for Text-to-SQL.

The chat template comes from the tokenizer (Qwen3 ships its own Jinja template).
This module only produces the role/content list and a clean system prompt — the
tokenizer turns that into model-specific tokens via `apply_chat_template`.

Thinking mode is *disabled* for SQL: we want raw SQL out, not a <think>...</think>
trace. We rely on `tokenizer.apply_chat_template(..., enable_thinking=False)`.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a precise Text-to-SQL assistant. "
    "Given a database schema (CREATE TABLE statements) and a natural-language question, "
    "respond with a single SQL query that answers the question. "
    "Output only the SQL query, with no prose, no markdown fences, and no explanation."
)

USER_TEMPLATE = """### Schema
{schema}

### Question
{question}

### SQL"""


@dataclass(frozen=True)
class SQLExample:
    """One training/eval row."""

    schema: str  # CREATE TABLE ...
    question: str
    answer: str  # gold SQL


def build_messages(schema: str, question: str) -> list[dict[str, str]]:
    """Return the chat-format messages for a Text-to-SQL prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(schema=schema.strip(), question=question.strip()),
        },
    ]


def format_for_sft(example: SQLExample) -> list[dict[str, str]]:
    """Full conversation for SFT (includes the assistant turn with the gold SQL)."""
    return [
        *build_messages(example.schema, example.question),
        {"role": "assistant", "content": example.answer.strip()},
    ]
