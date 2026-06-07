"""SQL-aware evaluation metrics for Text-to-SQL.

Three complementary signals, applied to every (gold, pred) pair:

1. **Executable accuracy** — the only metric that actually matters in
   production. We materialise the CREATE TABLE schema in an in-memory SQLite
   DB, populate it with synthetic rows derived from the column types, run both
   queries, and compare result-sets (order-agnostic by default).

2. **Normalized exact match** — both SQL strings are parsed and re-emitted by
   sqlglot (which canonicalises whitespace, quoting, keyword case, and alias
   ordering). Cheap, dialect-aware sanity check.

3. **BLEU-4** — surface-form similarity. Useful as a "directionally improving"
   signal even when neither query executes, but not a quality gate on its own.

We tolerate SQLite-only dialect — the b-mc2/sql-create-context corpus is
SQLite-flavoured, so this is fine for benchmark purposes. For other dialects
swap the executor and adjust sqlglot's `read=...` argument.
"""

from __future__ import annotations

import contextlib
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import sacrebleu
import sqlglot

# ---------------------------------------------------------------------------
# 1. Normalized exact match (sqlglot)
# ---------------------------------------------------------------------------


def _normalize(sql: str) -> str | None:
    """Parse + re-emit SQL through sqlglot. Returns None on parse failure."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree.sql(dialect="sqlite", normalize=True, pretty=False)
    except Exception:
        return None


def normalized_exact_match(gold: str, pred: str) -> bool:
    """True iff `gold` and `pred` canonicalise to the same string."""
    g, p = _normalize(gold), _normalize(pred)
    return g is not None and p is not None and g == p


# ---------------------------------------------------------------------------
# 2. BLEU (corpus-level)
# ---------------------------------------------------------------------------


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    """sacrebleu corpus-BLEU. Scaled 0–100 like the original paper."""
    if not predictions:
        return 0.0
    refs = [[r.strip() for r in references]]
    hyps = [p.strip() for p in predictions]
    return sacrebleu.corpus_bleu(hyps, refs).score


# ---------------------------------------------------------------------------
# 3. Executable accuracy
# ---------------------------------------------------------------------------

# A type -> synthetic value generator. We populate every table with a handful
# of rows so SELECTs aren't degenerate (COUNT, AVG, MAX etc. produce signal).
_TYPE_SAMPLES: dict[str, list[Any]] = {
    "INT": [1, 2, 3, 4, 5],
    "INTEGER": [1, 2, 3, 4, 5],
    "BIGINT": [10, 20, 30, 40, 50],
    "REAL": [1.5, 2.5, 3.5, 4.5, 5.5],
    "FLOAT": [1.5, 2.5, 3.5, 4.5, 5.5],
    "DOUBLE": [1.5, 2.5, 3.5, 4.5, 5.5],
    "NUMERIC": [10.0, 20.0, 30.0, 40.0, 50.0],
    "DECIMAL": [10.0, 20.0, 30.0, 40.0, 50.0],
    "TEXT": ["alpha", "beta", "gamma", "delta", "epsilon"],
    "VARCHAR": ["alpha", "beta", "gamma", "delta", "epsilon"],
    "CHAR": ["a", "b", "c", "d", "e"],
    "DATE": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
    "DATETIME": [
        "2024-01-01 10:00",
        "2024-02-01 11:00",
        "2024-03-01 12:00",
        "2024-04-01 13:00",
        "2024-05-01 14:00",
    ],
    "TIMESTAMP": [
        "2024-01-01 10:00:00",
        "2024-02-01 11:00:00",
        "2024-03-01 12:00:00",
        "2024-04-01 13:00:00",
        "2024-05-01 14:00:00",
    ],
    "BOOL": [0, 1, 0, 1, 0],
    "BOOLEAN": [0, 1, 0, 1, 0],
}


def _sample_for_column(col_type: str, idx: int, rng: random.Random) -> Any:
    """Pick a value for column `col_type` on row `idx`."""
    t = col_type.upper().split("(")[0].strip()
    samples = _TYPE_SAMPLES.get(t)
    if samples is None:
        # Default to text — most "unknown" types are some variant of string.
        samples = _TYPE_SAMPLES["TEXT"]
    return samples[idx % len(samples)]


def _populate_sqlite(conn: sqlite3.Connection, schema: str, n_rows: int = 5, seed: int = 0) -> None:
    """Execute CREATE TABLE statements and insert `n_rows` synthetic rows per table."""
    rng = random.Random(seed)
    # Run the raw CREATE TABLEs as-is (SQLite is permissive enough).
    for stmt in schema.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        with contextlib.suppress(sqlite3.Error):
            conn.execute(stmt)

    # Populate every table via PRAGMA table_info.
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (table_name,) in cur.fetchall():
        info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        if not info:
            continue
        cols = [(row[1], row[2]) for row in info]  # (name, type)
        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join(f'"{c[0]}"' for c in cols)
        insert_sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})'
        for i in range(n_rows):
            row = [_sample_for_column(ct, i, rng) for _, ct in cols]
            with contextlib.suppress(sqlite3.Error):
                conn.execute(insert_sql, row)
    conn.commit()


def _run(conn: sqlite3.Connection, sql: str, timeout_s: float = 2.0) -> tuple[bool, Any]:
    """Run `sql` and return (ok, rows_or_error). Result rows are sorted for
    order-agnostic comparison."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return False, "empty"
    try:
        # SQLite doesn't support per-statement timeouts cleanly; we set a
        # busy_timeout and rely on small synthetic tables to keep things fast.
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_s * 1000)}")
        rows = conn.execute(sql).fetchall()
        # Order-agnostic unless query uses ORDER BY (then order matters).
        if "order by" not in sql.lower():
            rows = sorted(rows, key=lambda r: tuple(repr(x) for x in r))
        return True, rows
    except sqlite3.Error as e:
        return False, f"sqlite_error: {e}"
    except Exception as e:
        return False, f"error: {e}"


@dataclass
class ExecResult:
    """Per-example outcome of the executable-accuracy check."""

    gold_ok: bool
    pred_ok: bool
    match: bool
    detail: str = ""


def executable_match(schema: str, gold: str, pred: str, n_rows: int = 5) -> ExecResult:
    """Run both queries on a freshly-populated in-memory SQLite DB.

    `match` is True iff both queries execute successfully AND their result-sets
    are equal. A gold query that itself fails to execute marks the example as
    unscoreable (gold_ok=False) — we still surface that so we can report
    coverage in the final table.
    """
    conn = sqlite3.connect(":memory:")
    try:
        _populate_sqlite(conn, schema, n_rows=n_rows)
        g_ok, g_rows = _run(conn, gold)
        p_ok, p_rows = _run(conn, pred)
        match = g_ok and p_ok and g_rows == p_rows
        detail = ""
        if not p_ok:
            detail = str(p_rows)
        elif not g_ok:
            detail = f"gold failed: {g_rows}"
        return ExecResult(gold_ok=g_ok, pred_ok=p_ok, match=match, detail=detail)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Convenience: aggregate metrics over a list of (schema, gold, pred)
# ---------------------------------------------------------------------------


def aggregate(
    schemas: list[str],
    golds: list[str],
    preds: list[str],
) -> dict[str, float]:
    """Compute exec-acc / EM / BLEU over a parallel list of examples.

    Returns a dict suitable for JSON / table rendering.
    """
    assert len(schemas) == len(golds) == len(preds), "Lists must align"
    n = len(preds)
    if n == 0:
        return {"n": 0, "exec_acc": 0.0, "exact_match": 0.0, "bleu": 0.0, "gold_coverage": 0.0}

    exec_hits = 0
    em_hits = 0
    gold_scoreable = 0

    for schema, g, p in zip(schemas, golds, preds, strict=True):
        res = executable_match(schema, g, p)
        if res.gold_ok:
            gold_scoreable += 1
            if res.match:
                exec_hits += 1
        if normalized_exact_match(g, p):
            em_hits += 1

    bleu = corpus_bleu(preds, golds)
    # Denominator for exec-acc is "examples where the gold itself runs" — that's
    # the cleanest number. We report gold_coverage so the reader sees both.
    denom = max(gold_scoreable, 1)
    return {
        "n": n,
        "exec_acc": round(100 * exec_hits / denom, 2),
        "exact_match": round(100 * em_hits / n, 2),
        "bleu": round(bleu, 2),
        "gold_coverage": round(100 * gold_scoreable / n, 2),
    }


# ---------------------------------------------------------------------------
# Post-processing: model output -> bare SQL string
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_sql_output(text: str) -> str:
    """Best-effort extraction of the SQL query from a free-form model response.

    Handles three failure modes we've seen in practice:
    - Models wrap output in ```sql ... ``` despite the system prompt.
    - Qwen3 sometimes emits a <think>...</think> block even with thinking
      disabled (rare, but cheap to strip).
    - Models append a trailing explanation after the query.

    The strategy: strip <think>, then prefer fenced content if present, then
    take the first CREATE/SELECT/WITH/INSERT/UPDATE/DELETE up to the first
    blank line.
    """
    if not text:
        return ""
    text = _THINK_RE.sub("", text).strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Stop at the first blank line — anything after that is usually prose.
    text = text.split("\n\n", 1)[0].strip()

    # Find the first SQL keyword and cut prefix prose ("Here's the SQL: ...").
    m = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start() :].strip()

    return text.rstrip(";").strip()
