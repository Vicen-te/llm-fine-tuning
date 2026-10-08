"""Tests for sql_ft.eval_sql — executable accuracy, exact-match, BLEU, output cleaning.

These tests run on CPU only (no torch, no transformers). They cover the
non-obvious branches:
- equivalent queries that differ only in whitespace/case → match
- semantically different queries → no match
- implicit-vs-explicit JOINs → match (because result-sets are identical)
- ORDER BY queries → order matters
- gold queries that don't execute → gold_ok=False (scored separately)
- output-cleaning edge cases (code fences, <think> blocks, prose prefix)
"""

from __future__ import annotations

import pytest

from sql_ft.eval_sql import (
    aggregate,
    clean_sql_output,
    corpus_bleu,
    executable_match,
    literals_by_column,
    loose_exact_match,
    normalized_exact_match,
)

# ---------------------------------------------------------------------------
# executable_match
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = "CREATE TABLE employees (id INT, name TEXT, salary REAL)"


class TestExecutableMatch:
    def test_identical_queries_match(self):
        r = executable_match(
            SIMPLE_SCHEMA, "SELECT AVG(salary) FROM employees", "SELECT AVG(salary) FROM employees"
        )
        assert r.gold_ok and r.pred_ok and r.match

    def test_whitespace_and_case_differences_match(self):
        r = executable_match(
            SIMPLE_SCHEMA,
            "SELECT AVG(salary) FROM employees",
            "select   avg(salary)   from   employees ;",
        )
        assert r.match

    def test_different_aggregates_do_not_match(self):
        r = executable_match(
            SIMPLE_SCHEMA, "SELECT AVG(salary) FROM employees", "SELECT MAX(salary) FROM employees"
        )
        assert r.gold_ok and r.pred_ok and not r.match

    def test_implicit_vs_explicit_join_match(self):
        """Two equivalent join formulations should produce identical result-sets."""
        schema = "CREATE TABLE a (id INT, x INT); CREATE TABLE b (id INT, y INT)"
        gold = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
        pred = "SELECT a.x, b.y FROM a, b WHERE a.id = b.id"
        r = executable_match(schema, gold, pred)
        assert r.match

    def test_order_by_makes_order_matter(self):
        r = executable_match(
            SIMPLE_SCHEMA,
            "SELECT name FROM employees ORDER BY salary ASC",
            "SELECT name FROM employees ORDER BY salary DESC",
        )
        # Different ordering → different result-set when ORDER BY is in both
        assert r.gold_ok and r.pred_ok and not r.match

    def test_invalid_pred_does_not_crash(self):
        r = executable_match(SIMPLE_SCHEMA, "SELECT name FROM employees", "this is not sql")
        assert r.gold_ok and not r.pred_ok and not r.match

    def test_invalid_gold_marks_unscoreable(self):
        r = executable_match(
            SIMPLE_SCHEMA, "SELECT name FROM nonexistent_table", "SELECT name FROM employees"
        )
        assert not r.gold_ok
        # `match` is False whenever either side failed.
        assert not r.match

    def test_literal_filter_is_empty_on_generic_rows(self):
        """Generic rows never contain 'Nintendo', so both queries return
        nothing and the plain metric counts a wrong filter as a hit."""
        gold = "SELECT salary FROM employees WHERE name = 'Nintendo'"
        pred = "SELECT salary FROM employees WHERE name = 'Sega'"
        r = executable_match(SIMPLE_SCHEMA, gold, pred)
        assert r.match and not r.gold_nonempty

    def test_seeded_rows_make_literal_filter_discriminative(self):
        gold = "SELECT salary FROM employees WHERE name = 'Nintendo'"
        same = executable_match(SIMPLE_SCHEMA, gold, gold, seed_literals=True)
        assert same.match and same.gold_nonempty
        wrong = executable_match(
            SIMPLE_SCHEMA,
            gold,
            "SELECT salary FROM employees WHERE name = 'Sega'",
            seed_literals=True,
        )
        assert not wrong.match
        lower = executable_match(
            SIMPLE_SCHEMA,
            gold,
            "SELECT salary FROM employees WHERE name = 'nintendo'",
            seed_literals=True,
        )
        assert not lower.match  # SQLite string equality is case-sensitive

    def test_seeded_rows_keep_numeric_comparisons_partial(self):
        gold = "SELECT COUNT(*) FROM employees WHERE id > 3"
        r = executable_match(
            SIMPLE_SCHEMA, gold, "SELECT COUNT(*) FROM employees", seed_literals=True
        )
        assert r.gold_nonempty and not r.match

    def test_count_of_nothing_is_not_nonempty(self):
        gold = "SELECT COUNT(*) FROM employees WHERE name = 'Nintendo'"
        assert not executable_match(SIMPLE_SCHEMA, gold, gold).gold_nonempty


class TestLiteralsByColumn:
    def test_equality_and_quoting_styles(self):
        sql = 'SELECT a FROM t WHERE "publisher" = "Nintendo" AND t.title = \'Mario\''
        assert literals_by_column(sql) == {"publisher": ["Nintendo"], "title": ["Mario"]}

    def test_numbers_get_neighbours(self):
        assert literals_by_column("SELECT a FROM t WHERE x > 3") == {"x": [3, 4, 2]}
        assert literals_by_column("SELECT a FROM t WHERE x >= 1.5") == {"x": [1.5, 2.5, 0.5]}

    def test_between_in_and_like(self):
        sql = "SELECT a FROM t WHERE y BETWEEN 1 AND 2 AND z IN ('p', 5) AND n LIKE '%bob%'"
        assert literals_by_column(sql) == {
            "y": [1, 2, 0, 3],
            "z": ["p", 5, 6, 4],
            "n": ["%bob%"],
        }

    def test_no_filters(self):
        assert literals_by_column("SELECT COUNT(*) FROM t") == {}
        assert literals_by_column("SELECT a FROM t WHERE COUNT(a) > 1") == {}


# ---------------------------------------------------------------------------
# normalized_exact_match
# ---------------------------------------------------------------------------


class TestNormalizedExactMatch:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("SELECT * FROM t", "select * from t"),
            ("SELECT * FROM t;", "SELECT * FROM t"),
            ("SELECT a,b FROM t", "SELECT a, b FROM t"),
        ],
    )
    def test_canonical_equivalents_match(self, a, b):
        assert normalized_exact_match(a, b)

    def test_different_queries_dont_match(self):
        assert not normalized_exact_match("SELECT a FROM t", "SELECT b FROM t")

    def test_unparseable_strings_dont_crash(self):
        assert not normalized_exact_match("not sql", "SELECT 1")
        assert not normalized_exact_match("", "")

    def test_quote_style_and_literal_case_are_strict(self):
        assert not normalized_exact_match(
            'SELECT a FROM t WHERE b = "X"', "SELECT a FROM t WHERE b = 'X'"
        )
        assert not normalized_exact_match(
            "SELECT a FROM t WHERE b = 'X'", "SELECT a FROM t WHERE b = 'x'"
        )


class TestLooseExactMatch:
    @pytest.mark.parametrize(
        "a,b",
        [
            ('SELECT a FROM t WHERE b = "Nintendo"', "SELECT a FROM t WHERE b = 'nintendo'"),
            ("select   a from t", "SELECT a FROM t;"),
            ("SELECT a FROM t WHERE b = `X`", "SELECT a FROM t WHERE b = 'x'"),
        ],
    )
    def test_folds_quotes_case_and_whitespace(self, a, b):
        assert loose_exact_match(a, b)

    def test_still_rejects_different_queries(self):
        assert not loose_exact_match("SELECT a FROM t", "SELECT b FROM t")
        assert not loose_exact_match(
            "SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'y'"
        )

    def test_empty_never_matches(self):
        assert not loose_exact_match("", "")


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


class TestBLEU:
    def test_identical_corpus_is_perfect(self):
        preds = ["SELECT a FROM t", "SELECT b FROM u"]
        refs = ["SELECT a FROM t", "SELECT b FROM u"]
        assert corpus_bleu(preds, refs) == pytest.approx(100.0, abs=0.01)

    def test_empty_predictions_is_zero(self):
        assert corpus_bleu([], []) == 0.0

    def test_completely_different_is_low(self):
        bleu = corpus_bleu(
            ["foo bar baz qux quux"],
            ["SELECT name FROM employees WHERE id = 1"],
        )
        assert bleu < 5.0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_all_correct_gives_100(self):
        schemas = [SIMPLE_SCHEMA] * 3
        golds = ["SELECT COUNT(*) FROM employees"] * 3
        preds = ["SELECT COUNT(*) FROM employees"] * 3
        out = aggregate(schemas, golds, preds)
        assert out["exec_acc"] == 100.0
        assert out["exact_match"] == 100.0
        assert out["bleu"] == pytest.approx(100.0, abs=0.01)
        assert out["gold_coverage"] == 100.0
        assert out["n"] == 3

    def test_mixed_correct_and_wrong(self):
        schemas = [SIMPLE_SCHEMA] * 2
        golds = ["SELECT COUNT(*) FROM employees", "SELECT AVG(salary) FROM employees"]
        preds = [
            "SELECT COUNT(*) FROM employees",  # correct
            "SELECT MAX(salary) FROM employees",
        ]  # wrong
        out = aggregate(schemas, golds, preds)
        assert out["exec_acc"] == 50.0
        assert out["exact_match"] == 50.0
        assert out["n"] == 2

    def test_seeded_and_loose_variants(self):
        schemas = [SIMPLE_SCHEMA] * 2
        golds = [
            'SELECT salary FROM employees WHERE name = "Nintendo"',
            'SELECT salary FROM employees WHERE name = "Sega"',
        ]
        preds = [
            "SELECT salary FROM employees WHERE name = 'nintendo'",  # same shape, wrong case
            "SELECT salary FROM employees WHERE name = 'Sega'",  # equivalent
        ]
        out = aggregate(schemas, golds, preds)
        assert out["exec_acc"] == 100.0  # both empty on generic rows
        assert out["exec_acc_seeded"] == 50.0  # case mismatch returns nothing
        assert out["exact_match"] == 0.0  # quote style differs
        assert out["exact_match_loose"] == 100.0
        assert out["gold_nonempty"] == 0.0
        assert out["gold_nonempty_seeded"] == 100.0

    def test_empty_inputs(self):
        out = aggregate([], [], [])
        assert out["n"] == 0
        assert out["exec_acc"] == 0.0
        assert out["exec_acc_seeded"] == 0.0
        assert out["exact_match_loose"] == 0.0


# ---------------------------------------------------------------------------
# clean_sql_output
# ---------------------------------------------------------------------------


class TestCleanSqlOutput:
    def test_plain_sql_passes_through(self):
        assert clean_sql_output("SELECT * FROM t") == "SELECT * FROM t"

    def test_strips_code_fences(self):
        raw = "```sql\nSELECT 1\n```"
        assert clean_sql_output(raw) == "SELECT 1"

    def test_strips_generic_fences(self):
        raw = "```\nSELECT 1\n```"
        assert clean_sql_output(raw) == "SELECT 1"

    def test_strips_think_block(self):
        raw = "<think>thinking out loud</think>\nSELECT name FROM users WHERE id = 1"
        assert clean_sql_output(raw) == "SELECT name FROM users WHERE id = 1"

    def test_strips_prose_prefix(self):
        raw = "Here is the SQL query you asked for:\nSELECT * FROM t"
        assert clean_sql_output(raw) == "SELECT * FROM t"

    def test_drops_trailing_prose(self):
        raw = "SELECT name FROM users\n\nThis returns all user names."
        assert clean_sql_output(raw) == "SELECT name FROM users"

    def test_strips_trailing_semicolon(self):
        assert clean_sql_output("SELECT 1;") == "SELECT 1"

    def test_empty_input_returns_empty(self):
        assert clean_sql_output("") == ""
        assert clean_sql_output("   ") == ""

    def test_with_starting_keyword(self):
        raw = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        assert clean_sql_output(raw) == "WITH cte AS (SELECT 1) SELECT * FROM cte"
