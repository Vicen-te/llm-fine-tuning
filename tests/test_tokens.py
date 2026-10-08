"""Tests for sql_ft.tokens — the stop / pad / EOS bookkeeping.

These guard the defect where the fine-tune never stopped: the SFT text ended
on `<|im_end|>\\n` (so TRL appended a second EOS) and generation only stopped
on the model config's `<|endoftext|>`. Fake tokenizers keep this CPU-only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sql_ft.tokens import ensure_eos_ending, pad_token_for, stop_token_ids

VOCAB = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "SELECT": 3, "\n": 4}


def fake_tokenizer(eos: str = "<|im_end|>", pad: str | None = "<|endoftext|>", vocab=None):
    vocab = VOCAB if vocab is None else vocab
    return SimpleNamespace(
        eos_token=eos,
        eos_token_id=vocab.get(eos),
        pad_token=pad,
        pad_token_id=vocab.get(pad) if pad is not None else None,
        get_vocab=lambda: vocab,
    )


class TestStopTokenIds:
    def test_tokenizer_eos_comes_first(self):
        assert stop_token_ids(fake_tokenizer()) == [2]

    def test_keeps_model_config_ids(self):
        cfg = SimpleNamespace(eos_token_id=0)
        assert stop_token_ids(fake_tokenizer(), cfg) == [2, 0]

    def test_accepts_list_and_dedups(self):
        cfg = SimpleNamespace(eos_token_id=[2, 0, 0])
        assert stop_token_ids(fake_tokenizer(), cfg) == [2, 0]

    def test_missing_config_eos_is_ignored(self):
        assert stop_token_ids(fake_tokenizer(), SimpleNamespace(eos_token_id=None)) == [2]


class TestPadTokenFor:
    def test_keeps_distinct_pad(self):
        assert pad_token_for(fake_tokenizer()) == "<|endoftext|>"

    def test_unset_pad_falls_back_to_endoftext(self):
        assert pad_token_for(fake_tokenizer(pad=None)) == "<|endoftext|>"

    def test_pad_equal_to_eos_is_replaced(self):
        assert pad_token_for(fake_tokenizer(pad="<|im_end|>")) == "<|endoftext|>"

    def test_no_candidate_raises(self):
        tok = fake_tokenizer(eos="</s>", pad=None, vocab={"</s>": 0, "x": 1})
        with pytest.raises(ValueError):
            pad_token_for(tok)


class TestEnsureEosEnding:
    def test_strips_newline_after_eos(self):
        assert ensure_eos_ending("SELECT 1<|im_end|>\n", "<|im_end|>") == "SELECT 1<|im_end|>"

    def test_appends_missing_eos(self):
        assert ensure_eos_ending("SELECT 1\n", "<|im_end|>") == "SELECT 1<|im_end|>"

    def test_already_clean_is_unchanged(self):
        assert ensure_eos_ending("SELECT 1<|im_end|>", "<|im_end|>") == "SELECT 1<|im_end|>"
