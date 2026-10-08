"""Special-token bookkeeping shared by training, merging and inference.

Qwen tokenizers use two different end tokens: `<|im_end|>` closes a chat turn
and is the tokenizer's `eos_token`; `<|endoftext|>` is the padding token and
the only `eos_token_id` in Qwen3.5-2B's `config.json` (the checkpoint ships no
`generation_config.json`). An SFT target therefore ends on `<|im_end|>`, and
generation has to stop on it too — otherwise the fine-tune, which never emits
`<|endoftext|>`, runs to `max_new_tokens` after every query.

Pure Python, no torch / transformers import, so the CPU test suite covers it.
"""

from __future__ import annotations

from typing import Any


def stop_token_ids(tokenizer: Any, generation_config: Any = None) -> list[int]:
    """Token ids generation must stop on: the tokenizer's EOS (the chat turn
    end) first, then whatever the model's generation config already lists."""
    extra = getattr(generation_config, "eos_token_id", None)
    extra = [extra] if isinstance(extra, int) else list(extra or [])
    ids: list[int] = []
    for tid in [tokenizer.eos_token_id, *extra]:
        if tid is not None and tid not in ids:
            ids.append(tid)
    return ids


def pad_token_for(tokenizer: Any) -> str:
    """A padding token distinct from EOS.

    Keeps the tokenizer's own pad token when it has one that is not the EOS
    token; otherwise falls back to `<|endoftext|>` (Qwen's padding token) or a
    conventional pad token present in the vocabulary. Fails loudly rather than
    aliasing pad to EOS, which lets a collator that masks on the pad id drop the
    turn-end token the model has to learn.
    """
    eos = tokenizer.eos_token
    pad = tokenizer.pad_token
    if pad is not None and pad != eos:
        return pad
    vocab = tokenizer.get_vocab()
    for candidate in ("<|endoftext|>", "<pad>", "[PAD]"):
        if candidate in vocab and candidate != eos:
            return candidate
    raise ValueError(f"no pad token distinct from eos {eos!r}; add one to the tokenizer")


def ensure_eos_ending(text: str, eos: str) -> str:
    """Make `text` end exactly on `eos`: strip whitespace after a trailing
    `eos`, or append it when the template left it out."""
    stripped = text.rstrip()
    if stripped.endswith(eos):
        return stripped
    return stripped + eos
