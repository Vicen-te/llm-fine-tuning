"""Inference helpers: HuggingFace (eager / 4-bit) and a thin vLLM HTTP client.

Two paths share the same prompt-building code:

- `HFGenerator` — used inside the evaluation and quantization scripts. Loads a
  model in fp16 (or 4-bit NF4 via bitsandbytes) directly via transformers.
  Fine for batch eval; not what you'd ship to production.

- `VLLMClient` — talks to a vLLM OpenAI-compatible HTTP server (the one we
  expose in `scripts/serve_vllm.py` / `docker/Dockerfile.vllm`). This is the
  production serving path.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from .prompts import build_messages

# ---------------------------------------------------------------------------
# Local HF inference
# ---------------------------------------------------------------------------


@dataclass
class GenConfig:
    """Generation hyperparameters. Greedy by default because SQL benefits from
    determinism — sampling temperatures > 0 introduce variability that hurts
    executable accuracy in our benchmarks."""

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class HFGenerator:
    """Lazy wrapper around a transformers model + tokenizer.

    Loading is done up front by the caller (so they can pick fp16 vs 4-bit) and
    passed in here. We only own the per-prompt encode/generate/decode loop.
    """

    def __init__(self, model: Any, tokenizer: Any, enable_thinking: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        # Qwen3 chat templates expect a known pad token; reuse eos if missing.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format(self, schema: str, question: str) -> str:
        msgs = build_messages(schema, question)
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    @contextmanager
    def _eval_mode(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    def generate(self, schema: str, question: str, cfg: GenConfig | None = None) -> str:
        """Generate one completion. Returns the decoded *new* tokens only."""
        import torch

        cfg = cfg or GenConfig()
        prompt = self._format(schema, question)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with self._eval_mode(), torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature if cfg.do_sample else 1.0,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def time_generate(
        self, schema: str, question: str, cfg: GenConfig | None = None
    ) -> tuple[str, float]:
        """Same as `generate`, but also returns wall-clock seconds for the call.
        Used by the quantization comparison to report latency.
        """
        t0 = time.perf_counter()
        text = self.generate(schema, question, cfg)
        return text, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# vLLM HTTP client (talks to scripts/serve_vllm.py or vLLM's OpenAI server)
# ---------------------------------------------------------------------------


class VLLMClient:
    """Minimal OpenAI-compatible chat-completions client for vLLM."""

    def __init__(
        self, base_url: str = "http://localhost:8000", model: str = "sql-ft", timeout: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def generate(self, schema: str, question: str, cfg: GenConfig | None = None) -> str:
        cfg = cfg or GenConfig()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(schema, question),
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature if cfg.do_sample else 0.0,
            "top_p": cfg.top_p,
        }
        r = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
