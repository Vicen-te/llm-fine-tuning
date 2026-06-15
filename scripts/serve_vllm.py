"""FastAPI wrapper around a vLLM OpenAI server.

We don't reimplement vLLM serving — that lives inside the `vllm/vllm-openai`
container (see `docker/docker-compose.yml`). This script is the thin
domain-aware layer in front of it:

- `GET  /healthz`    — proxied liveness check
- `GET  /models`     — proxied model list (vLLM `/v1/models`)
- `POST /v1/chat/completions` — straight passthrough so any OpenAI client works
- `POST /sql`        — domain endpoint: takes {schema, question}, returns
                       {sql, raw, latency_ms}. Builds the chat prompt with
                       the same `build_messages` used at training time, so the
                       prompt format matches what the model was fine-tuned on.

Why a sidecar instead of patching vLLM directly: vLLM ships an OpenAI-compatible
server that already handles PagedAttention, continuous batching, and streaming.
Replacing it with a hand-rolled inference loop would throw away the throughput
gains that are the whole point of using vLLM.

Run:
    uvicorn scripts.serve_vllm:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_ft.eval_sql import clean_sql_output
from sql_ft.prompts import build_messages

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "sql-ft")
REQUEST_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open one shared AsyncClient for the app's lifetime."""
    app.state.client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=REQUEST_TIMEOUT)
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="Qwen3-SQL serving wrapper",
    description="Thin FastAPI in front of a vLLM OpenAI server. "
    "Adds a domain-aware /sql endpoint for Text-to-SQL.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SQLRequest(BaseModel):
    schema_: str = Field(alias="schema", description="CREATE TABLE statement(s).")
    question: str = Field(description="Natural-language question.")
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    model_config = {"populate_by_name": True}


class SQLResponse(BaseModel):
    sql: str = Field(description="Cleaned SQL query (post-processed).")
    raw: str = Field(description="Raw model output, pre-cleaning.")
    latency_ms: float
    model: str
    usage: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness probe — proxies vLLM's health endpoint."""
    try:
        r = await app.state.client.get("/health")
        return {
            "status": "ok" if r.status_code == 200 else "degraded",
            "vllm_status": r.status_code,
        }
    except httpx.HTTPError as e:
        return {"status": "vllm_unreachable", "error": str(e)}


@app.get("/models")
async def models() -> Any:
    """List models the vLLM backend is serving."""
    try:
        r = await app.state.client.get("/v1/models")
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"vLLM backend unreachable: {e}") from e
    return r.json()


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any]) -> Any:
    """Passthrough to vLLM's OpenAI-compatible chat endpoint.

    Useful for integration with any OpenAI client library (point its base_url
    at this service and you're done).
    """
    try:
        r = await app.state.client.post("/v1/chat/completions", json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"vLLM backend unreachable: {e}") from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@app.post("/sql", response_model=SQLResponse)
async def sql(req: SQLRequest) -> SQLResponse:
    """Domain endpoint. Build a Text-to-SQL prompt with the exact format used
    at training time, send it to vLLM, post-process to a clean SQL string."""
    messages = build_messages(req.schema_, req.question)
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        # Greedy at temperature=0; vLLM ignores top_p in that case.
        # We don't expose stream=true here on purpose — the /sql endpoint is
        # a single-shot SQL string. Use /v1/chat/completions for streaming.
        "stream": False,
        # Disable Qwen3 thinking; the model was trained with enable_thinking=False
        # and we want raw SQL out.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.perf_counter()
    try:
        r = await app.state.client.post("/v1/chat/completions", json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"vLLM backend unreachable: {e}") from e
    latency_ms = round(1000 * (time.perf_counter() - t0), 1)

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    return SQLResponse(
        sql=clean_sql_output(raw),
        raw=raw,
        latency_ms=latency_ms,
        model=data.get("model", VLLM_MODEL),
        usage=data.get("usage"),
    )


if __name__ == "__main__":
    # `python scripts/serve_vllm.py` for quick local launch.
    import uvicorn

    uvicorn.run(
        "scripts.serve_vllm:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
    )
