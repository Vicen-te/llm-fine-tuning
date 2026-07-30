# Serving

Two containers via Docker Compose:

- **vllm** — a vLLM OpenAI-compatible server on `:8000`.
- **api** — a FastAPI wrapper (`scripts/serve_vllm.py`) on `:8080` exposing
  `/sql` and proxying `/v1/chat/completions`.

## Run

```bash
cd docker
cp .env.example .env     # set MODEL_ID (a Hub id or a local /models path)
docker compose up --build
```

By default `MODEL_ID` points at the Hub model. To serve a locally merged model
without pushing first, set `MODEL_DIR=../outputs` and
`MODEL_ID=/models/qwen3.5-2b-sql-merged` in `.env`.

## Endpoints

- `POST /sql` — `{schema, question}` → `{sql, raw, latency_ms, usage}`. Rebuilds
  the training-time prompt and cleans the model output.
- `GET /healthz` — wrapper + vLLM health.
- `http://localhost:8080/docs` — interactive Swagger UI.
- vLLM's own OpenAI API is on `:8000` (`/v1/chat/completions`, `/v1/models`).

```bash
curl -X POST http://localhost:8080/sql \
  -H "Content-Type: application/json" \
  -d '{"schema":"CREATE TABLE t (id INT, city TEXT, pop INT)",
       "question":"Which cities have more than 1M people?"}'
```

## The vLLM patch

Every released vLLM up to v0.26.0 ships the text-only Qwen3.5 model classes but
only registers the multimodal `Qwen3_5ForConditionalGeneration`, so a text-only
checkpoint is routed through the Qwen3-VL path and crashes on the absent
`vision_config` (model build) and `video_token_id` (mrope) —
[vLLM #39231](https://github.com/vllm-project/vllm/issues/39231).

`docker/Dockerfile.vllm` builds on the official image and runs
`docker/patch_vllm_qwen35.py`, which skips the vision tower when there is no
`vision_config` and short-circuits the mrope position computation for text-only
input. The model is then served with `--language-model-only`.

The patch is a stop-gap. Upstream added `Qwen3_5ForCausalLM` to the
text-generation registry in
[#50210](https://github.com/vllm-project/vllm/pull/50210), merged to `main` two
days after v0.26.0 was cut. Once a release carries it, drop the `build:` block
from `docker/docker-compose.yml`, use `vllm/vllm-openai` directly and remove
`--language-model-only`.
