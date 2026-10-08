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

## The vLLM patch (no longer needed)

The compose file now uses the stock `vllm/vllm-openai:v0.31.0` image with no
patch and no `--language-model-only`; the merged model loads as
`Qwen3_5ForCausalLM` and stops on `<|im_end|>`.

Released vLLM up to v0.26.0 shipped the text-only Qwen3.5 model classes but only
registered the multimodal `Qwen3_5ForConditionalGeneration`, so a text-only
checkpoint was routed through the Qwen3-VL path and crashed on the absent
`vision_config` (model build) and `video_token_id` (mrope) —
[vLLM #39231](https://github.com/vllm-project/vllm/issues/39231). Until then this
repo built a patched image on top of the official one that skipped the vision
tower when there was no `vision_config`, short-circuited the mrope position
computation for text-only input and served with `--language-model-only` (the
`Dockerfile.vllm` and `patch_vllm_qwen35.py` files remain in the git history).

The upstream fix is not this repo's work: a vLLM contributor added
`Qwen3_5ForCausalLM` to the text-generation registry in
[#50210](https://github.com/vllm-project/vllm/pull/50210), first released in
v0.27.0. What this repo contributed is the diagnosis of the two crash sites
above (also left as a comment on the earlier, unmerged attempt
[#39316](https://github.com/vllm-project/vllm/pull/39316)) and the patched image
that bridged the gap.

## Stop tokens

The merged checkpoint ships a `generation_config.json` listing both
`<|im_end|>` (the chat turn end, the tokenizer's EOS) and `<|endoftext|>` as
stop ids. Qwen3.5-2B itself ships none, so a model loaded from the base only
stops on `<|endoftext|>`, which the fine-tune never emits; vLLM also stops on
the tokenizer's EOS, so serving was unaffected, but a plain
`transformers.generate` call was not (see `docs/evaluation.md`).
