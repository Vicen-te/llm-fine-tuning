# Convenience shortcuts. Run `make help` for the list.
# All Python entrypoints live in scripts/ and accept --help.

PY := python
HF_USERNAME ?= Vicen-te
BASE_MODEL  ?= Qwen/Qwen3.5-2B
ADAPTER_DIR ?= outputs/qwen3.5-2b-sql-lora
MERGED_DIR  ?= outputs/qwen3.5-2b-sql-merged
QUANT_DIR   ?= $(MERGED_DIR)-nf4

# If the NF4 dir exists (i.e. `make quantize` has already run), include it as a
# third row in the eval table. Otherwise eval compares just base vs ft.
QUANT_FLAG := $(if $(wildcard $(QUANT_DIR)),--quantized-model $(QUANT_DIR),)

.PHONY: help install data train train-qlora merge evaluate quantize \
        push-dataset push-adapter push-merged serve-docker test lint fmt clean

help:
	@echo "Setup:"
	@echo "  install         pip install -r requirements.txt"
	@echo ""
	@echo "Pipeline:"
	@echo "  data            Build train/eval JSONL from b-mc2/sql-create-context"
	@echo "  train           LoRA SFT in bf16  (configs/train_lora.yaml)"
	@echo "  train-qlora     QLoRA SFT in 4-bit (configs/train_qlora.yaml)"
	@echo "  merge           Merge adapter into base model -> $(MERGED_DIR)"
	@echo "  evaluate        Run base vs ft vs ft-4bit on 50-example benchmark"
	@echo "  quantize        Save NF4 4-bit copy + size/latency comparison"
	@echo ""
	@echo "Publish:"
	@echo "  push-dataset    huggingface.co/datasets/$(HF_USERNAME)/sql-create-context-mini"
	@echo "  push-adapter    huggingface.co/$(HF_USERNAME)/qwen3.5-2b-sql-lora"
	@echo "  push-merged     huggingface.co/$(HF_USERNAME)/qwen3.5-2b-sql"
	@echo ""
	@echo "Serve:"
	@echo "  serve-docker    docker compose up the vLLM container on :8000"
	@echo ""
	@echo "Dev:"
	@echo "  test            Run the pytest suite (CPU-only, no GPU needed)"
	@echo "  lint            ruff check + ruff format --check"
	@echo "  fmt             ruff format (write changes)"
	@echo "  clean           Remove outputs/, evals/results/, __pycache__"

install:
	pip install -r requirements.txt

data:
	$(PY) scripts/prepare_dataset.py \
	    --source b-mc2/sql-create-context \
	    --out-dir data/processed \
	    --n-train 300 --n-eval 50 --seed 42

train:
	$(PY) scripts/train.py --config configs/train_lora.yaml

train-qlora:
	$(PY) scripts/train.py --config configs/train_qlora.yaml

merge:
	$(PY) scripts/merge_adapter.py \
	    --base-model $(BASE_MODEL) \
	    --adapter-dir $(ADAPTER_DIR) \
	    --output-dir $(MERGED_DIR)

evaluate:
	$(PY) scripts/evaluate.py \
	    --base-model $(BASE_MODEL) \
	    --merged-model $(MERGED_DIR) \
	    $(QUANT_FLAG) \
	    --eval-file data/processed/eval.jsonl \
	    --out-dir evals/results

quantize:
	$(PY) scripts/quantize_4bit.py \
	    --model $(MERGED_DIR) \
	    --output $(QUANT_DIR) \
	    --eval-file data/processed/eval.jsonl \
	    --report evals/results/quantization.json

push-dataset:
	$(PY) scripts/push_dataset_to_hub.py \
	    --train-file data/processed/train.jsonl \
	    --eval-file data/processed/eval.jsonl \
	    --repo-id $(HF_USERNAME)/sql-create-context-mini

push-adapter:
	$(PY) scripts/push_model_to_hub.py \
	    --model-dir $(ADAPTER_DIR) \
	    --repo-id $(HF_USERNAME)/qwen3.5-2b-sql-lora \
	    --kind adapter

push-merged:
	$(PY) scripts/push_model_to_hub.py \
	    --model-dir $(MERGED_DIR) \
	    --repo-id $(HF_USERNAME)/qwen3.5-2b-sql \
	    --kind merged

serve-docker:
	cd docker && docker compose up --build

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

fmt:
	$(PY) -m ruff format src tests scripts

clean:
	rm -rf outputs/ evals/results/ evals/predictions/ data/processed/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
