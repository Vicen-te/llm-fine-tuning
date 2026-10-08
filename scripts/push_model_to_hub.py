"""Push either a LoRA adapter or a merged model to the Hugging Face Hub.

The two paths share most logic but emit different model cards:

- `--kind adapter` — small repo with adapter_model.safetensors, adapter_config.json,
  tokenizer files. Users load it on top of the base via `PeftModel.from_pretrained`.
- `--kind merged`  — full standalone model, drop-in for vLLM / transformers.

In both cases the model card declares the base model, training data, and
evaluation results, so the model page is self-describing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

console = Console()
load_dotenv()


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------

ADAPTER_CARD = """\
---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
tags:
- lora
- peft
- text2sql
- sql
- qwen3.5
- fine-tuning
language:
- en
pipeline_tag: text-generation
datasets:
- {dataset_id}
---

# Qwen3.5-2B · SQL LoRA adapter

LoRA adapter trained on top of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
for Text-to-SQL generation.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype="auto", device_map="auto")
tok  = AutoTokenizer.from_pretrained("{repo_id}")
model = PeftModel.from_pretrained(base, "{repo_id}")

messages = [
    {{"role": "system", "content": "You are a precise Text-to-SQL assistant. Output only the SQL query."}},
    {{"role": "user", "content": "### Schema\\nCREATE TABLE employees (id INT, name TEXT, salary REAL)\\n\\n### Question\\nWhat is the average salary?\\n\\n### SQL"}},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok([text], return_tensors="pt").to(model.device)
# Qwen3.5-2B ships no generation_config.json, so stop on the chat turn end explicitly.
out = model.generate(**inputs, max_new_tokens=128, do_sample=False, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True))
```

## Training

- **Base model**: Qwen/Qwen3.5-2B (instruction-tuned, thinking mode disabled for SQL)
- **Method**: LoRA (rank=16, α=32, dropout=0.05) on all linear layers
- **Dataset**: {dataset_id} — 300 train / 200 eval examples
- **Hardware**: single GPU, bf16, 3 epochs, effective batch 16, cosine LR 2e-4
- **Trainer**: TRL `SFTTrainer`

## Evaluation

See the [project repo](https://github.com/Vicen-te/llm-fine-tuning)
for the full evaluation report (executable accuracy, exact match, BLEU)
against the same base model on a held-out 200-example split.

## Limitations

- SQLite-flavoured SQL only; other dialects untested.
- The training set is intentionally small (300 rows); this is a small-scale
  fine-tune, not a production-grade Text-to-SQL system.
"""

MERGED_CARD = """\
---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: transformers
tags:
- text2sql
- sql
- qwen3.5
- fine-tuned
language:
- en
pipeline_tag: text-generation
datasets:
- {dataset_id}
---

# Qwen3.5-2B · SQL (merged)

[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) with a LoRA SQL
adapter merged in. Drop-in replacement for the base — same architecture, same
tokenizer, no PEFT runtime dependency.

## Usage with transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}", dtype="auto", device_map="auto")
```

The checkpoint ships a `generation_config.json` that stops on both
`<|im_end|>` (the chat turn end) and `<|endoftext|>`; the base model lists only
the latter, which a fine-tune on chat-formatted targets never emits.

## Usage with vLLM

```bash
vllm serve {repo_id} --max-model-len 4096 --served-model-name sql-ft
```

## Training

- **Base model**: Qwen/Qwen3.5-2B
- **Method**: LoRA (rank=16, α=32) → merged via `peft.merge_and_unload()`
- **Dataset**: {dataset_id} — 300 train / 200 eval
- **Recipe**: 3 epochs, bf16, effective batch 16, cosine LR 2e-4

## Evaluation

Compared against the base model on a held-out 200-example split. See the
[project repo](https://github.com/Vicen-te/llm-fine-tuning) for the
full report (executable accuracy, exact match, BLEU, latency, 4-bit
quantization trade-off).

## License

Apache 2.0, inherited from the base model.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", required=True, help="Local folder to upload.")
    p.add_argument("--repo-id", required=True, help="e.g. YOUR_HF_USERNAME/qwen3.5-2b-sql-lora")
    p.add_argument("--kind", choices=["adapter", "merged"], required=True)
    p.add_argument(
        "--dataset-id", default=None, help="Defaults to <repo-owner>/sql-create-context-mini"
    )
    p.add_argument("--private", action="store_true")
    p.add_argument("--commit-message", default="Upload model")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]HF_TOKEN not set (see .env.example).[/red]")
        return 2

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        console.print(f"[red]{model_dir} does not exist.[/red]")
        return 2

    # Write the README into the upload folder so it ships with the model.
    card_template = ADAPTER_CARD if args.kind == "adapter" else MERGED_CARD
    dataset_id = args.dataset_id or f"{args.repo_id.split('/')[0]}/sql-create-context-mini"
    card = card_template.format(repo_id=args.repo_id, dataset_id=dataset_id)
    # Normalize to LF + UTF-8 so the YAML front matter parses on the Hub; the
    # card string inherits this module's CRLF endings on Windows, which a default
    # text-mode write would turn into \r\r\n and break the parser.
    card = card.replace("\r\n", "\n").replace("\r", "\n")
    (model_dir / "README.md").write_text(card, encoding="utf-8", newline="\n")

    # Attach the eval summary if it exists so the headline metrics live on
    # the model page itself, not just in the repo.
    results_path = Path("evals/results/eval.json")
    if results_path.exists():
        summary = json.loads(results_path.read_text(encoding="utf-8"))
        (model_dir / "eval_results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    console.log(f"uploading {model_dir} -> {args.repo_id} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        token=token,
        # The trainer leaves intermediate checkpoints (with optimizer state) and
        # TensorBoard logs in the adapter folder; only the final model belongs
        # on the Hub.
        ignore_patterns=["checkpoint-*", "checkpoint-*/**", "runs/**"],
    )
    console.print(f"[green]Pushed to https://huggingface.co/{args.repo_id}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
