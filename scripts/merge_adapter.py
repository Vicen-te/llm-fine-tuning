"""Merge a trained LoRA adapter back into the base model weights.

The adapter (small PEFT folder) is what you train; the merged model is what
vLLM serves. vLLM does support LoRA adapters natively, but the merged path is
simpler in production and matches how Hugging Face Inference Endpoints expect a
model.

Usage:
    python scripts/merge_adapter.py \
        --base-model Qwen/Qwen3.5-2B \
        --adapter-dir outputs/qwen3.5-2b-sql-lora \
        --output-dir outputs/qwen3.5-2b-sql-merged

Stack note: transformers 5 takes `dtype=` (the old `torch_dtype=` is gone).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_ft.tokens import stop_token_ids

load_dotenv()  # load HF_TOKEN / HF_USERNAME from .env if present
console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    console.rule("[bold]Loading base + adapter[/bold]")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
    )
    base.config.use_cache = True
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)

    console.log("merging adapter into base weights ...")
    merged = peft_model.merge_and_unload()  # returns the base with LoRA folded in

    # Tokenizer was saved alongside the adapter — keep it co-located here too.
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)

    # The base ships no generation_config.json, so a loaded model only stops on
    # config.eos_token_id (<|endoftext|>); the fine-tune ends its turn with the
    # tokenizer's EOS (<|im_end|>) and never emits the other. Write both as stop
    # ids so transformers, vLLM and the Hub widget all stop at the turn end.
    merged.generation_config.eos_token_id = stop_token_ids(tok, merged.generation_config)
    merged.generation_config.pad_token_id = tok.pad_token_id
    console.log(f"generation_config eos_token_id = {merged.generation_config.eos_token_id}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    console.log(f"saving merged model to {out} ...")
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    console.print(f"[green]Done.[/green] Merged model + tokenizer at {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
