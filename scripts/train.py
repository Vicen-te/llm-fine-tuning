"""SFT with LoRA / QLoRA on Qwen3.5-2B for Text-to-SQL.

YAML-driven so the same script handles both regimes:
    python scripts/train.py --config configs/train_lora.yaml
    python scripts/train.py --config configs/train_qlora.yaml

Key implementation choices:
- TRL's `SFTTrainer` consumes our `text` column (full chat already rendered by
  the tokenizer's chat template) and handles tokenization + packing.
- For QLoRA we load the base in 4-bit NF4 via `BitsAndBytesConfig` and then
  call `prepare_model_for_kbit_training` to enable gradient checkpointing on
  quantized layers correctly.
- The adapter saved at the end is a small PEFT folder; we never modify the base
  weights. Merging is a separate step (`scripts/merge_adapter.py`).

Stack notes (transformers 5 / trl 1.x):
- `from_pretrained` takes `dtype=` (the old `torch_dtype=` is gone in v5) and
  defaults to `auto`; we set it explicitly.
- `SFTConfig` takes `max_length=` (renamed from `max_seq_length`).
- Qwen3.5 small models are MoE, so LoRA `target_modules` is set to `all-linear`
  in the configs and peft discovers the linear layers itself.
"""

from __future__ import annotations

import os
import sys

# On Windows, trl reads its bundled .jinja chat templates with the locale
# default encoding (cp1252), which chokes on their UTF-8 bytes and makes
# `import trl` raise UnicodeDecodeError. Re-exec under UTF-8 mode so training
# works without the caller having to set PYTHONUTF8=1 first.
if os.name == "nt" and not sys.flags.utf8_mode:
    import subprocess

    raise SystemExit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)

import argparse
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Make the package importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sql_ft.data import build_sft_dataset, read_jsonl
from sql_ft.tokens import pad_token_for

load_dotenv()  # load HF_TOKEN / HF_USERNAME from .env if present
console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="Path to YAML training config.")
    p.add_argument("--resume-from", default=None, help="Optional checkpoint dir to resume from.")
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_bnb_config(model_cfg: dict[str, Any]):
    """4-bit NF4 with double-quant + bf16 compute. Standard QLoRA recipe."""
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")]
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["train"]

    console.rule("[bold]Loading tokenizer & model[/bold]")
    console.log(f"base_model = {model_cfg['base_model']}  use_4bit = {model_cfg['use_4bit']}")

    # Heavy imports are deferred so `--help` stays cheap.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    # The pad token must differ from EOS: TRL's collator pads labels with -100
    # by position, but anything downstream that masks on the pad id would also
    # drop the turn-end token the model has to learn. Qwen ships <|endoftext|>
    # as pad and <|im_end|> as EOS; fall back to the former if pad is unset.
    tokenizer.pad_token = pad_token_for(tokenizer)
    console.log(f"eos = {tokenizer.eos_token!r}  pad = {tokenizer.pad_token!r}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    base_dtype = dtype_map[model_cfg.get("dtype", "bfloat16")]

    load_kwargs: dict[str, Any] = {
        "dtype": base_dtype,
        "device_map": "auto",
        "attn_implementation": model_cfg.get("attn_impl", "sdpa"),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if model_cfg.get("use_4bit", False):
        load_kwargs["quantization_config"] = _build_bnb_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **load_kwargs)
    # Disable HF cache during training (gradient checkpointing is incompatible).
    model.config.use_cache = False

    if model_cfg.get("use_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )

    # Attach LoRA --------------------------------------------------------
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Datasets -----------------------------------------------------------
    console.rule("[bold]Building datasets[/bold]")
    train_rows = read_jsonl(data_cfg["train_file"])
    eval_rows = read_jsonl(data_cfg["eval_file"])
    console.log(f"train rows = {len(train_rows)}   eval rows = {len(eval_rows)}")

    train_ds = build_sft_dataset(train_rows, tokenizer, enable_thinking=data_cfg["enable_thinking"])
    eval_ds = build_sft_dataset(eval_rows, tokenizer, enable_thinking=data_cfg["enable_thinking"])

    # Trainer ------------------------------------------------------------
    sft_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        report_to=train_cfg.get("report_to", "tensorboard"),
        # TRL-specific:
        max_length=data_cfg["max_seq_length"],
        packing=train_cfg.get("packing", False),
        dataset_text_field="text",
        loss_type=train_cfg.get("loss_type", "chunked_nll"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    console.rule("[bold]Training[/bold]")
    trainer.train(resume_from_checkpoint=args.resume_from)

    console.rule("[bold]Saving adapter[/bold]")
    out_dir = Path(train_cfg["output_dir"])
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    console.log(f"adapter saved to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
