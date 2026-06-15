"""Compare a merged model in bf16 vs 4-bit NF4 (bitsandbytes).

Three numbers a deployment review will ask for:
1. **Disk footprint**  — how much smaller the quantized model is on disk.
2. **Latency**         — mean ms/example on the same prompts.
3. **Quality**         — executable accuracy + exact match + BLEU on the same
                          50-example benchmark, so degradation is visible.

We also persist the 4-bit weights to `--output` (using bitsandbytes save +
`save_pretrained`). The result is loadable later via
`AutoModelForCausalLM.from_pretrained(output_dir)` with the bnb config baked
into the saved `config.json` via `quantization_config`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_ft.data import read_jsonl
from sql_ft.eval_sql import aggregate, clean_sql_output
from sql_ft.inference import GenConfig, HFGenerator

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        required=True,
        help="Path or HF id of the *merged* fine-tuned model (bf16 weights).",
    )
    p.add_argument("--output", required=True, help="Where to save the NF4-quantized copy.")
    p.add_argument(
        "--eval-file", required=True, help="JSONL benchmark used for quality comparison."
    )
    p.add_argument(
        "--report",
        default="evals/results/quantization.json",
        help="Where to write the comparison JSON.",
    )
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--warmup", type=int, default=2)
    return p.parse_args()


def _dir_size_mb(path: Path) -> float:
    """Total bytes under `path` as MB."""
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 1)


def _generate_all(
    gen: HFGenerator, rows: list[dict[str, Any]], cfg: GenConfig
) -> tuple[list[str], list[float]]:
    """Run inference over every row. Returns (cleaned predictions, per-row seconds)."""
    preds: list[str] = []
    secs: list[float] = []
    for i, row in enumerate(rows, 1):
        text, dt = gen.time_generate(row["schema"], row["question"], cfg)
        preds.append(clean_sql_output(text))
        secs.append(dt)
        if i % 5 == 0 or i == len(rows):
            console.log(f"  {i:>3}/{len(rows)}  avg {sum(secs) / i:.3f}s/ex")
    return preds, secs


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = read_jsonl(args.eval_file)
    cfg = GenConfig(max_new_tokens=args.max_new_tokens, do_sample=False)

    # -----------------------------------------------------------------------
    # 1) Quantize and persist to disk
    # -----------------------------------------------------------------------
    console.rule("[bold]Quantizing to NF4[/bold]")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_4bit = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_4bit.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    src_size = _dir_size_mb(Path(args.model)) if Path(args.model).is_dir() else None
    nf4_size = _dir_size_mb(out_dir)

    # -----------------------------------------------------------------------
    # 2) Latency + quality on the 4-bit model
    # -----------------------------------------------------------------------
    console.rule("[bold]Eval 4-bit[/bold]")
    gen = HFGenerator(model_4bit, tok)

    # Warmup so CUDA kernels are JIT-compiled and don't skew the first sample.
    for _ in range(max(0, args.warmup)):
        gen.generate(rows[0]["schema"], rows[0]["question"], GenConfig(max_new_tokens=32))

    preds_4bit, secs_4bit = _generate_all(gen, rows, cfg)
    metrics_4bit = aggregate(
        [r["schema"] for r in rows],
        [r["answer"] for r in rows],
        preds_4bit,
    )
    # Free VRAM before loading the bf16 model.
    del model_4bit, gen
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 3) Latency + quality on the bf16 reference
    # -----------------------------------------------------------------------
    console.rule("[bold]Eval bf16 reference[/bold]")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    gen = HFGenerator(model_bf16, tok)
    for _ in range(max(0, args.warmup)):
        gen.generate(rows[0]["schema"], rows[0]["question"], GenConfig(max_new_tokens=32))

    preds_bf16, secs_bf16 = _generate_all(gen, rows, cfg)
    metrics_bf16 = aggregate(
        [r["schema"] for r in rows],
        [r["answer"] for r in rows],
        preds_bf16,
    )

    # -----------------------------------------------------------------------
    # 4) Render + persist comparison
    # -----------------------------------------------------------------------
    def mean(xs: list[float]) -> float:
        return round(1000 * sum(xs) / max(len(xs), 1), 1)

    report = {
        "model": str(args.model),
        "eval_file": args.eval_file,
        "n_examples": len(rows),
        "disk_mb": {"bf16": src_size, "nf4": nf4_size},
        "size_reduction_pct": (
            round(100 * (src_size - nf4_size) / src_size, 1)
            if src_size and nf4_size and src_size > 0
            else None
        ),
        "latency_ms_mean": {"bf16": mean(secs_bf16), "nf4": mean(secs_4bit)},
        "metrics": {"bf16": metrics_bf16, "nf4": metrics_4bit},
        "quality_delta": {
            "exec_acc": round(metrics_4bit["exec_acc"] - metrics_bf16["exec_acc"], 2),
            "exact_match": round(metrics_4bit["exact_match"] - metrics_bf16["exact_match"], 2),
            "bleu": round(metrics_4bit["bleu"] - metrics_bf16["bleu"], 2),
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "do_sample": False,
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = Table(title="bf16 vs 4-bit NF4")
    t.add_column("metric", style="cyan")
    t.add_column("bf16", justify="right")
    t.add_column("nf4 (4-bit)", justify="right")
    t.add_column("delta", justify="right")
    t.add_row(
        "disk (MB)",
        f"{src_size}" if src_size else "n/a",
        f"{nf4_size}",
        f"-{report['size_reduction_pct']}%" if report["size_reduction_pct"] is not None else "n/a",
    )
    t.add_row(
        "latency (ms/ex)",
        f"{report['latency_ms_mean']['bf16']}",
        f"{report['latency_ms_mean']['nf4']}",
        f"{round(report['latency_ms_mean']['nf4'] - report['latency_ms_mean']['bf16'], 1)}",
    )
    t.add_row(
        "exec_acc (%)",
        f"{metrics_bf16['exec_acc']}",
        f"{metrics_4bit['exec_acc']}",
        f"{report['quality_delta']['exec_acc']:+}",
    )
    t.add_row(
        "exact_match (%)",
        f"{metrics_bf16['exact_match']}",
        f"{metrics_4bit['exact_match']}",
        f"{report['quality_delta']['exact_match']:+}",
    )
    t.add_row(
        "BLEU",
        f"{metrics_bf16['bleu']}",
        f"{metrics_4bit['bleu']}",
        f"{report['quality_delta']['bleu']:+}",
    )
    console.print(t)
    console.log(f"report -> {args.report}")

    # If the source dir isn't a directory we can't auto-compute its size; emit
    # a warning so the README author knows to fill it in manually.
    if src_size is None:
        console.print(
            "[yellow]Note: --model is a Hub id, not a local dir, so disk size for the "
            "bf16 model is left blank in the report.[/yellow]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
