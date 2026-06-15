"""Evaluate base vs fine-tuned (and optionally 4-bit) on the 50-example benchmark.

Produces three artefacts under `evals/results/`:

- `eval.json`            — machine-readable summary (also uploaded with the model card).
- `eval.md`              — Markdown table to drop into the README.
- `predictions_<name>.jsonl` — per-example predictions for inspection.

Metrics (all from `sql_ft.eval_sql`):
- **Executable accuracy** — run gold + pred on an in-memory SQLite DB; compare result sets.
- **Exact match**         — sqlglot-canonicalised string equality.
- **BLEU**                — sacrebleu corpus BLEU on the raw SQL strings.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_ft.data import read_jsonl, write_jsonl
from sql_ft.eval_sql import aggregate, clean_sql_output
from sql_ft.inference import GenConfig, HFGenerator

console = Console()


@dataclass
class ModelSpec:
    """One model variant to evaluate."""

    name: str  # short label used in the table ("base", "ft", "ft-nf4")
    model_id: str  # local path or HF id
    load_in_4bit: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--base-model", required=True, help="HF id or local path of the base (unfine-tuned) model."
    )
    p.add_argument(
        "--merged-model", required=True, help="Local path of the merged fine-tuned model."
    )
    p.add_argument(
        "--quantized-model",
        default=None,
        help="Optional path to the NF4-quantized fine-tuned model "
        "(produced by scripts/quantize_4bit.py). If omitted we skip that row.",
    )
    p.add_argument(
        "--eval-file",
        required=True,
        help="Held-out benchmark JSONL with {schema, question, answer}.",
    )
    p.add_argument("--out-dir", default="evals/results")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    return p.parse_args()


def _build_model(spec: ModelSpec, dtype: str = "bfloat16") -> tuple[Any, Any]:
    """Return (model, tokenizer). Releases nothing — caller frees."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(spec.model_id, use_fast=True)
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if spec.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["dtype"] = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    return model, tok


def _run_model(
    spec: ModelSpec, rows: list[dict[str, Any]], cfg: GenConfig, warmup: int
) -> dict[str, Any]:
    """Load `spec`, generate predictions for every row, score, free.
    Returns a dict ready to merge into the final report."""
    import torch

    console.rule(f"[bold cyan]Model: {spec.name}[/bold cyan]")
    console.log(f"loading {spec.model_id} (4bit={spec.load_in_4bit}) ...")
    model, tok = _build_model(spec)
    gen = HFGenerator(model, tok)

    # Warmup hides one-time CUDA kernel JIT cost.
    for _ in range(max(0, warmup)):
        gen.generate(rows[0]["schema"], rows[0]["question"], GenConfig(max_new_tokens=32))

    preds_raw: list[str] = []
    preds_clean: list[str] = []
    secs: list[float] = []
    for i, r in enumerate(rows, 1):
        text, dt = gen.time_generate(r["schema"], r["question"], cfg)
        preds_raw.append(text)
        preds_clean.append(clean_sql_output(text))
        secs.append(dt)
        if i % 10 == 0 or i == len(rows):
            console.log(f"  {i:>3}/{len(rows)}  mean {sum(secs) / i:.2f}s/ex")

    metrics = aggregate(
        [r["schema"] for r in rows],
        [r["answer"] for r in rows],
        preds_clean,
    )
    avg_latency_ms = round(1000 * sum(secs) / max(len(secs), 1), 1)

    # Free VRAM before the next model loads.
    del model, gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "spec": asdict(spec),
        "metrics": metrics,
        "latency_ms_mean": avg_latency_ms,
        "predictions": [
            {"id": i, "question": r["question"], "gold": r["answer"], "pred_raw": pr, "pred": pc}
            for i, (r, pr, pc) in enumerate(zip(rows, preds_raw, preds_clean, strict=True))
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """Produce the Markdown table that drops straight into the README."""
    lines = [
        "# Evaluation report",
        "",
        f"- Benchmark: `{report['eval_file']}` ({report['n_examples']} examples)",
        f"- Generation: greedy (do_sample=False), max_new_tokens={report['max_new_tokens']}",
        "",
        "| Model | exec_acc % | exact_match % | BLEU | latency ms/ex |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in report["runs"]:
        m = r["metrics"]
        lines.append(
            f"| `{r['spec']['name']}` | "
            f"{m['exec_acc']} | {m['exact_match']} | {m['bleu']} | "
            f"{r['latency_ms_mean']} |"
        )
    lines.append("")
    lines.append(
        "Gold-coverage on this benchmark "
        f"(gold queries that themselves execute): "
        f"{report['runs'][0]['metrics']['gold_coverage']}%."
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    rows = read_jsonl(args.eval_file)
    if not rows:
        console.print(f"[red]No rows in {args.eval_file}[/red]")
        return 2

    specs = [
        ModelSpec("base", args.base_model, load_in_4bit=False),
        ModelSpec("ft", args.merged_model, load_in_4bit=False),
    ]
    if args.quantized_model:
        specs.append(ModelSpec("ft-nf4", args.quantized_model, load_in_4bit=True))

    cfg = GenConfig(max_new_tokens=args.max_new_tokens, do_sample=False)

    runs: list[dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    for spec in specs:
        run = _run_model(spec, rows, cfg, args.warmup)
        runs.append(run)
        # Per-model predictions to disk (we keep them out of the summary JSON).
        preds_file = out_dir / "predictions" / f"{spec.name}.jsonl"
        write_jsonl(preds_file, run.pop("predictions"))
        console.log(f"predictions -> {preds_file}")

    report = {
        "eval_file": args.eval_file,
        "n_examples": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "runs": runs,
    }

    (out_dir / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _render_markdown(report)
    (out_dir / "eval.md").write_text(md, encoding="utf-8")

    # Console summary -------------------------------------------------------
    t = Table(title="Evaluation summary")
    t.add_column("model", style="cyan")
    t.add_column("exec_acc %", justify="right")
    t.add_column("exact_match %", justify="right")
    t.add_column("BLEU", justify="right")
    t.add_column("latency ms/ex", justify="right")
    for r in runs:
        m = r["metrics"]
        t.add_row(
            r["spec"]["name"],
            str(m["exec_acc"]),
            str(m["exact_match"]),
            str(m["bleu"]),
            str(r["latency_ms_mean"]),
        )
    console.print(t)
    console.print(f"\nMarkdown report -> {out_dir / 'eval.md'}")
    console.print(f"JSON report     -> {out_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
