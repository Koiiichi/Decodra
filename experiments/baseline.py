"""Benchmark constrained Decodra generation against unconstrained JSON attempts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, ValidationError

from src.engine import DecodraEngine
from src.schemas import BENCHMARK_SCHEMAS


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line flags."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run 3 schemas x 1 model x 3 samples.")
    parser.add_argument("--samples", type=int, default=10, help="Samples per schema/model pair.")
    parser.add_argument("--output", default="results/benchmark_results.json", help="Result JSON path.")
    return parser.parse_args()


def validate_json_text(text: str, schema: type[BaseModel]) -> tuple[bool, dict[str, Any] | None]:
    """Validate raw generated text as first-pass JSON for ``schema``."""

    candidate: Any | None = None
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return False, None
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            return False, None

    try:
        validated = schema.model_validate(candidate)
    except ValidationError:
        return False, None
    return True, validated.model_dump(mode="json")


def mean(values: list[float]) -> float:
    """Return a numeric mean with an empty-list fallback."""

    return float(statistics.mean(values)) if values else 0.0


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the configured benchmark and return serializable results."""

    try:
        import torch

        if not torch.cuda.is_available():
            print("Warning: GPU unavailable; running benchmark on CPU.")
    except Exception:
        print("Warning: torch unavailable before engine load; benchmark may fail.")

    models = ["gpt2", "distilgpt2", "gpt2-medium"]
    schemas = BENCHMARK_SCHEMAS
    samples = args.samples

    if args.quick:
        models = ["gpt2"]
        schemas = BENCHMARK_SCHEMAS[:3]
        samples = 3

    all_results: dict[str, Any] = {
        "quick": args.quick,
        "samples_per_schema": samples,
        "models": {},
    }

    for model_name in models:
        print(f"\nRunning model: {model_name}")
        engine = DecodraEngine(model_name=model_name)
        model_rows: list[dict[str, Any]] = []

        for schema in schemas:
            constrained_valid = 0
            unconstrained_valid = 0
            constrained_times: list[float] = []
            unconstrained_times: list[float] = []
            overheads: list[float] = []
            field_confidences: dict[str, list[float]] = {}
            token_budget = engine.schema_token_budget(schema)

            for sample_idx in range(samples):
                prompt = (
                    "Extract the following information as strict JSON for "
                    f"{schema.__name__}. Sample {sample_idx + 1}:"
                )

                result = engine.generate(
                    prompt=prompt,
                    schema=schema,
                    return_confidence=True,
                    measure_overhead=False,
                )
                try:
                    schema.model_validate(result["output"])
                    constrained_valid += 1
                except ValidationError:
                    pass

                constrained_times.append(float(result["_meta"]["constrained_time_ms"]))
                for field_name, confidence in result["_confidence"].items():
                    field_confidences.setdefault(field_name, []).append(float(confidence))

                unconstrained = engine.generate_unconstrained(
                    prompt=prompt,
                    max_new_tokens=token_budget,
                )
                is_valid, _ = validate_json_text(unconstrained["text"], schema)
                unconstrained_valid += int(is_valid)
                unconstrained_times.append(float(unconstrained["time_ms"]))

            constrained_mean = mean(constrained_times)
            unconstrained_mean = mean(unconstrained_times)
            overhead_pct = (
                ((constrained_mean - unconstrained_mean) / unconstrained_mean) * 100.0
                if unconstrained_mean > 0.0
                else 0.0
            )
            overheads.append(overhead_pct)

            model_rows.append(
                {
                    "schema": schema.__name__,
                    "samples": samples,
                    "first_pass_valid_rate_constrained": constrained_valid / samples,
                    "first_pass_valid_rate_unconstrained": unconstrained_valid / samples,
                    "mean_confidence_scores": {
                        field_name: mean(values)
                        for field_name, values in field_confidences.items()
                    },
                    "constrained_time_ms": constrained_mean,
                    "unconstrained_time_ms": unconstrained_mean,
                    "overhead_pct": overhead_pct,
                }
            )

            print(
                f"  {schema.__name__}: constrained={constrained_valid}/{samples}, "
                f"unconstrained={unconstrained_valid}/{samples}, overhead={overhead_pct:.1f}%"
            )

        all_results["models"][model_name] = model_rows

    return all_results


def print_summary(results: dict[str, Any]) -> None:
    """Print the benchmark summary table."""

    print("\nModel          | Schemas | Constrained Valid% | Unconstrained Valid% | Overhead%")
    print("--------------------------------------------------------------------------")
    for model_name, rows in results["models"].items():
        constrained = mean([row["first_pass_valid_rate_constrained"] for row in rows]) * 100.0
        unconstrained = mean([row["first_pass_valid_rate_unconstrained"] for row in rows]) * 100.0
        overhead = mean([row["overhead_pct"] for row in rows])
        print(
            f"{model_name:<14} | {len(rows):>7} | {constrained:>18.1f}% | "
            f"{unconstrained:>20.1f}% | {overhead:>8.1f}%"
        )


def main() -> None:
    """Run the benchmark and write results to disk."""

    args = parse_args()
    results = run_benchmark(args)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()
