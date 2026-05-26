"""Benchmark constrained Decodra generation against unconstrained JSON attempts."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
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
    parser.add_argument("--passes", type=int, default=3, help="Measured benchmark passes.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup samples per schema/model pair.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for run ordering.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override schema-estimated token budgets for both benchmark sides.",
    )
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


def minimum(values: list[float]) -> float:
    """Return the minimum with an empty-list fallback."""

    return float(min(values)) if values else 0.0


def maximum(values: list[float]) -> float:
    """Return the maximum with an empty-list fallback."""

    return float(max(values)) if values else 0.0


@dataclass
class SideMetrics:
    """Collected metrics for one benchmark side."""

    valid: int = 0
    runtimes_ms: list[float] = field(default_factory=list)
    parse_validation_ms: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    completion_reasons: dict[str, int] = field(default_factory=dict)
    field_confidences: dict[str, list[float]] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def record_completion(self, reason: str) -> None:
        """Count one completion reason occurrence."""

        self.completion_reasons[reason] = self.completion_reasons.get(reason, 0) + 1

    def record_error(self, error: Exception) -> None:
        """Count one generation error by exception type."""

        error_name = type(error).__name__
        self.errors[error_name] = self.errors.get(error_name, 0) + 1


def summarize_side(metrics: SideMetrics, total_samples: int) -> dict[str, Any]:
    """Build a serializable summary for one benchmark side."""

    runtime_mean = mean(metrics.runtimes_ms)
    token_mean = mean([float(value) for value in metrics.token_counts])
    parse_mean = mean(metrics.parse_validation_ms)
    return {
        "first_pass_valid_rate": metrics.valid / total_samples if total_samples else 0.0,
        "runtime_ms_mean": runtime_mean,
        "runtime_ms_min": minimum(metrics.runtimes_ms),
        "runtime_ms_max": maximum(metrics.runtimes_ms),
        "parse_validation_ms_mean": parse_mean,
        "parse_validation_ms_min": minimum(metrics.parse_validation_ms),
        "parse_validation_ms_max": maximum(metrics.parse_validation_ms),
        "generated_tokens_mean": token_mean,
        "generated_tokens_min": int(min(metrics.token_counts)) if metrics.token_counts else 0,
        "generated_tokens_max": int(max(metrics.token_counts)) if metrics.token_counts else 0,
        "milliseconds_per_token": runtime_mean / token_mean if token_mean > 0.0 else 0.0,
        "tokens_per_second": (token_mean / runtime_mean) * 1000.0 if runtime_mean > 0.0 else 0.0,
        "completion_reasons": dict(metrics.completion_reasons),
        "generation_errors": dict(metrics.errors),
        "generation_error_count": sum(metrics.errors.values()),
    }


def write_results_snapshot(results: dict[str, Any], output: str) -> None:
    """Persist a benchmark snapshot so long CPU runs leave partial results."""

    output_path = ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


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
    passes = args.passes
    warmup = args.warmup

    if args.quick:
        models = ["gpt2"]
        schemas = BENCHMARK_SCHEMAS[:3]
        samples = 3
        passes = min(passes, 2)
        warmup = min(warmup, 1)

    all_results: dict[str, Any] = {
        "quick": args.quick,
        "samples_per_schema": samples,
        "passes": passes,
        "warmup_samples_per_schema": warmup,
        "seed": args.seed,
        "timing_boundaries": {
            "runtime_ms": (
                "Generation/decode runtime reported by DecodraEngine only. "
                "JSON parsing, regex extraction, and schema validation are timed separately."
            ),
            "parse_validation_ms": (
                "json.loads/regex extraction/Pydantic validation for unconstrained; "
                "Pydantic validation for constrained benchmark output."
            ),
        },
        "decoding_config": {
            "batch_size": 1,
            "do_sample": False,
            "temperature": None,
            "max_new_tokens_override": args.max_new_tokens,
            "constrained_stopping": "schema_complete_or_token_budget",
            "unconstrained_stopping": "eos_or_token_budget",
        },
        "models": {},
    }
    rng = random.Random(args.seed)

    for model_name in models:
        print(f"\nRunning model: {model_name}", flush=True)
        engine = DecodraEngine(model_name=model_name)
        model_rows: list[dict[str, Any]] = []

        for schema in schemas:
            constrained = SideMetrics()
            unconstrained = SideMetrics()
            token_budget = args.max_new_tokens or engine.schema_token_budget(schema)
            measured_samples = samples * passes

            for warmup_idx in range(warmup):
                prompt = (
                    "Extract the following information as strict JSON for "
                    f"{schema.__name__}. Warmup {warmup_idx + 1}:"
                )
                try:
                    engine.generate(
                        prompt=prompt,
                        schema=schema,
                        return_confidence=True,
                        max_new_tokens=token_budget,
                        measure_overhead=False,
                    )
                    engine.generate_unconstrained(prompt=prompt, max_new_tokens=token_budget)
                except RuntimeError as exc:
                    print(
                        f"  Warmup warning for {schema.__name__}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

            jobs = [
                (pass_idx, sample_idx, side)
                for pass_idx in range(passes)
                for sample_idx in range(samples)
                for side in ("constrained", "unconstrained")
            ]
            rng.shuffle(jobs)

            for pass_idx, sample_idx, side in jobs:
                prompt = (
                    "Extract the following information as strict JSON for "
                    f"{schema.__name__}. Pass {pass_idx + 1}, sample {sample_idx + 1}:"
                )

                if side == "constrained":
                    attempt_start = time.perf_counter()
                    try:
                        result = engine.generate(
                            prompt=prompt,
                            schema=schema,
                            return_confidence=True,
                            max_new_tokens=token_budget,
                            measure_overhead=False,
                        )
                    except RuntimeError as exc:
                        constrained.record_error(exc)
                        constrained.record_completion("generation_error")
                        constrained.runtimes_ms.append(
                            (time.perf_counter() - attempt_start) * 1000.0
                        )
                        constrained.token_counts.append(token_budget)
                        constrained.parse_validation_ms.append(0.0)
                        continue

                    validation_start = time.perf_counter()
                    try:
                        schema.model_validate(result["output"])
                        constrained.valid += 1
                    except ValidationError:
                        pass
                    constrained.parse_validation_ms.append(
                        (time.perf_counter() - validation_start) * 1000.0
                    )
                    constrained.runtimes_ms.append(float(result["_meta"]["constrained_time_ms"]))
                    constrained.token_counts.append(int(result["_meta"]["tokens_generated"]))
                    constrained.record_completion("schema_complete")
                    for field_name, confidence in result["_confidence"].items():
                        constrained.field_confidences.setdefault(field_name, []).append(
                            float(confidence)
                        )
                    continue

                unconstrained_result = engine.generate_unconstrained(
                    prompt=prompt,
                    max_new_tokens=token_budget,
                )
                validation_start = time.perf_counter()
                is_valid, _ = validate_json_text(unconstrained_result["text"], schema)
                unconstrained.parse_validation_ms.append(
                    (time.perf_counter() - validation_start) * 1000.0
                )
                unconstrained.valid += int(is_valid)
                unconstrained.runtimes_ms.append(float(unconstrained_result["time_ms"]))
                unconstrained.token_counts.append(int(unconstrained_result["tokens_generated"]))
                unconstrained.record_completion(
                    "eos"
                    if int(unconstrained_result["tokens_generated"]) < token_budget
                    else "token_budget_reached"
                )

            constrained_summary = summarize_side(constrained, measured_samples)
            unconstrained_summary = summarize_side(unconstrained, measured_samples)
            token_delta = (
                constrained_summary["generated_tokens_mean"]
                - unconstrained_summary["generated_tokens_mean"]
            )
            runtime_delta_ms = (
                constrained_summary["runtime_ms_mean"]
                - unconstrained_summary["runtime_ms_mean"]
            )
            efficiency_note = (
                "Constrained generation emitted fewer tokens on average; runtime delta "
                "should be interpreted as a stopping-efficiency effect, not negative overhead."
                if token_delta < 0
                else "Token counts are comparable; ms/token is the primary parity metric."
            )

            model_rows.append(
                {
                    "schema": schema.__name__,
                    "samples": samples,
                    "passes": passes,
                    "measured_samples": measured_samples,
                    "max_new_tokens": token_budget,
                    "constrained": {
                        **constrained_summary,
                        "mean_confidence_scores": {
                            field_name: mean(values)
                            for field_name, values in constrained.field_confidences.items()
                        },
                    },
                    "unconstrained": unconstrained_summary,
                    "token_count_delta": token_delta,
                    "runtime_delta_ms": runtime_delta_ms,
                    "generation_efficiency_note": efficiency_note,
                }
            )

            print(
                f"  {schema.__name__}: constrained_valid={constrained.valid}/{measured_samples}, "
                f"unconstrained_valid={unconstrained.valid}/{measured_samples}, "
                f"tokens={constrained_summary['generated_tokens_mean']:.1f}/"
                f"{unconstrained_summary['generated_tokens_mean']:.1f}, "
                f"ms/token={constrained_summary['milliseconds_per_token']:.1f}/"
                f"{unconstrained_summary['milliseconds_per_token']:.1f}"
                f", errors={constrained_summary['generation_error_count']}/"
                f"{unconstrained_summary['generation_error_count']}",
                flush=True,
            )

            all_results["models"][model_name] = model_rows
            write_results_snapshot(all_results, args.output)

        all_results["models"][model_name] = model_rows

    return all_results


def print_summary(results: dict[str, Any]) -> None:
    """Print the benchmark summary table."""

    print("\nConstrained")
    print("Model          | Schemas | Valid% | Avg Tokens | ms/token | Runtime(ms)")
    print("------------------------------------------------------------------")
    for model_name, rows in results["models"].items():
        valid = mean([row["constrained"]["first_pass_valid_rate"] for row in rows]) * 100.0
        tokens = mean([row["constrained"]["generated_tokens_mean"] for row in rows])
        ms_per_token = mean([row["constrained"]["milliseconds_per_token"] for row in rows])
        runtime = mean([row["constrained"]["runtime_ms_mean"] for row in rows])
        print(
            f"{model_name:<14} | {len(rows):>7} | {valid:>6.1f}% | "
            f"{tokens:>10.1f} | {ms_per_token:>8.1f} | {runtime:>11.1f}"
        )

    print("\nUnconstrained")
    print("Model          | Schemas | Valid% | Avg Tokens | ms/token | Runtime(ms)")
    print("------------------------------------------------------------------")
    for model_name, rows in results["models"].items():
        valid = mean([row["unconstrained"]["first_pass_valid_rate"] for row in rows]) * 100.0
        tokens = mean([row["unconstrained"]["generated_tokens_mean"] for row in rows])
        ms_per_token = mean([row["unconstrained"]["milliseconds_per_token"] for row in rows])
        runtime = mean([row["unconstrained"]["runtime_ms_mean"] for row in rows])
        print(
            f"{model_name:<14} | {len(rows):>7} | {valid:>6.1f}% | "
            f"{tokens:>10.1f} | {ms_per_token:>8.1f} | {runtime:>11.1f}"
        )


def main() -> None:
    """Run the benchmark and write results to disk."""

    args = parse_args()
    results = run_benchmark(args)
    write_results_snapshot(results, args.output)
    print_summary(results)
    print(f"\nSaved full results to {ROOT / args.output}")


if __name__ == "__main__":
    main()
