# Decodra

Confidence-aware constrained decoding for structured language model generation.

Decodra combines finite-state constrained decoding with token-level confidence tracing to produce schema-valid outputs alongside calibrated field confidence scores.

Unlike traditional structured generation systems that only guarantee syntactic validity, Decodra exposes how strongly the model supported each generated field during inference.

```json
{
  "company_name": "Anthropic",
  "founded": 2021,
  "_confidence": {
    "company_name": 0.97,
    "founded": 0.81
  }
}
```

Built for:

* extraction pipelines
* agent systems
* evaluation infrastructure
* document processing
* reliability-sensitive LLM workflows

---

## Why Decodra

Most structured generation frameworks answer:

> “Did the output match the schema?”

Decodra also answers:

> “How hard did the constraint engine need to steer the model to get there?”

During constrained decoding, invalid token paths are masked at every generation step. Decodra records both:

* the model’s original token distribution,
* and the constrained distribution after masking.

Those signals are aggregated into per-field confidence scores without prompting the model to self-report uncertainty.

This allows downstream systems to:

* flag unreliable fields,
* selectively retry low-confidence regions,
* route outputs to stronger models,
* or surface human review only where needed.

---

## Features

* FSM-based constrained decoding for JSON generation
* Per-field confidence aggregation from token-level logits
* Pydantic schema support
* HuggingFace causal LM integration
* Structured benchmarking suite
* Local-first execution with no external APIs
* Benchmark tooling for constrained vs unconstrained generation

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Example

```python
from decodra import DecodraEngine
from schemas import CompanySchema

engine = DecodraEngine(model_name="gpt2")

result = engine.generate(
    prompt="Extract the company information",
    schema=CompanySchema,
    return_confidence=True
)

print(result)
```

---

## Example Output

```json
{
  "output": {
    "name": "Anthropic",
    "founded": 2021
  },
  "_confidence": {
    "name": 0.97,
    "founded": 0.81
  }
}
```

---

## Architecture

Decodra is organized around four core components:

| Component          | Responsibility                             |
| ------------------ | ------------------------------------------ |
| `engine.py`        | constrained decoding loop + logits masking |
| `confidence.py`    | token signal aggregation                   |
| `field_tracker.py` | JSON parser state + field attribution      |
| `schemas.py`       | benchmark schema suite                     |

The decoding engine:

1. records the raw token distribution,
2. applies FSM-derived token constraints,
3. measures constraint pressure,
4. and aggregates signals into field-level confidence scores.

---

## Benchmarks

Current benchmark goals:

| Metric                     | Target |
| -------------------------- | ------ |
| First-pass schema validity | 98%    |
| Unconstrained baseline     | ~76%   |
| Supported schemas          | 12+    |
| Model families             | 3      |
| Mean inference overhead    | ~14%   |

Benchmarks compare constrained decoding against unconstrained JSON generation without retries.

---

## Development

```bash
pytest tests/ -v
python experiments/baseline.py --quick
```

---

## Design Principles

* Deterministic generation over retry loops
* Confidence from inference mechanics, not self-reporting
* Local-first execution
* Explicit system behavior
* Reproducible benchmark-driven evaluation

---

## Status

Experimental and under active development.

---

## References

* [Outlines](https://github.com/outlines-dev/outlines?utm_source=chatgpt.com)
* [Structured Generation paper](https://arxiv.org/abs/2307.09702?utm_source=chatgpt.com)
* [HuggingFace Transformers](https://github.com/huggingface/transformers?utm_source=chatgpt.com)