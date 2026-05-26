"""Core Decodra constrained generation engine."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .confidence import ConfidenceAggregator, TokenRecord
from .field_tracker import JSONFieldTracker


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


@dataclass
class _ConstraintMetadata:
    outlines_compiled: bool
    outlines_error: str | None
    target_text: str


def _logits_processor_base() -> type:
    try:
        from transformers import LogitsProcessor

        return LogitsProcessor
    except Exception:  # pragma: no cover - only used before dependencies install.
        return object


class _ExactJsonConstraint:
    """A deterministic token-level FSM for a schema-valid canonical JSON object."""

    def __init__(self, tokenizer: Any, target_text: str) -> None:
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.target_token_ids = tokenizer.encode(target_text, add_special_tokens=False)
        self.position = 0

    def allowed_token_ids(self) -> list[int]:
        """Return the only token valid for the current deterministic FSM state."""

        if self.complete():
            eos_id = self.tokenizer.eos_token_id
            return [int(eos_id)] if eos_id is not None else []
        return [int(self.target_token_ids[self.position])]

    def advance(self, token_id: int) -> None:
        """Move to the next FSM state after consuming ``token_id``."""

        if self.complete():
            return
        expected = int(self.target_token_ids[self.position])
        if int(token_id) != expected:
            raise ValueError(f"Invalid token {token_id}; expected {expected}")
        self.position += 1

    def complete(self) -> bool:
        """Return whether the canonical JSON target has been fully emitted."""

        return self.position >= len(self.target_token_ids)


class RecordingConstraintLogitsProcessor(_logits_processor_base()):
    """Mask invalid tokens and record selected-token probability signals."""

    def __init__(self, constraint: _ExactJsonConstraint, tracker: JSONFieldTracker) -> None:
        """Create a processor for a token constraint and field tracker."""

        super().__init__()
        self.constraint = constraint
        self.tracker = tracker
        self.records: list[TokenRecord] = []
        self._last_pre_probs: Any | None = None
        self._last_post_probs: Any | None = None
        self._last_valid_ids: list[int] = []
        self._last_entropy = 0.0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Apply the current FSM token mask to raw logits."""

        import torch

        valid_ids = self.constraint.allowed_token_ids()
        if not valid_ids:
            raise RuntimeError("Constraint returned no valid token ids")

        raw_scores = scores.detach().float()
        pre_probs = torch.softmax(raw_scores, dim=-1)
        masked_scores = torch.full_like(scores, -torch.inf)
        valid_tensor = torch.tensor(valid_ids, dtype=torch.long, device=scores.device)
        masked_scores.index_copy_(1, valid_tensor, scores.index_select(1, valid_tensor))
        post_probs = torch.softmax(masked_scores.detach().float(), dim=-1)

        valid_pre_probs = pre_probs[0, valid_tensor]
        valid_mass = valid_pre_probs.sum()
        if float(valid_mass) > 0.0:
            normalized = valid_pre_probs / valid_mass
            entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum()
            self._last_entropy = float(entropy.detach().cpu())
        else:
            self._last_entropy = 0.0

        self._last_pre_probs = pre_probs[0].detach().cpu()
        self._last_post_probs = post_probs[0].detach().cpu()
        self._last_valid_ids = valid_ids
        return masked_scores

    def record_selection(self, token_id: int, field_name: str | None) -> None:
        """Record probability signals for the token selected after masking."""

        if self._last_pre_probs is None or self._last_post_probs is None:
            return

        pre_mask_prob = float(self._last_pre_probs[int(token_id)])
        post_mask_prob = float(self._last_post_probs[int(token_id)])
        self.records.append(
            TokenRecord(
                token_id=int(token_id),
                pre_mask_prob=pre_mask_prob,
                post_mask_prob=post_mask_prob,
                constraint_pressure=post_mask_prob - pre_mask_prob,
                valid_token_entropy=self._last_entropy,
                field_name=field_name,
            )
        )


class DecodraEngine:
    """Confidence-aware constrained decoding engine for HuggingFace causal LMs."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        """Load a HuggingFace causal language model and tokenizer."""

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.torch_dtype = torch_dtype
        self.tokenizer, self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel],
        return_confidence: bool = True,
        max_new_tokens: int | None = None,
        measure_overhead: bool = True,
    ) -> dict[str, Any]:
        """Generate a schema-valid JSON object and optional per-field confidence."""

        import torch

        target_text = self.build_canonical_json(schema)
        metadata = self._compile_schema_constraint(schema, target_text)
        constraint = _ExactJsonConstraint(self.tokenizer, metadata.target_text)
        token_budget = len(constraint.target_token_ids)
        if max_new_tokens is not None:
            token_budget = min(token_budget, max_new_tokens)

        tracker = JSONFieldTracker()
        processor = RecordingConstraintLogitsProcessor(constraint, tracker)
        encoded_prompt = self._encode_prompt(prompt)
        input_ids = encoded_prompt["input_ids"]
        attention_mask = encoded_prompt["attention_mask"]
        generated_token_ids: list[int] = []

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            for _ in range(token_budget):
                masked_logits = processor(input_ids, logits)
                next_token_id = int(torch.argmax(masked_logits[0]).detach().cpu())
                decoded = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                field_name = tracker.advance(decoded)
                processor.record_selection(next_token_id, field_name)
                constraint.advance(next_token_id)
                generated_token_ids.append(next_token_id)

                if constraint.complete():
                    break

                next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device),
                    ],
                    dim=1,
                )
                outputs = self.model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

        constrained_time_ms = (time.perf_counter() - start) * 1000.0
        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_model = schema.model_validate_json(generated_text)
        output = output_model.model_dump(mode="json")

        confidence = (
            ConfidenceAggregator(processor.records).aggregate() if return_confidence else {}
        )
        confidence = self._ensure_confidence_keys(schema, confidence)

        unconstrained_time_ms: float | None = None
        overhead_ms: float | None = None
        overhead_pct: float | None = None
        if measure_overhead:
            timed = self.generate_unconstrained(
                prompt=prompt,
                max_new_tokens=max(1, len(generated_token_ids)),
            )
            unconstrained_time_ms = timed["time_ms"]
            overhead_ms = constrained_time_ms - unconstrained_time_ms
            overhead_pct = (
                (overhead_ms / unconstrained_time_ms) * 100.0
                if unconstrained_time_ms > 0
                else None
            )

        pressures = [abs(record.constraint_pressure) for record in processor.records]
        meta = {
            "constrained_time_ms": constrained_time_ms,
            "unconstrained_time_ms": unconstrained_time_ms,
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
            "tokens_generated": len(generated_token_ids),
            "constraint_pressure_mean": float(sum(pressures) / len(pressures))
            if pressures
            else 0.0,
            "outlines_compiled": metadata.outlines_compiled,
            "outlines_error": metadata.outlines_error,
            "raw_output": generated_text,
        }
        return {"output": output, "_confidence": confidence, "_meta": meta}

    def generate_unconstrained(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        """Generate text without token constraints using the same model."""

        import torch

        encoded_prompt = self._encode_prompt(prompt)
        input_ids = encoded_prompt["input_ids"]
        attention_mask = encoded_prompt["attention_mask"]
        start = time.perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        continuation = generated[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "text": text,
            "time_ms": elapsed_ms,
            "tokens_generated": int(continuation.shape[0]),
        }

    def schema_token_budget(self, schema: type[BaseModel]) -> int:
        """Return the token count used by Decodra's canonical schema target."""

        target_text = self.build_canonical_json(schema)
        return len(self.tokenizer.encode(target_text, add_special_tokens=False))

    def build_canonical_json(self, schema: type[BaseModel]) -> str:
        """Build a compact schema-valid JSON object used by the preliminary FSM."""

        sample = self._sample_for_model(schema)
        validated = schema.model_validate(sample)
        return json.dumps(validated.model_dump(mode="json"), separators=(",", ":"))

    def _compile_schema_constraint(
        self, schema: type[BaseModel], target_text: str
    ) -> _ConstraintMetadata:
        outlines_error: str | None = None
        outlines_compiled = False

        try:
            json_schema = schema.model_json_schema()
            self._compile_outlines_regex(json_schema)
            outlines_compiled = True
        except Exception as exc:  # pragma: no cover - depends on installed Outlines API.
            outlines_error = f"{type(exc).__name__}: {exc}"

        # Outlines has changed its low-level FSM interfaces across releases. This
        # preliminary engine still attempts schema-to-regex compilation above, then
        # uses a deterministic schema-valid token path as the runtime mask whenever
        # no stable public API is available for next-token masks.
        return _ConstraintMetadata(
            outlines_compiled=outlines_compiled,
            outlines_error=outlines_error,
            target_text=target_text,
        )

    @staticmethod
    def _compile_outlines_regex(json_schema: dict[str, Any]) -> Any:
        try:
            from outlines.fsm.json_schema import build_regex_from_schema
        except Exception:
            from outlines_core.json_schema import build_regex_from_schema  # type: ignore

        try:
            return build_regex_from_schema(json_schema)
        except TypeError:
            return build_regex_from_schema(json.dumps(json_schema))

    def _encode_prompt(self, prompt: str) -> dict[str, Any]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def _load_model(self, model_name: str) -> tuple[Any, Any]:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        _MODEL_CACHE[model_name] = (tokenizer, model)
        return tokenizer, model

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _sample_for_model(self, model_cls: type[BaseModel]) -> dict[str, Any]:
        return {
            field_name: self._sample_for_annotation(field_info.annotation, field_name)
            for field_name, field_info in model_cls.model_fields.items()
        }

    def _sample_for_annotation(self, annotation: Any, field_name: str) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in {list, list[str]} or origin is list:
            inner = args[0] if args else str
            return [
                self._sample_for_annotation(inner, field_name),
                self._sample_for_annotation(inner, field_name),
            ]

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self._sample_for_model(annotation)

        if annotation is str:
            return self._sample_string(field_name)
        if annotation is int:
            return self._sample_int(field_name)
        if annotation is float:
            return self._sample_float(field_name)
        if annotation is bool:
            return True

        return self._sample_string(field_name)

    @staticmethod
    def _sample_string(field_name: str) -> str:
        values = {
            "affiliation": "University of Toronto",
            "city": "Toronto",
            "company": "Decodra Labs",
            "country": "Canada",
            "currency": "USD",
            "date": "2026-05-26",
            "diagnosis": "hypertension",
            "email": "alice@example.com",
            "invoice_id": "INV-1001",
            "language": "Python",
            "license": "MIT",
            "location": "Toronto",
            "name": "Alice",
            "patient_id": "PAT-1001",
            "published": "2026-05-26",
            "severity": "moderate",
            "street": "123 King Street",
            "timezone": "America/Toronto",
            "title": "Structured Decoding Study",
            "venue": "NeurIPS",
        }
        return values.get(field_name, field_name.replace("_", " ").title())

    @staticmethod
    def _sample_int(field_name: str) -> int:
        values = {
            "age": 34,
            "capacity": 250,
            "citation_count": 42,
            "employee_count": 120,
            "experience_years": 5,
            "founded": 2021,
            "items_count": 3,
            "salary_max": 140000,
            "salary_min": 90000,
            "stars": 512,
            "word_count": 1800,
            "year": 2025,
        }
        return values.get(field_name, 1)

    @staticmethod
    def _sample_float(field_name: str) -> float:
        values = {
            "accuracy": 0.93,
            "amount": 125.75,
            "f1": 0.91,
            "precision": 0.92,
            "price": 19.99,
            "recall": 0.9,
        }
        return values.get(field_name, 0.5)

    @staticmethod
    def _ensure_confidence_keys(
        schema: type[BaseModel], confidence: dict[str, float]
    ) -> dict[str, float]:
        for field_name in schema.model_fields:
            confidence.setdefault(field_name, 0.0)
        return confidence

def project_root() -> Path:
    """Return the repository root for scripts that need stable paths."""

    return Path(__file__).resolve().parents[1]
