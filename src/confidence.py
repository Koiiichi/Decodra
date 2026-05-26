"""Confidence aggregation from constrained decoding token signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TokenRecord:
    """Token-level signal captured during constrained generation."""

    token_id: int
    pre_mask_prob: float
    post_mask_prob: float
    constraint_pressure: float
    valid_token_entropy: float
    field_name: str | None


class ConfidenceAggregator:
    """Aggregate token-level probability signals into field confidence scores."""

    def __init__(self, records: Iterable[TokenRecord | Mapping[str, Any]] | None = None):
        """Create an aggregator with an optional initial token record sequence."""

        self.records = list(records or [])
        self._scores: dict[str, float] = {}

    def aggregate(
        self, records: Iterable[TokenRecord | Mapping[str, Any]] | None = None
    ) -> dict[str, float]:
        """Return per-field confidence scores clipped to the inclusive [0, 1] range."""

        if records is not None:
            self.records = list(records)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.records:
            normalized = self._normalize_record(record)
            field_name = normalized.get("field_name")
            if not field_name:
                continue
            grouped.setdefault(str(field_name), []).append(normalized)

        scores: dict[str, float] = {}
        for field_name, field_records in grouped.items():
            pre_mask_probs = np.array(
                [float(record["pre_mask_prob"]) for record in field_records],
                dtype=np.float64,
            )
            pressures = np.array(
                [abs(float(record["constraint_pressure"])) for record in field_records],
                dtype=np.float64,
            )

            mean_pre_mask = float(pre_mask_probs.mean()) if pre_mask_probs.size else 0.0
            mean_pressure = float(pressures.mean()) if pressures.size else 0.0
            confidence = mean_pre_mask * (1.0 - mean_pressure)
            scores[field_name] = float(np.clip(confidence, 0.0, 1.0))

        self._scores = scores
        return scores

    def get_flagged_fields(
        self,
        threshold: float = 0.7,
        scores: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return fields whose confidence is below ``threshold``."""

        active_scores = dict(scores if scores is not None else self._scores)
        if not active_scores:
            active_scores = self.aggregate()
        return {
            field_name: confidence
            for field_name, confidence in active_scores.items()
            if confidence < threshold
        }

    @staticmethod
    def _normalize_record(record: TokenRecord | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(record, TokenRecord):
            return asdict(record)
        return dict(record)
