from src.confidence import ConfidenceAggregator


def test_aggregation_with_mock_token_records():
    records = [
        {
            "token_id": 1,
            "pre_mask_prob": 0.8,
            "post_mask_prob": 0.9,
            "constraint_pressure": 0.1,
            "valid_token_entropy": 0.2,
            "field_name": "name",
        },
        {
            "token_id": 2,
            "pre_mask_prob": 0.6,
            "post_mask_prob": 0.8,
            "constraint_pressure": 0.2,
            "valid_token_entropy": 0.3,
            "field_name": "name",
        },
    ]

    scores = ConfidenceAggregator(records).aggregate()

    assert scores["name"] == 0.595


def test_zero_constraint_pressure_returns_mean_pre_mask_prob():
    records = [
        {
            "token_id": 1,
            "pre_mask_prob": 0.4,
            "post_mask_prob": 0.4,
            "constraint_pressure": 0.0,
            "valid_token_entropy": 0.0,
            "field_name": "age",
        },
        {
            "token_id": 2,
            "pre_mask_prob": 0.8,
            "post_mask_prob": 0.8,
            "constraint_pressure": 0.0,
            "valid_token_entropy": 0.0,
            "field_name": "age",
        },
    ]

    scores = ConfidenceAggregator(records).aggregate()

    assert abs(scores["age"] - 0.6) < 1e-9


def test_get_flagged_fields_returns_fields_below_threshold():
    aggregator = ConfidenceAggregator()
    flagged = aggregator.get_flagged_fields(
        threshold=0.7,
        scores={"name": 0.8, "age": 0.5},
    )

    assert flagged == {"age": 0.5}


def test_single_token_field():
    records = [
        {
            "token_id": 1,
            "pre_mask_prob": 0.75,
            "post_mask_prob": 1.0,
            "constraint_pressure": 0.25,
            "valid_token_entropy": 0.0,
            "field_name": "city",
        }
    ]

    scores = ConfidenceAggregator(records).aggregate()

    assert scores["city"] == 0.5625
