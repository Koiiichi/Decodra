from src.engine import DecodraEngine
from src.schemas import PersonSchema


def test_decodra_engine_loads_gpt2():
    engine = DecodraEngine(model_name="gpt2")

    assert engine.model is not None
    assert engine.tokenizer is not None


def test_generate_returns_expected_shape_and_valid_output():
    engine = DecodraEngine(model_name="gpt2")

    result = engine.generate(
        prompt="Extract the following information:",
        schema=PersonSchema,
        return_confidence=True,
        measure_overhead=False,
    )

    assert set(result) == {"output", "_confidence", "_meta"}
    PersonSchema.model_validate(result["output"])
    assert set(PersonSchema.model_fields).issubset(result["_confidence"])
    assert all(0.0 <= value <= 1.0 for value in result["_confidence"].values())
