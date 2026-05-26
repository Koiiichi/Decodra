from src.field_tracker import JSONFieldTracker


def test_simple_field_identifies_active_value_field():
    tracker = JSONFieldTracker()
    touched = []

    for char in '{"name": "Alice"}':
        touched.append(tracker.advance(char))

    assert "name" in touched
    assert tracker.current_field() is None


def test_nested_object_tracking_uses_dot_notation():
    tracker = JSONFieldTracker()
    touched = []

    for char in '{"address": {"city": "Toronto"}}':
        touched.append(tracker.advance(char))

    assert "address.city" in touched
    assert tracker.current_field() is None


def test_tracker_resets_between_fields():
    tracker = JSONFieldTracker()
    per_char = []

    for char in '{"name":"Alice","age":34}':
        tracker.advance(char)
        per_char.append(tracker.current_field())

    assert "name" in per_char
    assert "age" in per_char
    assert tracker.current_field() is None
