"""Incremental JSON field tracking for generated token streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Container:
    kind: str
    field_path: str | None = None
    expecting: str = "value"


class JSONFieldTracker:
    """Track the active JSON field while a JSON document is decoded incrementally."""

    def __init__(self) -> None:
        """Initialize an empty tracker."""

        self.reset()

    def reset(self) -> None:
        """Reset all parser state so the tracker can process a new JSON document."""

        self.stack: list[_Container] = []
        self.path: list[str] = []
        self.pending_key: str | None = None
        self._active_field: str | None = None
        self._last_token_field: str | None = None
        self._in_string = False
        self._escape = False
        self._string_role: str | None = None
        self._string_buffer: list[str] = []
        self._string_field: str | None = None
        self._in_scalar = False
        self._scalar_field: str | None = None

    def advance(self, decoded_token: str) -> str | None:
        """Advance the parser with decoded text and return the field touched by it."""

        touched_field: str | None = None
        for char in decoded_token:
            before = self._active_field
            self._advance_char(char)
            after = self._active_field

            if before:
                touched_field = before
            elif after and self._token_char_counts_as_value(char):
                touched_field = after

        self._last_token_field = touched_field
        return touched_field

    def current_field(self) -> str | None:
        """Return the currently generated field path, or ``None`` for keys/structure."""

        return self._active_field

    def last_token_field(self) -> str | None:
        """Return the field associated with the most recent token passed to ``advance``."""

        return self._last_token_field

    def _advance_char(self, char: str) -> None:
        if self._in_string:
            self._consume_string_char(char)
            return

        if char.isspace():
            return

        if self._in_scalar and char not in ",}]":
            self._active_field = self._scalar_field
            return

        if self._in_scalar and char in ",}]":
            self._in_scalar = False
            self._scalar_field = None
            self._active_field = None

        if char == "{":
            self._open_object()
        elif char == "}":
            self._close_container("object")
        elif char == "[":
            self._open_array()
        elif char == "]":
            self._close_container("array")
        elif char == '"':
            self._open_string()
        elif char == ":":
            self._transition_to_value()
        elif char == ",":
            self._transition_after_comma()
        else:
            self._start_scalar_value()

    def _consume_string_char(self, char: str) -> None:
        if self._escape:
            self._string_buffer.append(char)
            self._escape = False
            return

        if char == "\\":
            self._escape = True
            return

        if char == '"':
            value = "".join(self._string_buffer)
            if self._string_role == "key":
                self.pending_key = value
                if self.stack and self.stack[-1].kind == "object":
                    self.stack[-1].expecting = "colon"
            elif self._string_role == "value":
                self._active_field = None
                self._string_field = None

            self._in_string = False
            self._string_role = None
            self._string_buffer = []
            return

        self._string_buffer.append(char)
        if self._string_role == "value":
            self._active_field = self._string_field

    def _open_object(self) -> None:
        parent = self.stack[-1] if self.stack else None
        if parent and parent.expecting in {"value", "value_or_end"}:
            field_path = self._current_value_field()
            leaf = self.pending_key
            parent.expecting = "comma_or_end"
            self.pending_key = None
            self._active_field = None
            if leaf:
                self.path.append(leaf)
            self.stack.append(_Container("object", field_path, "key_or_end"))
            return

        self.stack.append(_Container("object", None, "key_or_end"))
        self._active_field = None

    def _close_container(self, kind: str) -> None:
        if self._in_scalar:
            self._in_scalar = False
            self._scalar_field = None

        if self.stack:
            closed = self.stack.pop()
            if closed.kind == "object" and closed.field_path and self.path:
                self.path.pop()

        self.pending_key = None
        self._active_field = None

        if self.stack:
            self.stack[-1].expecting = "comma_or_end"

    def _open_array(self) -> None:
        field_path = self._current_value_field()
        if self.stack:
            self.stack[-1].expecting = "comma_or_end"
        self.pending_key = None
        self._active_field = None
        self.stack.append(_Container("array", field_path, "value_or_end"))

    def _open_string(self) -> None:
        if not self.stack:
            self._in_string = True
            self._string_role = "value"
            self._string_field = None
            self._string_buffer = []
            return

        container = self.stack[-1]
        if container.kind == "object" and container.expecting in {"key_or_end", "key"}:
            self._in_string = True
            self._string_role = "key"
            self._string_field = None
            self._string_buffer = []
            self._active_field = None
            return

        field_path = self._current_value_field()
        container.expecting = "comma_or_end"
        self.pending_key = None
        self._in_string = True
        self._string_role = "value"
        self._string_field = field_path
        self._string_buffer = []
        self._active_field = field_path

    def _transition_to_value(self) -> None:
        if self.stack and self.stack[-1].kind == "object":
            self.stack[-1].expecting = "value"
            self._active_field = self._current_value_field()

    def _transition_after_comma(self) -> None:
        self.pending_key = None
        self._active_field = None
        if not self.stack:
            return

        container = self.stack[-1]
        if container.kind == "object":
            container.expecting = "key_or_end"
        elif container.kind == "array":
            container.expecting = "value_or_end"

    def _start_scalar_value(self) -> None:
        if not self.stack:
            return

        container = self.stack[-1]
        if container.expecting not in {"value", "value_or_end"}:
            return

        field_path = self._current_value_field()
        container.expecting = "comma_or_end"
        self.pending_key = None
        self._in_scalar = True
        self._scalar_field = field_path
        self._active_field = field_path

    def _current_value_field(self) -> str | None:
        if not self.stack:
            return None

        container = self.stack[-1]
        if container.kind == "array":
            return container.field_path
        if self.pending_key is None:
            return None
        return ".".join([*self.path, self.pending_key])

    def _token_char_counts_as_value(self, char: str) -> bool:
        return not char.isspace() and char not in "{}[]:,"
