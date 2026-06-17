"""Output-format strategies and the pure choice parser.

The LLM never scores afterstates; it picks one move from an enumerated list by its
**integer index** (afterstate-first — see CLAUDE.md §4). :class:`OutputFormat` is the
swept knob for *how* that index is requested:

* ``INDEX_TEXT`` — free text, "reply with only the index". Works on every model.
* ``STRUCTURED`` — OpenRouter ``response_format`` JSON schema constraining the answer
  to ``{"choice": <int>}``. More reliable where supported; models that reject it
  surface :class:`~bgrl.llm.client.StructuredOutputUnsupported`, and the agent falls
  back to ``INDEX_TEXT``.

:func:`parse_choice` is pure (no I/O) and returns ``None`` for *every* failure mode —
malformed text, wrong type, or an index outside ``[0, n)`` — so the agent has a single
validity predicate driving its re-prompt/fallback loop.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from enum import Enum

# An integer token that is not part of a float: not preceded by a digit/dot and not
# followed by a decimal fraction. So "Answer: 3" -> 3, but "2.0" matches nothing.
_INT_TOKEN = re.compile(r"(?<![\d.])-?\d+(?!\.\d)")


class OutputFormat(Enum):
    """How the chosen-move index is requested from (and parsed out of) the model."""

    INDEX_TEXT = "index_text"
    STRUCTURED = "structured"

    def response_format(self, n: int) -> dict | None:
        """The OpenRouter ``response_format`` block for ``n`` legal moves, or ``None``."""
        if self is OutputFormat.STRUCTURED:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "move_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "choice": {"type": "integer", "minimum": 0, "maximum": max(n - 1, 0)}
                        },
                        "required": ["choice"],
                        "additionalProperties": False,
                    },
                },
            }
        return None

    def instruction(self, n: int) -> str:
        """The prompt sentence telling the model how to format its answer."""
        if self is OutputFormat.STRUCTURED:
            return (
                f'Respond with a JSON object of the form {{"choice": <index>}} where <index> '
                f"is an integer from 0 to {n - 1} identifying your chosen move."
            )
        return (
            f"Respond with only the integer index (0 to {n - 1}) of your chosen move, "
            "and nothing else."
        )


def parse_choice(text: str, *, n: int, fmt: OutputFormat) -> int | None:
    """Extract a valid move index in ``[0, n)`` from ``text``, else ``None``.

    For ``STRUCTURED`` the JSON ``choice`` field is tried first (tolerating markdown
    fences / surrounding prose), then a bare-integer fallback for models that ignore
    the schema. For ``INDEX_TEXT`` only the bare-integer path is used. Floats, bools,
    out-of-range values, and text with no integer all yield ``None``.
    """
    if n <= 0:
        return None
    idx = _extract_index(text, fmt)
    if idx is None or not (0 <= idx < n):
        return None
    return idx


def _extract_index(text: str, fmt: OutputFormat) -> int | None:
    if fmt is OutputFormat.STRUCTURED:
        found_object, choice = _parse_structured(text)
        # A JSON object was returned: trust its (possibly invalid) ``choice`` and do
        # not dig a stray integer out of a wrong-shaped object. Only when the model
        # emitted no JSON object at all do we tolerate a bare integer.
        if found_object:
            return choice
    return _first_int_token(text)


def _first_int_token(text: str) -> int | None:
    match = _INT_TOKEN.search(text)
    return int(match.group()) if match is not None else None


def _parse_structured(text: str) -> tuple[bool, int | None]:
    """Return ``(found_json_object, choice_index)``.

    ``found_json_object`` is True iff some candidate parsed to a JSON *object*; the
    second element is the validated ``choice`` (``None`` if missing/ill-typed).
    """
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return True, _coerce_choice(obj["choice"]) if "choice" in obj else None
    return False, None


def _json_candidates(text: str) -> Iterator[str]:
    stripped = text.strip()
    yield stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:  # tolerate fences / prose wrapping the JSON object
        yield stripped[start : end + 1]


def _coerce_choice(value: object) -> int | None:
    if isinstance(value, bool):  # bool is an int subclass; not a valid index
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _first_int_token(value)
    return None
