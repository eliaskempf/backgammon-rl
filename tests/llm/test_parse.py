"""Tests for the pure choice parser and the output-format strategies."""

from __future__ import annotations

import pytest

from bgrl.llm.parse import OutputFormat, parse_choice

INDEX = OutputFormat.INDEX_TEXT
STRUCT = OutputFormat.STRUCTURED


@pytest.mark.parametrize(
    ("text", "fmt", "expected"),
    [
        ("3", INDEX, 3),
        ("0", INDEX, 0),
        ("Answer: 3", INDEX, 3),
        ("I choose move 2 because it makes a point.", INDEX, 2),
        ('{"choice": 2}', STRUCT, 2),
        ('```json\n{"choice": 4}\n```', STRUCT, 4),
        ('The best is {"choice": 1}.', STRUCT, 1),
        ('{"choice": "3"}', STRUCT, 3),
        ("4", STRUCT, 4),  # model ignored the schema -> bare-int fallback
    ],
)
def test_parse_choice_valid(text, fmt, expected):
    assert parse_choice(text, n=5, fmt=fmt) == expected


@pytest.mark.parametrize(
    ("text", "fmt"),
    [
        ("99", INDEX),  # out of range (n=5)
        ("-1", INDEX),  # negative
        ("2.0", INDEX),  # float, not an integer index
        ("", INDEX),  # empty
        ("pick the safe one", INDEX),  # prose, no integer
        ("not json at all", STRUCT),
        ('{"choice": 2.0}', STRUCT),  # float choice rejected
        ('{"choice": true}', STRUCT),  # bool is not a valid index
        ('{"move": 1}', STRUCT),  # wrong key, no bare int -> None
    ],
)
def test_parse_choice_invalid_returns_none(text, fmt):
    assert parse_choice(text, n=5, fmt=fmt) is None


def test_parse_choice_zero_legal_moves_is_none():
    assert parse_choice("0", n=0, fmt=INDEX) is None


def test_structured_response_format_schema():
    rf = STRUCT.response_format(7)
    schema = rf["json_schema"]["schema"]
    assert schema["properties"]["choice"] == {"type": "integer", "minimum": 0, "maximum": 6}
    assert schema["required"] == ["choice"]
    assert schema["additionalProperties"] is False


def test_index_text_has_no_response_format():
    assert INDEX.response_format(5) is None


def test_instructions_mention_bounds():
    assert "0 to 4" in INDEX.instruction(5)
    assert "0 to 4" in STRUCT.instruction(5)
    assert "choice" in STRUCT.instruction(5)
