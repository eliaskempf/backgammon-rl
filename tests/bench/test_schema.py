"""Benchmark result JSON round-trip."""

from bgrl.bench.schema import SCHEMA_VERSION, build_result, read_json, write_json


def test_schema_roundtrip(tmp_path):
    result = build_result(
        tag="unit",
        timestamp="2026-01-01T00:00:00+00:00",
        fingerprint={"hostname": "x"},
        config={"games": 1},
        env={"games_per_sec": 1.0},
        net=None,
    )
    assert result["schema_version"] == SCHEMA_VERSION
    path = write_json(result, tmp_path / "r.json")
    assert read_json(path) == result
