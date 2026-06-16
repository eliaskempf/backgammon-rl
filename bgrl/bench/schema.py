"""Benchmark result schema + JSON I/O.

A result is a plain JSON-serialisable dict (kept simple so cross-host files merge
trivially). ``schema_version`` guards against silently comparing incompatible runs.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


def build_result(
    *,
    tag: str,
    timestamp: str,
    fingerprint: dict,
    config: dict,
    env: dict,
    net: dict | None,
) -> dict:
    """Assemble the top-level result record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "timestamp_utc": timestamp,
        "fingerprint": fingerprint,
        "config": config,
        "env": env,
        "net": net,
    }


def write_json(result: dict, path: str | Path) -> Path:
    """Write ``result`` as pretty JSON, creating parent dirs. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True))
    return p


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
