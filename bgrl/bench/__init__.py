"""Throughput-benchmark library (machine fingerprint, self-play, runner, schema).

This package ``__init__`` stays torch-free on purpose: spawn self-play workers
import :mod:`bgrl.bench.selfplay` (and therefore this ``__init__``), and we don't
want torch loaded in every worker. The torch-using orchestration lives in
:mod:`bgrl.bench.runner`, imported explicitly by callers that need it.
"""

from .fingerprint import machine_fingerprint
from .schema import SCHEMA_VERSION, build_result, read_json, write_json
from .selfplay import play_random_game, run_selfplay

__all__ = [
    "SCHEMA_VERSION",
    "build_result",
    "machine_fingerprint",
    "play_random_game",
    "read_json",
    "run_selfplay",
    "write_json",
]
