"""Tests for the per-head calibration diagnostic (:mod:`bgrl.training.calibration`)."""

from __future__ import annotations

import numpy as np
import pytest

from bgrl.env import Outcome, Player, WinKind
from bgrl.nets.base import OUTCOME_DIM
from bgrl.training.calibration import (
    HEAD_NAMES,
    _ece,
    _realized_outcome,
    calibration_report,
    reliability_table,
)


def test_realized_outcome_from_winner_and_loser_pov() -> None:
    # Winner sees the cumulative win heads; the loser sees the flipped loss heads.
    g = Outcome(winner=Player.WHITE, kind=WinKind.GAMMON)
    assert np.array_equal(_realized_outcome(g, Player.WHITE), [1, 1, 0, 0, 0])
    assert np.array_equal(_realized_outcome(g, Player.BLACK), [0, 0, 0, 1, 0])

    bg = Outcome(winner=Player.BLACK, kind=WinKind.BACKGAMMON)
    assert np.array_equal(_realized_outcome(bg, Player.BLACK), [1, 1, 1, 0, 0])
    assert np.array_equal(_realized_outcome(bg, Player.WHITE), [0, 0, 0, 1, 1])

    single = Outcome(winner=Player.WHITE, kind=WinKind.SINGLE)
    assert np.array_equal(_realized_outcome(single, Player.WHITE), [1, 0, 0, 0, 0])
    assert np.array_equal(_realized_outcome(single, Player.BLACK), [0, 0, 0, 0, 0])


def test_ece_zero_when_perfectly_calibrated() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(500)
    assert _ece(p, p.copy(), bins=10) == 0.0  # prediction equals realisation in every bin


def test_ece_detects_a_collapsed_head() -> None:
    # Predict 0 everywhere while half the outcomes realise: ECE is the full gap of 0.5.
    predicted = np.zeros(400)
    realized = np.concatenate([np.ones(200), np.zeros(200)])
    assert _ece(predicted, realized, bins=10) == 0.5


def test_reliability_table_partitions_samples_and_reduces_to_ece() -> None:
    rng = np.random.default_rng(2)
    predicted = rng.random(1000)
    realized = (rng.random(1000) < predicted).astype(float)  # roughly calibrated
    table = reliability_table(predicted, realized, bins=10)

    # Every sample lands in exactly one populated bin, and bins are ordered & within [0, 1].
    assert sum(b.count for b in table) == 1000
    assert all(0.0 <= b.lo < b.hi <= 1.0 for b in table)
    assert [b.lo for b in table] == sorted(b.lo for b in table)

    # The table is the per-bin data behind _ece: re-reducing it must reproduce _ece exactly.
    derived = sum((b.count / 1000) * abs(b.predicted_mean - b.realized_mean) for b in table)
    assert derived == pytest.approx(_ece(predicted, realized, bins=10))


def test_reliability_table_empty_input() -> None:
    empty = np.zeros(0)
    assert reliability_table(empty, empty, bins=10) == []


class _ConstNet:
    """A ValueNet stub that returns a fixed outcome vector for every position."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = np.asarray(vec, dtype=np.float64)

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        batch = np.asarray(features).shape[:-1]
        return np.broadcast_to(self._vec, (*batch, OUTCOME_DIM)).copy()


def test_calibration_report_structure_and_collapse_signal() -> None:
    # A net with the four magnitude heads pinned to 0 (the collapse): their predicted
    # mean must be exactly 0, while p_win reports the constant it always emits. This
    # exercises the full self-play -> predict -> aggregate path deterministically.
    net = _ConstNet([0.6, 0.0, 0.0, 0.0, 0.0])
    report = calibration_report(net, games=8, rng=np.random.default_rng(1))

    assert report.games == 8
    assert report.samples > 0
    assert tuple(h.name for h in report.heads) == HEAD_NAMES

    by_name = {h.name: h for h in report.heads}
    assert by_name["p_win"].predicted_mean == pytest.approx(0.6)
    for collapsed in ("p_win_g", "p_win_bg", "p_lose_g", "p_lose_bg"):
        assert by_name[collapsed].predicted_mean == 0.0  # exact: mean of zeros
        assert by_name[collapsed].realized_mean >= 0.0

    metrics = report.to_metrics()
    assert metrics["cal/p_win_pred"] == pytest.approx(0.6)
    assert set(metrics) == {
        f"cal/{name}_{stat}" for name in HEAD_NAMES for stat in ("pred", "real", "ece")
    }
