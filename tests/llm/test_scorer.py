"""Tests for the position scorers."""

from __future__ import annotations

import numpy as np
import pytest

from bgrl.agents.value_agent import ValueAgent
from bgrl.llm.scorer import (
    GnubgPositionScorer,
    ReferenceAgentScorer,
    ScoreMode,
)


class _StubNet:
    """A deterministic ValueNet whose p_win varies across afterstates."""

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        s = features.sum(axis=-1)
        p = 1.0 / (1.0 + np.exp(-(s - s.mean())))
        out = np.zeros((*features.shape[:-1], 5))
        out[..., 0] = p
        return out


def test_position_set_is_deterministic():
    scorer = ReferenceAgentScorer.pubeval()
    first = scorer.positions(n=4, seed=1)
    second = scorer.positions(n=4, seed=1)
    assert len(first) == 4
    assert first == second  # frozen, seed-determined => identical (the CRN analogue)


def test_positions_are_genuine_choices():
    positions = ReferenceAgentScorer.pubeval().positions(n=5, seed=2)
    assert all(len(pos.legal) > 1 for pos in positions)


def test_agreement_scoring():
    scorer = ReferenceAgentScorer.pubeval()
    pos = scorer.positions(n=1, seed=3)[0]
    assert scorer.score(pos, pos.reference_move) == 1.0
    other = next(move for move, _ in pos.legal if move != pos.reference_move)
    assert scorer.score(pos, other) == 0.0


def test_equity_loss_scoring_zero_at_reference_and_nonpositive_elsewhere():
    net = _StubNet()
    scorer = ReferenceAgentScorer(ValueAgent(net), value_net=net, mode=ScoreMode.EQUITY_LOSS)
    pos = scorer.positions(n=1, seed=4)[0]
    assert pos.equities is not None and len(pos.equities) == len(pos.legal)
    assert scorer.score(pos, pos.reference_move) == pytest.approx(0.0)
    assert all(scorer.score(pos, move) <= 1e-9 for move, _ in pos.legal)


def test_equity_loss_requires_value_net():
    with pytest.raises(ValueError, match="value_net"):
        ReferenceAgentScorer(ValueAgent(_StubNet()), mode=ScoreMode.EQUITY_LOSS)


def test_gnubg_scorer_is_a_documented_stub():
    scorer = GnubgPositionScorer()
    assert scorer.name == "gnubg"
    with pytest.raises(NotImplementedError, match="WP3"):
        scorer.positions(n=1, seed=0)
    with pytest.raises(NotImplementedError, match="WP3"):
        scorer.score(None, None)  # type: ignore[arg-type]
