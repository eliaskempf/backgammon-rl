"""ValueAgent selection: the perspective sign convention (CLAUDE.md s6) + tie-breaks.

The afterstate's POV is the *opponent's*, so a higher opponent p_win must make the
agent AVOID that move. These tests pin that sign with a stub net of known outputs.
"""

import numpy as np

from bgrl.agents import ValueAgent
from bgrl.env import Env


class _StubNet:
    """Returns a preset opponent p_win per batch row (other heads zero)."""

    def __init__(self, opp_pwin):
        self._opp_pwin = np.asarray(opp_pwin, dtype=np.float32)

    def evaluate(self, features):
        out = np.zeros((features.shape[0], 5), dtype=np.float32)
        out[:, 0] = self._opp_pwin
        return out


def _opening():
    s = Env.initial_state()
    return s, Env.legal_moves(s, (3, 1))


def test_avoids_move_best_for_opponent():
    s, legal = _opening()
    opp = np.full(len(legal), 0.5)
    opp[2] = 0.95  # opponent wins big from afterstate 2 -> the mover must avoid it
    assert ValueAgent(_StubNet(opp)).act(s, (3, 1), legal) != legal[2][0]


def test_picks_move_worst_for_opponent():
    s, legal = _opening()
    opp = np.full(len(legal), 0.5)
    opp[3] = 0.05  # opponent almost surely loses from afterstate 3 -> the mover picks it
    assert ValueAgent(_StubNet(opp)).act(s, (3, 1), legal) == legal[3][0]


def test_deterministic_tie_break_is_first():
    s, legal = _opening()
    agent = ValueAgent(_StubNet(np.full(len(legal), 0.5)))  # all equal
    assert agent.act(s, (3, 1), legal) == legal[0][0]


def test_rng_tie_break_varies_but_stays_legal():
    s, legal = _opening()
    moves = {m for m, _ in legal}
    agent = ValueAgent(_StubNet(np.full(len(legal), 0.5)), rng=np.random.default_rng(0))
    picks = {agent.act(s, (3, 1), legal) for _ in range(30)}
    assert len(picks) > 1  # randomised among the tied-best moves
    assert picks <= moves
