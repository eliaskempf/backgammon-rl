"""ExpectimaxAgent: negamax-over-chance correctness, terminal scoring, pruning.

The lookahead tests feed hand-crafted ``(Move, afterstate)`` lists straight to
``act`` so they don't depend on root move-generation order; the real
:func:`~bgrl.env.legal_moves` is still exercised one ply down (the opponent's replies),
which is exactly what the search must reason about.
"""

import numpy as np
import pytest

from bgrl.agents import ExpectimaxAgent, ValueAgent
from bgrl.agents.expectimax_agent import _terminal_equity
from bgrl.env import OFF, WEIGHTED_ROLLS, Env, EnvState, Move, Outcome, Player, SubMove, WinKind
from bgrl.env.encoding import N_FEATURES
from bgrl.nets.equity import CENTERED_CUBE

WHITE, BLACK = Player.WHITE, Player.BLACK


# --- stub nets -------------------------------------------------------------------------


class _FeatureNet:
    """Deterministic ValueNet: opponent p_win is a fixed smooth function of the features."""

    def __init__(self):
        self._w = np.random.default_rng(0).standard_normal(N_FEATURES).astype(np.float32)

    def evaluate(self, features):
        arr = np.asarray(features, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        out = np.zeros((flat.shape[0], 5), dtype=np.float32)
        out[:, 0] = 1.0 / (1.0 + np.exp(-(flat @ self._w)))
        return out.reshape(*arr.shape[:-1], 5)


class _ConstNet:
    """Every encoded position yields the same mover p_win, so equity is constant."""

    def __init__(self, p_win):
        self._p = float(p_win)

    def evaluate(self, features):
        arr = np.asarray(features, dtype=np.float32)
        out = np.zeros((*arr.shape[:-1], 5), dtype=np.float32)
        out[..., 0] = self._p
        return out


class _CountingNet:
    """Wraps a net and counts evaluate() calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def evaluate(self, features):
        self.calls += 1
        return self._inner.evaluate(features)


# --- helpers ---------------------------------------------------------------------------


def _opening(dice=(3, 1)):
    s = Env.initial_state()
    return s, Env.legal_moves(s, dice)


def _small():
    """A low-branching all-home position (fast enough for exact 2-ply)."""
    board = [0] * 24
    board[2], board[3] = 2, 1  # WHITE: 3 home checkers + 12 off
    board[20], board[21] = -1, -2  # BLACK: 3 home checkers + 12 off
    s = EnvState(board=tuple(board), bar=(0, 0), off=(12, 12), turn=WHITE)
    return s, Env.legal_moves(s, (2, 1))


# --- 1. plies=0 reproduces ValueAgent --------------------------------------------------


def test_plies0_matches_value_agent_opening():
    net = _FeatureNet()
    greedy, search = ValueAgent(net), ExpectimaxAgent(net, plies=0)
    s = Env.initial_state()
    for dice in [(3, 1), (6, 5), (2, 2), (5, 3), (4, 1)]:
        legal = Env.legal_moves(s, dice)
        assert search.act(s, dice, legal) == greedy.act(s, dice, legal)


def test_plies0_matches_value_agent_midgame():
    net = _FeatureNet()
    greedy, search = ValueAgent(net), ExpectimaxAgent(net, plies=0)
    s = _opening()[1][0][1]  # an afterstate one move into the game (BLACK to move)
    legal = Env.legal_moves(s, (5, 2))
    assert search.act(s, (5, 2), legal) == greedy.act(s, (5, 2), legal)


def test_plies0_issues_single_net_call():
    net = _CountingNet(_FeatureNet())
    s, legal = _opening()
    ExpectimaxAgent(net, plies=0).act(s, (3, 1), legal)
    assert net.calls == 1  # one batched evaluation over all root afterstates


# --- 2. dice roll enumeration ----------------------------------------------------------


def test_weighted_rolls_distribution():
    assert len(WEIGHTED_ROLLS) == 21
    assert sum(w for _, w in WEIGHTED_ROLLS) == pytest.approx(1.0)
    doubles = [(r, w) for r, w in WEIGHTED_ROLLS if r[0] == r[1]]
    nondoubles = [(r, w) for r, w in WEIGHTED_ROLLS if r[0] != r[1]]
    assert len(doubles) == 6 and all(w == pytest.approx(1 / 36) for _, w in doubles)
    assert len(nondoubles) == 15 and all(w == pytest.approx(2 / 36) for _, w in nondoubles)
    assert all(a < b for (a, b), _ in nondoubles)  # each non-double listed once, no reversal
    # total 36-outcome mass: doubles count once, non-doubles twice
    assert sum(1 if a == b else 2 for (a, b), _ in WEIGHTED_ROLLS) == 36


# --- 3. exact terminal scoring ---------------------------------------------------------


def test_terminal_equity_wins():
    assert _terminal_equity(Outcome(WHITE, WinKind.SINGLE), WHITE, CENTERED_CUBE) == 1.0
    assert _terminal_equity(Outcome(WHITE, WinKind.GAMMON), WHITE, CENTERED_CUBE) == 2.0
    assert _terminal_equity(Outcome(WHITE, WinKind.BACKGAMMON), WHITE, CENTERED_CUBE) == 3.0


def test_terminal_equity_losses():
    # perspective is the loser (BLACK); single loss is the all-zeros vector -> -1.
    assert _terminal_equity(Outcome(WHITE, WinKind.SINGLE), BLACK, CENTERED_CUBE) == -1.0
    assert _terminal_equity(Outcome(WHITE, WinKind.GAMMON), BLACK, CENTERED_CUBE) == -2.0
    assert _terminal_equity(Outcome(WHITE, WinKind.BACKGAMMON), BLACK, CENTERED_CUBE) == -3.0


def test_winning_terminal_chosen_despite_pessimistic_net():
    """A move that bears off the 15th checker is taken even when the net hates everything."""
    # WHITE just won (off 15); turn flips to the loser BLACK. Net rates all positions as a
    # near-certain loss for the mover, so only exact terminal scoring can pick this move.
    win_after = EnvState(board=(0,) * 24, bar=(0, 0), off=(15, 2), turn=BLACK)
    board = [0] * 24
    board[12] = 1  # a harmless non-terminal alternative (WHITE off 14, one checker left)
    plain_after = EnvState(board=tuple(board), bar=(0, 0), off=(14, 2), turn=BLACK)
    win_move = Move((SubMove(0, OFF),))
    plain_move = Move((SubMove(13, 12),))
    legal = [(plain_move, plain_after), (win_move, win_after)]  # winner listed second
    agent = ExpectimaxAgent(_ConstNet(0.99), plies=2)  # opp p_win 0.99 -> mover looks doomed
    assert agent.act(Env.initial_state(), (1, 1), legal) == win_move


# --- 4. lookahead changes the move (sign correctness) ----------------------------------


def test_one_ply_hits_to_deny_opponent_bearoff():
    """0-ply is indifferent; 1-ply hits a blot to stop the opponent bearing off next roll."""
    net = _ConstNet(0.5)  # neutral: every non-terminal position has equity 0
    open_white = [0] * 24
    open_white[12] = 1  # WHITE's lone checker; home board (0..5) left open so BLACK can enter
    # safe_after: BLACK has a blot on point 20 (its home); bears off and wins on any die >= 4.
    safe_board = open_white.copy()
    safe_board[20] = -1
    safe_after = EnvState(board=tuple(safe_board), bar=(0, 0), off=(14, 14), turn=BLACK)
    # hit_after: that blot was hit -> BLACK on the bar, far from home, cannot win next roll.
    hit_after = EnvState(board=tuple(open_white), bar=(0, 1), off=(14, 14), turn=BLACK)

    safe_move = Move((SubMove(23, 19),))
    hit_move = Move((SubMove(23, 20), SubMove(20, 19)))
    legal = [(safe_move, safe_after), (hit_move, hit_after)]  # non-hit first
    after_of = dict(legal)

    m0 = ValueAgent(net).act(Env.initial_state(), (3, 1), legal)
    m1 = ExpectimaxAgent(net, plies=1).act(Env.initial_state(), (3, 1), legal)
    assert after_of[m0].bar[BLACK] == 0  # 0-ply (indifferent) leaves the blot alone
    assert after_of[m1].bar[BLACK] == 1  # 1-ply sees the bear-off and hits
    assert m0 != m1


# --- 5. forced pass inside the search consumes a ply -----------------------------------


def _dancing_afterstate():
    """BLACK on the bar against a fully closed WHITE home board -> dances on every roll."""
    board = [0] * 24
    for p in range(6):
        board[p] = 2  # WHITE closes all entry points (12 checkers)
    board[18] = -12  # BLACK's other 12 checkers, out of WHITE's home
    return EnvState(board=tuple(board), bar=(0, 1), off=(3, 2), turn=BLACK)


def test_pass_consumes_a_ply():
    # Constant net -> equity +0.5 from any mover's POV. With BLACK dancing on all 21 rolls,
    # eval_pov(s, 1) = -eval_pov(pass-to-WHITE, 0) = -0.5, one sign flip below the 0-ply +0.5.
    agent = ExpectimaxAgent(_ConstNet(0.75), plies=1)
    agent._cache = {}
    s = _dancing_afterstate()
    assert agent._eval_pov(s, 0) == pytest.approx(0.5)
    assert agent._eval_pov(s, 1) == pytest.approx(-0.5)


def test_pass_heavy_position_runs_at_depth():
    agent = ExpectimaxAgent(_ConstNet(0.5), plies=2)
    s = _dancing_afterstate()
    legal = [(Move((SubMove(18, 19),)), s)]
    assert agent.act(Env.initial_state(), (1, 1), legal) == legal[0][0]  # no crash, legal move


# --- 6. tie-breaking -------------------------------------------------------------------


def test_deterministic_tie_break_is_first():
    s, legal = _opening()
    agent = ExpectimaxAgent(_ConstNet(0.5), plies=1)  # all moves tie
    assert agent.act(s, (3, 1), legal) == legal[0][0]


def test_rng_tie_break_varies_but_stays_legal():
    s, legal = _opening()
    moves = {m for m, _ in legal}
    agent = ExpectimaxAgent(_ConstNet(0.5), plies=1, rng=np.random.default_rng(0))
    picks = {agent.act(s, (3, 1), legal) for _ in range(30)}
    assert len(picks) > 1 and picks <= moves


# --- 7. candidate pruning --------------------------------------------------------------


def test_large_top_k_matches_exact_search():
    net = _FeatureNet()
    s, legal = _small()
    exact = ExpectimaxAgent(net, plies=2).act(s, (2, 1), legal)
    pruned = ExpectimaxAgent(net, plies=2, top_k=50).act(s, (2, 1), legal)  # 50 >> branching
    assert pruned == exact


def test_pruned_choice_is_legal():
    net = _FeatureNet()
    s, legal = _small()
    moves = {m for m, _ in legal}
    assert ExpectimaxAgent(net, plies=2, top_k=1).act(s, (2, 1), legal) in moves


def test_constructor_validation():
    with pytest.raises(ValueError):
        ExpectimaxAgent(_FeatureNet(), plies=-1)
    with pytest.raises(ValueError):
        ExpectimaxAgent(_FeatureNet(), top_k=0)


# --- 8. win_prob (web display companion) -----------------------------------------------


def test_win_prob_matches_value_agent_at_any_depth():
    net = _FeatureNet()
    s = Env.initial_state()
    _, after = Env.legal_moves(s, (3, 1))[0]
    expected = ValueAgent(net).win_prob(after)
    for plies in (0, 1, 2):  # search depth doesn't change the net's 0-ply afterstate estimate
        assert ExpectimaxAgent(net, plies=plies).win_prob(after) == expected
