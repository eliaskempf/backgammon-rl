"""Pubeval — Tesauro's public-domain benchmark opponent (in-codebase strength oracle).

``pubeval`` is a fixed linear move-selection evaluator that Gerry Tesauro (IBM
Research) released as a public service so backgammon learning programs could be
compared on a common yardstick. "Win-rate vs pubeval" is the de-facto standard metric
in the TD-Gammon / RL-backgammon literature: ~50% is roughly intermediate-human play,
and strong nets reach 60%+. It is far weaker than gnubg (our WP3 oracle), but unlike
gnubg it needs no external process — exactly the absolute reference we want *during*
training, where win-rate-vs-random saturates almost immediately.

The score is a dot product ``W · x`` of a raw board encoding ``x`` with one of two
122-weight vectors — ``_WR`` for races (no contact), ``_WC`` for contact positions.
The weights and the :func:`_setx` encoding are reproduced **verbatim** from Tesauro's
``pubeval.c`` (public domain); only the mapping from our :class:`~bgrl.env.EnvState`
into pubeval's ``pos[]`` layout is ours. Source:
https://github.com/weekend37/Backgammon/blob/master/pubeval.c

pubeval's ``pos[0..27]`` board is from the **mover's** ("computer's") point of view:
elements 1..24 are points with the mover moving 24→1 (mover's men positive, opponent's
negative); ``pos[25]`` = mover men on the bar, ``pos[0]`` = opponent men on the bar
(negative); ``pos[26]`` = mover men off, ``pos[27]`` = opponent men off (negative). The
``race`` flag is taken from the position *before* the move.
"""

from __future__ import annotations

import numpy as np

from bgrl.env import Dice, EnvState, Move, Player

# fmt: off
_WR: tuple[float, ...] = (
    0.0, -0.1716, 0.2701, 0.29906, -0.08471, 0.0, -1.40375, -1.05121, 0.07217, -0.01351,
    0.0, -1.29506, -2.16183, 0.13246, -1.03508, 0.0, -2.29847, -2.34631, 0.17253, 0.08302,
    0.0, -1.27266, -2.87401, -0.07456, -0.3424, 0.0, -1.3464, -2.46556, -0.13022, -0.01591,
    0.0, 0.27448, 0.60015, 0.48302, 0.25236, 0.0, 0.39521, 0.68178, 0.05281, 0.09266, 0.0,
    0.24855, -0.06844, -0.37646, 0.05685, 0.0, 0.17405, 0.0043, 0.74427, 0.00576, 0.0,
    0.12392, 0.31202, -0.91035, -0.1627, 0.0, 0.01418, -0.10839, -0.02781, -0.88035, 0.0,
    1.07274, 2.00366, 1.16242, 0.2252, 0.0, 0.85631, 1.06349, 1.49549, 0.18966, 0.0,
    0.37183, -0.50352, -0.14818, 0.12039, 0.0, 0.13681, 0.13978, 1.11245, -0.12707, 0.0,
    -0.22082, 0.20178, -0.06285, -0.52728, 0.0, -0.13597, -0.19412, -0.09308, -1.26062,
    0.0, 3.05454, 5.16874, 1.5068, 5.35, 0.0, 2.19605, 3.8539, 0.88296, 2.30052, 0.0,
    0.92321, 1.08744, -0.11696, -0.7856, 0.0, -0.09795, -0.8305, -1.09167, -4.94251, 0.0,
    -1.00316, -3.66465, -2.56906, -9.67677, 0.0, -2.77982, -7.26713, -3.40177, -12.32252,
    0.0, 3.4204,
)

_WC: tuple[float, ...] = (
    0.25696, -0.66937, -1.66135, -2.02487, -2.53398, -0.16092, -1.11725, -1.06654, -0.9283,
    -1.99558, -1.10388, -0.80802, 0.09856, -0.62086, -1.27999, -0.5922, -0.73667, 0.89032,
    -0.38933, -1.59847, -1.50197, -0.60966, 1.56166, -0.47389, -1.8039, -0.83425, -0.97741,
    -1.41371, 0.245, 0.1097, -1.36476, -1.05572, 1.1542, 0.11069, -0.38319, -0.74816,
    -0.59244, 0.81116, -0.39511, 0.11424, -0.73169, -0.56074, 1.09792, 0.15977, 0.13786,
    -1.18435, -0.43363, 1.06169, -0.21329, 0.04798, -0.94373, -0.22982, 1.22737, -0.13099,
    -0.06295, -0.75882, -0.13658, 1.78389, 0.30416, 0.36797, -0.69851, 0.13003, 1.2307,
    0.40868, -0.21081, -0.64073, 0.31061, 1.59554, 0.65718, 0.25429, -0.80789, 0.0824,
    1.78964, 0.54304, 0.41174, -1.06161, 0.07851, 2.01451, 0.49786, 0.91936, -0.9075,
    0.05941, 1.8312, 0.58722, 1.28777, -0.83711, -0.33248, 2.64983, 0.52698, 0.82132,
    -0.58897, -1.18223, 3.35809, 0.62017, 0.57353, -0.07276, -0.36214, 4.37655, 0.45481,
    0.21746, 0.10504, -0.61977, 3.54001, 0.04612, -0.18108, 0.63211, -0.87046, 2.47673,
    -0.48016, -1.27157, 0.86505, -1.11342, 1.24612, -0.82385, -2.77082, 1.23606, -1.59529,
    0.10438, -1.30206, -4.1152, 5.62596, -2.758,
)
# fmt: on

_WR_ARR = np.asarray(_WR, dtype=np.float64)
_WC_ARR = np.asarray(_WC, dtype=np.float64)
_WIN_SCORE = 99_999_999.0  # pubeval's sentinel: all mover men off (a won position)


def _to_pubeval_board(state: EnvState, mover: Player) -> list[int]:
    """Map an :class:`EnvState` into pubeval's 28-int ``pos[]`` from ``mover``'s POV.

    The mover's checkers are positive and move toward pubeval point 1; the opponent's
    are negative. Our WHITE already moves toward index 0 (so ``pos[i+1] = board[i]``);
    a BLACK mover is reflected (``point i -> 24-i``) and sign-flipped so it, too, reads
    as "moving toward 1". ``bar``/``off`` are indexed ``(white, black)``.
    """
    pos = [0] * 28
    board = state.board
    opp = mover.opponent()
    if mover is Player.WHITE:
        for i in range(24):
            pos[i + 1] = board[i]
    else:
        for i in range(24):
            pos[24 - i] = -board[i]
    pos[25] = state.bar[mover]  # mover bar (positive)
    pos[0] = -state.bar[opp]  # opponent bar (negative)
    pos[26] = state.off[mover]  # mover off (positive)
    pos[27] = -state.off[opp]  # opponent off (negative)
    return pos


def _pubeval_score(race: bool, pos: list[int]) -> float:
    """Tesauro's ``pubeval``: ``W · setx(pos)`` (a verbatim port of ``pubeval.c``)."""
    if pos[26] == 15:  # all mover men off — a win, the best possible move
        return _WIN_SCORE
    arr = np.asarray(pos, dtype=np.float64)
    # Points in setx order: x-block j uses pos[24-j] (i.e. pos[24], pos[23], ..., pos[1]).
    p = arr[1:25][::-1]
    blocks = np.zeros((24, 5))
    blocks[:, 0] = p == -1  # opponent blot
    blocks[:, 1] = p == 1  # mover blot
    blocks[:, 2] = p >= 2  # mover-made point
    blocks[:, 3] = p == 3  # mover exactly 3
    blocks[:, 4] = np.where(p >= 4, (p - 3) / 2.0, 0.0)  # mover spares beyond 3
    x = np.empty(122)
    x[:120] = blocks.ravel()
    x[120] = -arr[0] / 2.0  # opponent men on the bar
    x[121] = arr[26] / 15.0  # mover men borne off
    return float(x @ (_WR_ARR if race else _WC_ARR))


def _is_race(state: EnvState) -> bool:
    """True when the two sides are out of contact (no hits possible).

    Symmetric in the position: WHITE moves toward 0 and BLACK toward 23, so they have
    passed each other once every WHITE point sits below every BLACK point. Men on the
    bar always imply contact (they must re-enter through the opponent's home).
    """
    if state.bar[Player.WHITE] or state.bar[Player.BLACK]:
        return False
    white_pts = [i for i, n in enumerate(state.board) if n > 0]
    black_pts = [i for i, n in enumerate(state.board) if n < 0]
    if not white_pts or not black_pts:
        return True
    return max(white_pts) < min(black_pts)


class PubevalAgent:
    """A fixed (non-learning) :class:`~bgrl.agents.base.Agent` that plays by pubeval.

    Scores each legal afterstate from the mover's POV and plays the maximum — the
    canonical benchmark opponent. Deterministic (no RNG, no net); the ``race`` flag is
    read from the position before the move, per pubeval's contract.
    """

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move:
        mover = state.turn
        race = _is_race(state)
        best_move, best_score = legal[0][0], -np.inf
        for move, after in legal:
            score = _pubeval_score(race, _to_pubeval_board(after, mover))
            if score > best_score:
                best_score, best_move = score, move
        return best_move
