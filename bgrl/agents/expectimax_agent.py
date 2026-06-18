"""n-ply expectimax search agent (WP2): deeper lookahead over the same value net.

A pure agent-layer add-on. The agent enumerates legal afterstates exactly like
:class:`~bgrl.agents.value_agent.ValueAgent`, but instead of scoring each at 0-ply it
looks ahead ``plies`` chance/decision levels, averaging over all 21 distinct dice rolls
at each chance node, and backs the values up. ``plies == 0`` reproduces ``ValueAgent``
move-for-move (gnubg ply convention: raw net = 0-ply, CLAUDE.md s9). Nothing in
``bgrl.env`` or ``bgrl.training`` is touched.

**Negamax over chance nodes.** ``_eval_pov(s, depth)`` is the equity of position ``s`` to
the player to move at ``s`` (``s.turn``). Every afterstate from
:func:`~bgrl.env.legal_moves` carries ``turn = opponent`` and
:func:`~bgrl.nets.equity.equity` is anti-symmetric, so each level negates its children: a
child's value-to-its-mover, negated, is the parent mover's value. This is exactly
:class:`ValueAgent`'s sign convention applied recursively (CLAUDE.md s6).

Terminal positions reached inside the search are scored *exactly* from the
:class:`~bgrl.env.Outcome` (never the net), faithfully including gammon/backgammon
magnitude. Optional ``top_k`` candidate pruning (score children at 0-ply, deepen only the
best) keeps deep plies tractable; it is an approximation, off by default. A per-move
transposition cache collapses positions reached by multiple roll/move orders.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgrl.agents.value_agent import ValueAgent
from bgrl.env import (
    Dice,
    EnvState,
    Move,
    Outcome,
    Player,
    encode,
    is_terminal,
    legal_moves,
    outcome,
)
from bgrl.nets.base import OUTCOME_DIM, ValueNet
from bgrl.nets.equity import CENTERED_CUBE, CubeContext, equity

_TIE_TOLERANCE = 1e-9


def _weighted_rolls() -> tuple[tuple[Dice, float], ...]:
    """The 21 distinct dice rolls with probabilities: doubles 1/36, non-doubles 2/36.

    Each non-double is listed once as ``(a, b)`` with ``a < b``; :func:`legal_moves`
    already explores both die orderings internally, so listing ``(b, a)`` too would
    double-count. The weights sum to exactly 1.
    """
    rolls: list[tuple[Dice, float]] = []
    for a in range(1, 7):
        for b in range(a, 7):
            rolls.append(((a, b), (1.0 if a == b else 2.0) / 36.0))
    return tuple(rolls)


_WEIGHTED_ROLLS = _weighted_rolls()


def _terminal_equity(result: Outcome, perspective: Player, cube: CubeContext) -> float:
    """Exact equity of a finished game to ``perspective`` (+/-1, 2, 3), via :func:`equity`.

    Builds the cumulative outcome 5-vector for ``result`` from ``perspective``'s point of
    view and reduces it through the shared :func:`~bgrl.nets.equity.equity`, so the
    win-magnitude scoring stays consistent with the net's equity reduction. Note that a
    single *loss* is the all-zeros vector: its -1 comes from the implied
    ``p_lose = 1 - p_win`` inside :func:`equity`, not from any explicit head.
    """
    vec = np.zeros(OUTCOME_DIM, dtype=np.float64)
    kind = int(result.kind)
    if result.winner is perspective:
        vec[0] = 1.0  # p_win
        if kind >= 2:
            vec[1] = 1.0  # p_win_gammon (cumulative: gammon or better)
        if kind >= 3:
            vec[2] = 1.0  # p_win_backgammon
    else:
        if kind >= 2:
            vec[3] = 1.0  # p_lose_gammon
        if kind >= 3:
            vec[4] = 1.0  # p_lose_backgammon
    return float(equity(vec, cube))


class ExpectimaxAgent:
    """Depth-limited expectimax over dice chance nodes, wrapping a ``ValueNet``.

    ``plies`` is the search depth in gnubg's convention (0 = raw net = ``ValueAgent``).
    ``top_k`` optionally prunes each decision node to its ``top_k`` 0-ply-best candidates
    before deepening (``None`` = exact search). ``cube`` and ``rng`` mirror
    :class:`~bgrl.agents.value_agent.ValueAgent`; ``rng`` breaks root ties uniformly
    (omitted = deterministic first-maximal, for reproducible evaluation).
    """

    def __init__(
        self,
        net: ValueNet,
        *,
        plies: int = 1,
        top_k: int | None = None,
        cube: CubeContext = CENTERED_CUBE,
        rng: np.random.Generator | None = None,
    ) -> None:
        if plies < 0:
            raise ValueError(f"plies must be >= 0, got {plies}")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
        self._net = net
        self._plies = plies
        self._top_k = top_k
        self._cube = cube
        self._rng = rng
        self._value_agent = ValueAgent(net, cube=cube, rng=rng)
        self._cache: dict[tuple[EnvState, int], float] = {}

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move:
        if self._plies == 0:  # raw net == ValueAgent (one batched evaluation)
            return self._value_agent.act(state, dice, legal)
        self._cache = {}  # the transposition cache is scoped to a single move
        afterstates = [after for _, after in legal]
        candidates = list(range(len(afterstates)))
        if self._top_k is not None and len(candidates) > self._top_k:
            shallow = self._leaf_mover_equities(afterstates)
            candidates = [int(i) for i in np.argpartition(-shallow, self._top_k)[: self._top_k]]
        values = np.array(
            [-self._eval_pov(afterstates[i], self._plies) for i in candidates],
            dtype=np.float64,
        )
        return legal[candidates[self._select(values)]][0]

    def _eval_pov(self, state: EnvState, depth: int) -> float:
        """Equity of ``state`` to the player to move at it (``state.turn``)."""
        if is_terminal(state):
            return self._terminal_value(state)
        if depth == 0:
            features = encode(state, perspective=state.turn)
            return float(equity(self._net.evaluate(features), self._cube))
        key = (state, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._chance_value(state, depth)
        self._cache[key] = value
        return value

    def _chance_value(self, state: EnvState, depth: int) -> float:
        """Expected equity to ``state.turn``: average over the 21 rolls of its best reply."""
        total = 0.0
        for roll, weight in _WEIGHTED_ROLLS:
            replies = legal_moves(state, roll)
            if replies:  # state.turn picks the reply best for itself
                roll_value = self._best_value_to_mover([a for _, a in replies], depth - 1)
            else:  # forced pass: board unchanged, turn flips, a ply is still spent
                passed = replace(state, turn=state.turn.opponent())
                roll_value = -self._eval_pov(passed, depth - 1)
            total += weight * roll_value
        return total

    def _best_value_to_mover(self, children: list[EnvState], child_depth: int) -> float:
        """Max over ``children`` of value to the just-moved mover (``-eval_pov(child)``)."""
        if child_depth == 0:  # batched leaf layer (the dominant cost)
            return float(self._leaf_mover_equities(children).max())
        states = children
        if self._top_k is not None and len(states) > self._top_k:
            shallow = self._leaf_mover_equities(states)
            keep = np.argpartition(-shallow, self._top_k)[: self._top_k]
            states = [states[int(i)] for i in keep]
        return max(-self._eval_pov(child, child_depth) for child in states)

    def _leaf_mover_equities(self, states: list[EnvState]) -> np.ndarray:
        """0-ply value to the just-moved mover for each afterstate (terminals scored exact).

        Mirrors :class:`ValueAgent`: encode each afterstate from its ``turn`` (opponent)
        POV, batch through the net, negate the opponent's equity. Terminal afterstates are
        scored from the outcome instead of the net.
        """
        out = np.empty(len(states), dtype=np.float64)
        rows: list[int] = []
        features: list[np.ndarray] = []
        for i, state in enumerate(states):
            if is_terminal(state):
                out[i] = -self._terminal_value(state)
            else:
                rows.append(i)
                features.append(encode(state, perspective=state.turn))
        if features:
            opp_outcome = self._net.evaluate(np.stack(features))
            out[rows] = -equity(opp_outcome, self._cube)
        return out

    def _terminal_value(self, state: EnvState) -> float:
        """Exact equity to ``state.turn`` at a terminal afterstate (it is the loser)."""
        result = outcome(state)
        assert result is not None, "caller guarantees is_terminal(state)"
        return _terminal_equity(result, state.turn, self._cube)

    def _select(self, equities: np.ndarray) -> int:
        """Index of the best equity; ``rng`` (if any) breaks ties uniformly."""
        if self._rng is None:
            return int(np.argmax(equities))
        top = np.flatnonzero(equities >= equities.max() - _TIE_TOLERANCE)
        return int(top[self._rng.integers(len(top))])
