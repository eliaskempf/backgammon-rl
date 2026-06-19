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
    WEIGHTED_ROLLS,
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
from bgrl.nets.base import ValueNet
from bgrl.nets.equity import CENTERED_CUBE, CubeContext, equity, outcome_to_vector

_TIE_TOLERANCE = 1e-9


def _terminal_equity(result: Outcome, perspective: Player, cube: CubeContext) -> float:
    """Exact equity of a finished game to ``perspective`` (+/-1, 2, 3), via :func:`equity`.

    Reduces the cumulative outcome 5-vector (:func:`~bgrl.nets.equity.outcome_to_vector`)
    through the shared :func:`~bgrl.nets.equity.equity`, so the win-magnitude scoring
    stays consistent with the net's equity reduction.
    """
    return float(equity(outcome_to_vector(result, perspective), cube))


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

    def win_prob(self, afterstate: EnvState) -> float:
        """P(the player who produced ``afterstate`` wins), from the 0-ply net.

        A display companion to :meth:`act` for the web layer (mirrors
        :meth:`~bgrl.agents.value_agent.ValueAgent.win_prob`). Search depth does not
        change the net's own estimate of a *given* afterstate, so this delegates to
        the wrapped 0-ply agent.
        """
        return self._value_agent.win_prob(afterstate)

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
        for roll, weight in WEIGHTED_ROLLS:
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
