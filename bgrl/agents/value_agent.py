"""0-ply greedy value agent: the shared value-based move selector.

Every value-based method picks its move the same way at the leaf — enumerate the
legal afterstates, score each by *equity*, take the best. TD(λ) (WP1) learns the
net behind this; n-ply expectimax (WP2) wraps deeper search around the same 0-ply
evaluation; MCTS-with-net (WP5) uses it at the leaves. So this selection logic is
algorithm-agnostic and lives here, in WP0, as the canonical "a net becomes an
agent" glue. A learning agent (WP1) subclasses this to add ``observe_step``.

**Perspective — the sign-flip lives here (CLAUDE.md section 6, read twice).** An
afterstate's ``turn`` is the *opponent* (they move next), so
``encode(afterstate, afterstate.turn)`` and the net yield the **opponent's**
equity. The current mover wants the afterstate that is worst for the opponent,
i.e. the largest ``-equity``. Because :func:`~bgrl.nets.equity.equity` is
anti-symmetric, ``-equity(opponent_view)`` is exactly the current mover's own
equity, so selection collapses to a single ``argmax``.
"""

from __future__ import annotations

import numpy as np

from bgrl.env import Dice, EnvState, Move, encode
from bgrl.nets.base import ValueNet
from bgrl.nets.equity import CENTERED_CUBE, CubeContext, equity

_TIE_TOLERANCE = 1e-9


class ValueAgent:
    """Greedy 0-ply agent: pick the legal move maximising the mover's equity.

    With ``rng`` omitted, ties break deterministically (first maximal move) for
    reproducible evaluation. Pass an ``rng`` to break ties uniformly at random —
    useful for self-play variety; TD-Gammon otherwise relies on the dice for
    exploration and plays greedily.
    """

    def __init__(
        self,
        net: ValueNet,
        *,
        cube: CubeContext = CENTERED_CUBE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._net = net
        self._cube = cube
        self._rng = rng

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move:
        features = np.stack([encode(after, perspective=after.turn) for _, after in legal])
        opp_outcome = self._net.evaluate(features)  # (B, OUTCOME_DIM), opponent's POV
        mover_equity = -equity(opp_outcome, self._cube)  # (B,), current mover's POV (ndarray)
        return legal[self._select(mover_equity)][0]

    def win_probs(self, legal: list[tuple[Move, EnvState]]) -> np.ndarray:
        """P(the current mover wins) for each legal afterstate, ``(B,)``.

        An afterstate's ``turn`` is the opponent, so the net (evaluated from that
        perspective) gives the *opponent's* outcome; ``1 - p_win_opponent`` is the
        current mover's win probability. In v1 only ``p_win`` is trained, so this is
        exactly the trained quantity — a display-friendly companion to the equity
        :meth:`act` selects on. Used by the web layer to surface the move's value.
        """
        features = np.stack([encode(after, perspective=after.turn) for _, after in legal])
        opp_outcome = self._net.evaluate(features)  # (B, OUTCOME_DIM), opponent's POV
        return 1.0 - opp_outcome[:, 0]

    def win_prob(self, afterstate: EnvState) -> float:
        """P(the player who produced ``afterstate`` wins) — the single-state form.

        ``afterstate.turn`` is the opponent of that player, mirroring
        :meth:`win_probs`; useful for scoring one already-chosen move.
        """
        features = encode(afterstate, perspective=afterstate.turn)[None, :]
        opp_outcome = self._net.evaluate(features)
        return float(1.0 - opp_outcome[0, 0])

    def _select(self, equities: np.ndarray) -> int:
        """Index of the best equity; ``rng`` (if any) breaks ties uniformly."""
        if self._rng is None:
            return int(np.argmax(equities))
        top = np.flatnonzero(equities >= equities.max() - _TIE_TOLERANCE)
        return int(top[self._rng.integers(len(top))])
