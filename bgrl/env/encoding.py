"""Tesauro 198-feature encoding, canonicalised to the mover's point of view.

Layout of the returned float32 vector (length :data:`N_FEATURES` = 200), indexed
in the *perspective* player's own orientation (their bear-off corner is rel point
0; WHITE rel k = absolute k, BLACK rel k = absolute ``23 - k``):

==========  ===========================================================
indices     meaning
==========  ===========================================================
0..95       mover occupancy: 24 rel points x 4 units (Tesauro thresholds)
96, 97      mover bar / 2,  mover off / 15
98..193     opponent occupancy: 24 rel points x 4 units
194, 195    opponent bar / 2,  opponent off / 15
196, 197    side-to-move flag [perspective_is_mover, perspective_is_not_mover]
198, 199    RESERVED for cube state (value, ownership); zero in v1
==========  ===========================================================

The 4-unit per-point scheme is ``[n>=1, n>=2, n>=3, max(0, n-3)/2]``.

This is mover-relative (not the reference's absolute WHITE-then-BLACK layout) so
the value net only ever learns one side — the perspective invariant of CLAUDE.md
section 6. The side-to-move flag encodes only *whether* the perspective player is
on move, never their colour, so a position and its colour-mirror encode
identically. Correctness of the move generator is cross-checked against the
reference on afterstates, not on this encoding.
"""

from __future__ import annotations

import numpy as np

from .board import NUM_POINTS, count_at
from .types import EnvState, Player

N_FEATURES_TESAURO = 198
N_RESERVED_CUBE = 2
N_FEATURES = N_FEATURES_TESAURO + N_RESERVED_CUBE  # 200
ENCODING_VERSION = 1

_MOVER_OCC = 0  # mover occupancy block start
_MOVER_BAR = 96
_MOVER_OFF = 97
_OPP_OCC = 98  # opponent occupancy block start
_OPP_BAR = 194
_OPP_OFF = 195
_TURN = 196  # two units
# 198, 199 reserved (cube)


def _write_point(out: np.ndarray, base: int, n: int) -> None:
    """Write the 4-unit Tesauro threshold encoding for ``n`` checkers."""
    if n >= 1:
        out[base] = 1.0
    if n >= 2:
        out[base + 1] = 1.0
    if n >= 3:
        out[base + 2] = 1.0
        out[base + 3] = (n - 3) / 2.0


def encode(state: EnvState, perspective: Player) -> np.ndarray:
    """Encode ``state`` from ``perspective``'s point of view (see module docs)."""
    out = np.zeros(N_FEATURES, dtype=np.float32)
    mover = perspective
    opp = perspective.opponent()
    white_pov = perspective is Player.WHITE
    board = state.board

    for k in range(NUM_POINTS):
        absp = k if white_pov else (NUM_POINTS - 1 - k)
        _write_point(out, _MOVER_OCC + 4 * k, count_at(board, absp, mover))
        _write_point(out, _OPP_OCC + 4 * k, count_at(board, absp, opp))

    out[_MOVER_BAR] = state.bar[mover] / 2.0
    out[_MOVER_OFF] = state.off[mover] / 15.0
    out[_OPP_BAR] = state.bar[opp] / 2.0
    out[_OPP_OFF] = state.off[opp] / 15.0
    out[_TURN if perspective is state.turn else _TURN + 1] = 1.0
    return out
