"""The single source of truth for applying one checker movement.

Every place that mutates a board — the move generator, the eventual web UI, and
gnubg replay — goes through :func:`apply_submove`, so hitting / bar-entry /
bear-off logic lives in exactly one spot.
"""

from __future__ import annotations

from .board import sign
from .types import BAR, OFF, EnvState, Player, SubMove


def apply_submove(state: EnvState, mover: Player, sm: SubMove) -> EnvState:
    """Return the state after ``mover`` plays one (assumed legal) submove.

    Handles bar entry (``src == BAR``), bearing off (``dst == OFF``) and hitting
    a lone opponent blot (sent to the opponent's bar). Does **not** change whose
    turn it is — turn flips once when the full play's afterstate is finalised —
    and does **not** validate legality; the move generator guarantees that.
    """
    board = list(state.board)
    bar = list(state.bar)
    off = list(state.off)
    s = sign(mover)
    opp = mover.opponent()

    # Remove the checker from its source (bar or a point).
    if sm.src == BAR:
        bar[mover] -= 1
    else:
        board[sm.src] -= s

    # Place the checker at its destination (off the board, or onto a point).
    if sm.dst == OFF:
        off[mover] += 1
    elif board[sm.dst] == -s:  # exactly one opponent checker -> hit it
        board[sm.dst] = s
        bar[opp] += 1
    else:
        board[sm.dst] += s

    return EnvState(
        board=tuple(board),
        bar=(bar[0], bar[1]),
        off=(off[0], off[1]),
        turn=state.turn,
        cube_value=state.cube_value,
        cube_owner=state.cube_owner,
    )
