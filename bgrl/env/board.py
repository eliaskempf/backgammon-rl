"""Pure helpers over the absolute, signed board representation.

These are the shared primitives the move generator and the single-movement
``apply_submove`` build on: movement direction, the home quadrant, bar-entry
points, landing legality, and checker accounting.
"""

from __future__ import annotations

from .types import EnvState, Player

NUM_POINTS = 24
CHECKERS_PER_SIDE = 15

# Movement direction along the absolute index: WHITE decreases, BLACK increases.
_DIRECTION = {Player.WHITE: -1, Player.BLACK: 1}
# Home quadrant: every checker must sit here before bearing off.
_HOME = {Player.WHITE: range(0, 6), Player.BLACK: range(18, 24)}
# Virtual bar position: a checker entering with die ``d`` lands at start + dir*d
# (WHITE: 24 - d => points 23..18; BLACK: d - 1 => points 0..5).
_ENTRY_START = {Player.WHITE: NUM_POINTS, Player.BLACK: -1}


def sign(player: Player) -> int:
    """+1 for WHITE, -1 for BLACK — the contribution of one checker to a point."""
    return 1 if player is Player.WHITE else -1


def direction(player: Player) -> int:
    """Index step of a forward move (-1 WHITE, +1 BLACK)."""
    return _DIRECTION[player]


def home_points(player: Player) -> range:
    """The six home-board point indices for ``player``."""
    return _HOME[player]


def entry_point(player: Player, die: int) -> int:
    """Absolute index a bar checker lands on when entering with ``die`` (1..6)."""
    return _ENTRY_START[player] + _DIRECTION[player] * die


def count_at(board: tuple[int, ...], point: int, player: Player) -> int:
    """Number of ``player`` checkers on ``point`` (0 if empty or opponent)."""
    v = board[point]
    if player is Player.WHITE:
        return v if v > 0 else 0
    return -v if v < 0 else 0


def can_land(board: tuple[int, ...], point: int, player: Player) -> bool:
    """True if ``player`` may move a checker onto ``point``.

    Legal targets are empty points, points the player already owns, and a lone
    opponent blot (which would be hit). Two or more opponent checkers block.
    """
    v = board[point] * sign(player)  # >0 own count, <0 opponent count, 0 empty
    return v >= -1


def all_home(state: EnvState, player: Player) -> bool:
    """True if every ``player`` checker is in the home board (bear-off allowed).

    Checkers on the bar count as outside the home board.
    """
    if state.bar[player] > 0:
        return False
    s = sign(player)
    in_home = sum(c for p in _HOME[player] if (c := state.board[p] * s) > 0)
    return in_home == CHECKERS_PER_SIDE - state.off[player]


def total_checkers(state: EnvState, player: Player) -> int:
    """Total ``player`` checkers across board + bar + off (15 for a valid state)."""
    s = sign(player)
    on_board = sum(c for p in range(NUM_POINTS) if (c := state.board[p] * s) > 0)
    return on_board + state.bar[player] + state.off[player]


def initial_board() -> tuple[int, ...]:
    """The standard opening checker layout in absolute, signed coordinates."""
    b = [0] * NUM_POINTS
    # BLACK (negative): 24-point/13-point/8-point/6-point in BLACK's numbering.
    b[0] = -2
    b[11] = -5
    b[16] = -3
    b[18] = -5
    # WHITE (positive): mirror image.
    b[5] = 5
    b[7] = 3
    b[12] = 5
    b[23] = 2
    return tuple(b)
