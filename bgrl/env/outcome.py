"""Terminal detection and win-magnitude (single / gammon / backgammon).

The reference env only signals "WHITE won"; we classify the magnitude here so
training targets and gnubg export are faithful from day one.
"""

from __future__ import annotations

from collections.abc import Iterable

from .board import CHECKERS_PER_SIDE, count_at, home_points
from .types import EnvState, Outcome, Player, WinKind


def is_terminal(state: EnvState) -> bool:
    """True once either side has borne off all 15 checkers."""
    return (
        state.off[Player.WHITE] == CHECKERS_PER_SIDE or state.off[Player.BLACK] == CHECKERS_PER_SIDE
    )


def _has_checker_in(state: EnvState, player: Player, points: Iterable[int]) -> bool:
    return any(count_at(state.board, p, player) > 0 for p in points)


def outcome(state: EnvState) -> Outcome | None:
    """Winner + magnitude, or ``None`` if the game is not over.

    * single — the loser has borne off at least one checker;
    * gammon — the loser has borne off none;
    * backgammon — the loser has borne off none *and* still has a checker on the
      bar or in the winner's home quadrant.
    """
    if state.off[Player.WHITE] == CHECKERS_PER_SIDE:
        winner = Player.WHITE
    elif state.off[Player.BLACK] == CHECKERS_PER_SIDE:
        winner = Player.BLACK
    else:
        return None

    loser = winner.opponent()
    if state.off[loser] > 0:
        return Outcome(winner, WinKind.SINGLE)
    if state.bar[loser] > 0 or _has_checker_in(state, loser, home_points(winner)):
        return Outcome(winner, WinKind.BACKGAMMON)
    return Outcome(winner, WinKind.GAMMON)
