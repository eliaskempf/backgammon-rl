"""Pure projections from frozen env value types to the API view models.

No side effects, no I/O — just shape translation, so these are trivially unit-tested.
The board stays in **absolute** coordinates (CLAUDE.md §6): we never flip it to a
mover-relative view here; the frontend renders from absolute indices.

``move_notation`` is a *display* helper only. The authoritative Jellyfish ``.mat``
notation (with hit marks and the match header) lives in
:mod:`bgrl.serialization.mat` — not this.
"""

from __future__ import annotations

from bgrl.env import BAR, OFF, Dice, EnvState, Move, Outcome, Player, SubMove, move_dice
from bgrl.web.schemas import (
    CheckerCounts,
    Color,
    CubeView,
    MoveView,
    OutcomeView,
    StateView,
    SubmoveView,
)


def color_of(player: Player) -> Color:
    return "white" if player is Player.WHITE else "black"


def player_of(color: Color) -> Player:
    return Player.WHITE if color == "white" else Player.BLACK


def point_number(point: int, mover: Player) -> int:
    """Absolute board index ``0..23`` -> standard 1..24 point in the mover's numbering.

    WHITE bears off past index 0, so its ace (1) point is index 0 -> ``point + 1``.
    BLACK bears off past index 23, so its ace point is index 23 -> ``24 - point``.
    The result equals the checker's pip distance to bearing off.
    """
    return point + 1 if mover is Player.WHITE else 24 - point


def _token(square: int, mover: Player) -> str:
    if square == BAR:
        return "bar"
    if square == OFF:
        return "off"
    return str(point_number(square, mover))


def move_notation(move: Move, mover: Player) -> str:
    """Human-legible move string, e.g. ``8/5 6/5``, ``bar/20``, ``6/off``.

    Display only (see module docstring). An empty move (a forced pass) renders as
    ``(no move)``.
    """
    if not move.submoves:
        return "(no move)"
    return " ".join(f"{_token(sm.src, mover)}/{_token(sm.dst, mover)}" for sm in move.submoves)


def state_view(state: EnvState) -> StateView:
    return StateView(
        board=list(state.board),
        bar=CheckerCounts(white=state.bar[Player.WHITE], black=state.bar[Player.BLACK]),
        off=CheckerCounts(white=state.off[Player.WHITE], black=state.off[Player.BLACK]),
        turn=color_of(state.turn),
        cube=CubeView(
            value=state.cube_value,
            owner=None if state.cube_owner is None else color_of(state.cube_owner),
        ),
    )


def submove_view(submove: SubMove, die: int | None = None) -> SubmoveView:
    return SubmoveView(src=submove.src, dst=submove.dst, die=die)


def move_view(
    move_id: int, move: Move, afterstate: EnvState, state: EnvState, dice: Dice
) -> MoveView:
    """Project ``move`` (played from ``state`` with ``dice``) into a ``MoveView``.

    ``state`` is the pre-move position (``state.turn`` is the mover); ``dice`` lets
    each submove carry the die it consumes (see :func:`bgrl.env.move_dice`).
    """
    dice_used = move_dice(state, dice, move)
    return MoveView(
        id=move_id,
        submoves=[submove_view(sm, d) for sm, d in zip(move.submoves, dice_used, strict=True)],
        notation=move_notation(move, state.turn),
        afterstate=state_view(afterstate),
    )


def outcome_view(outcome: Outcome | None) -> OutcomeView | None:
    if outcome is None:
        return None
    return OutcomeView(winner=color_of(outcome.winner), kind=outcome.kind.name.lower())  # type: ignore[arg-type]
