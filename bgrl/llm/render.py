"""Pluggable board serialisations — the model's "view" of the position.

Board rendering is a swept knob: ASCII board, compact pip list, moves-only, and a
placeholder position-id. Each is a :class:`BoardRenderer` (a ``name`` plus
``render(state, dice, mover) -> str``) so the harness can compare them head-to-head.

**Everything is mover-relative** (CLAUDE.md §6): points are numbered ``1..24`` from
the *mover's* ace point, the mover's checkers are ``X`` moving ``24 -> 1`` toward
bearing off, the opponent's are ``O``. WHITE's point ``p`` is absolute index ``p-1``;
BLACK's is ``24-p``. :func:`describe_move` uses the same numbering so a candidate like
``24/18 13/11`` reads in the mover's own frame. This mirrors the perspective flip in
:func:`bgrl.env.encode` but emits text rather than features.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bgrl.env import BAR, OFF, Dice, EnvState, Move, Player, SubMove

YOU_MARK = "X"
OPP_MARK = "O"


@runtime_checkable
class BoardRenderer(Protocol):
    """Serialises a position to text from the mover's point of view."""

    name: str

    def render(self, state: EnvState, dice: Dice, mover: Player) -> str: ...


# ------------------------------------------------------------------ shared primitives


def point_number(abs_idx: int, mover: Player) -> int:
    """Absolute point index ``0..23`` -> mover-relative point number ``1..24``."""
    return abs_idx + 1 if mover is Player.WHITE else 24 - abs_idx


def describe_move(move: Move, mover: Player) -> str:
    """Render one :class:`Move` in mover-relative notation, e.g. ``"24/18 13/11"``.

    Bar entry shows ``bar/<pt>`` and bearing off shows ``<pt>/off``; an empty move
    (a forced pass) renders as ``"(pass)"``.
    """
    if not move.submoves:
        return "(pass)"
    return " ".join(_describe_submove(sm, mover) for sm in move.submoves)


def _describe_submove(sm: SubMove, mover: Player) -> str:
    src = "bar" if sm.src == BAR else str(point_number(sm.src, mover))
    dst = "off" if sm.dst == OFF else str(point_number(sm.dst, mover))
    return f"{src}/{dst}"


def _occupancy(state: EnvState, mover: Player) -> tuple[dict[int, int], dict[int, int]]:
    """``(yours, theirs)``: point-number ``1..24`` -> checker count, mover-relative."""
    yours: dict[int, int] = {}
    theirs: dict[int, int] = {}
    for p in range(1, 25):
        abs_idx = p - 1 if mover is Player.WHITE else 24 - p
        signed = state.board[abs_idx]
        you = signed if mover is Player.WHITE else -signed
        if you > 0:
            yours[p] = you
        elif you < 0:
            theirs[p] = -you
    return yours, theirs


def pip_count(state: EnvState, side: Player) -> int:
    """``side``'s pip count: sum of point-number x checkers, with the bar at 25."""
    total = 25 * state.bar[side]
    for p in range(1, 25):
        abs_idx = p - 1 if side is Player.WHITE else 24 - p
        signed = state.board[abs_idx]
        cnt = signed if side is Player.WHITE else -signed
        if cnt > 0:
            total += p * cnt
    return total


# --------------------------------------------------------------------------- renderers


class PipListRenderer:
    """Compact, low-token listing of both sides' points plus bar/off and pip totals."""

    name = "pip_list"

    def render(self, state: EnvState, dice: Dice, mover: Player) -> str:
        yours, theirs = _occupancy(state, mover)
        opp = mover.opponent()
        lines = [
            f"Your checkers ({YOU_MARK}), moving 24->1 toward home: {_fmt_points(yours)}",
            f"Opponent checkers ({OPP_MARK}): {_fmt_points(theirs)}",
            f"Your bar: {state.bar[mover]}, off: {state.off[mover]}",
            f"Opponent bar: {state.bar[opp]}, off: {state.off[opp]}",
            f"Pip count - you: {pip_count(state, mover)}, opponent: {pip_count(state, opp)}",
        ]
        return "\n".join(lines)


def _fmt_points(points: dict[int, int]) -> str:
    if not points:
        return "(none)"
    return ", ".join(f"{p}:{points[p]}" for p in sorted(points, reverse=True))


class AsciiBoardRenderer:
    """A classic two-half ASCII board, mover-relative (``X`` = you, ``O`` = opponent).

    Top half holds points 13-24, bottom half 12-1, split by the bar. A point shows
    one mark per checker up to five; a sixth-or-more checker collapses the outermost
    row to the count. Bar and off are summarised on the trailing lines.
    """

    name = "ascii"

    _TOP = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    _BOTTOM = (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

    def render(self, state: EnvState, dice: Dice, mover: Player) -> str:
        yours, theirs = _occupancy(state, mover)
        opp = mover.opponent()
        rows = [_header(self._TOP)]
        rows += [_board_row(self._TOP, r, yours, theirs) for r in range(5)]
        rows.append("   " + "|" + "-" * 19 + "BAR" + "-" * 19 + "|")
        rows += [_board_row(self._BOTTOM, r, yours, theirs) for r in range(4, -1, -1)]
        rows.append(_header(self._BOTTOM))
        rows.append(
            f"bar: you {state.bar[mover]}, opp {state.bar[opp]}   "
            f"off: you {state.off[mover]}, opp {state.off[opp]}"
        )
        rows.append(f"pip: you {pip_count(state, mover)}, opp {pip_count(state, opp)}")
        return "\n".join(rows)


def _header(points: tuple[int, ...]) -> str:
    return "   " + " ".join(f"{p:>2}" for p in points)


def _board_row(
    points: tuple[int, ...], row: int, yours: dict[int, int], theirs: dict[int, int]
) -> str:
    return "   " + " ".join(_cell(p, row, yours, theirs) for p in points)


def _cell(p: int, row: int, yours: dict[int, int], theirs: dict[int, int]) -> str:
    you, them = yours.get(p, 0), theirs.get(p, 0)
    count = you or them
    if count == 0:
        return " ."
    mark = YOU_MARK if you else OPP_MARK
    if count > 5 and row == 4:  # collapse the outermost row to the overflow count
        return f"{count:>2}"
    return f" {mark}" if row < count else "  "


class MoveListRenderer:
    """Moves-only: no board, to test how much positional context the model needs.

    The enumerated candidate list (added by the prompt) carries the moves; this
    renderer deliberately withholds the board so the contribution of board context
    can be isolated in the sweep.
    """

    name = "moves_only"

    def render(self, state: EnvState, dice: Dice, mover: Player) -> str:
        return "(board representation omitted; choose from the listed candidate moves)"


class PositionIdRenderer:
    """A compact, deterministic position id (mover-relative signed point string).

    PLACEHOLDER: not gnubg's Position ID — that bit layout lives in WP3's
    serialization and is wired in post-merge (do not hand-roll it here). This emits a
    custom ``BGR`` token good enough to test whether an id-style format helps at all.
    """

    name = "position_id"

    def render(self, state: EnvState, dice: Dice, mover: Player) -> str:
        yours, theirs = _occupancy(state, mover)
        cells = []
        for p in range(24, 0, -1):  # mover point 24 -> 1
            cells.append(str(yours.get(p, 0) or -theirs.get(p, 0)))
        opp = mover.opponent()
        body = ",".join(cells)
        return (
            f"BGR:{body}"
            f"|bar:{state.bar[mover]},{state.bar[opp]}"
            f"|off:{state.off[mover]},{state.off[opp]}"
        )


ALL_RENDERERS: dict[str, BoardRenderer] = {
    r.name: r
    for r in (
        AsciiBoardRenderer(),
        PipListRenderer(),
        MoveListRenderer(),
        PositionIdRenderer(),
    )
}
"""Registry of renderers by ``name``, for the CLI/sweep to select by string."""
