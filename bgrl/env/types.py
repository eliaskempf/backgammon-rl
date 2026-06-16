"""Frozen value types for the backgammon environment.

Coordinates are ABSOLUTE: points are indexed ``0..23`` and the board is signed
(``board[i] > 0`` => that many WHITE checkers on point ``i``, ``< 0`` => BLACK,
``0`` => empty). A point never holds both colours. Mover-relative views are
produced only at the encoding boundary (see :mod:`bgrl.env.encoding`), never in
the stored state — that keeps a single canonical truth for export/debugging.

WHITE moves toward index 0 (home = points ``0..5``) and bears off past 0.
BLACK moves toward index 23 (home = points ``18..23``) and bears off past 23.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

Dice = tuple[int, int]

BAR: int = -1  # SubMove.src sentinel: a checker entering from the bar
OFF: int = -2  # SubMove.dst sentinel: a checker borne off


class Player(IntEnum):
    """The two sides. The int values double as indices into ``bar``/``off``."""

    WHITE = 0
    BLACK = 1

    def opponent(self) -> Player:
        return Player.BLACK if self is Player.WHITE else Player.WHITE


class WinKind(IntEnum):
    """Magnitude of a win, for cube-ready training targets and gnubg export."""

    SINGLE = 1
    GAMMON = 2
    BACKGAMMON = 3


@dataclass(frozen=True, slots=True)
class SubMove:
    """A single checker movement.

    ``src`` is a point index ``0..23`` or :data:`BAR`; ``dst`` is a point index
    ``0..23`` or :data:`OFF`.
    """

    src: int
    dst: int


@dataclass(frozen=True, slots=True)
class Move:
    """A full legal play for one ``(state, dice)``.

    ``submoves`` holds 1..4 :class:`SubMove` (4 only for doubles) in a canonical
    order, so two plays reaching the same board via the same checker movements
    compare equal.
    """

    submoves: tuple[SubMove, ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    """Who won and by how much, from the winner's identity (absolute)."""

    winner: Player
    kind: WinKind


@dataclass(frozen=True, slots=True)
class EnvState:
    """An immutable, hashable backgammon position (canonical absolute form).

    Equality/hash are structural over all fields, so equivalent positions are
    equal and afterstates can be de-duplicated by using the state as a dict key.
    """

    board: tuple[int, ...]  # length 24, signed checker counts
    bar: tuple[int, int]  # (white, black)
    off: tuple[int, int]  # (white, black)
    turn: Player
    cube_value: int = 1  # reserved (always 1 in v1)
    cube_owner: Player | None = None  # reserved (always centered/None in v1)
