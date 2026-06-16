"""The ``Env`` facade — the stable contract surface over the env internals.

Agents and the training loop talk to this; the afterstate-first design means
every value-based algorithm consumes the same ``legal_moves`` enumeration.
"""

from __future__ import annotations

from .board import initial_board
from .movegen import legal_moves as _legal_moves
from .outcome import is_terminal as _is_terminal
from .outcome import outcome as _outcome
from .types import Dice, EnvState, Move, Outcome, Player


class Env:
    """Pure, stateless game logic. All methods are static."""

    @staticmethod
    def initial_state() -> EnvState:
        """The standard opening position, WHITE to move."""
        return EnvState(board=initial_board(), bar=(0, 0), off=(0, 0), turn=Player.WHITE)

    @staticmethod
    def legal_moves(state: EnvState, dice: Dice) -> list[tuple[Move, EnvState]]:
        """``(Move, afterstate)`` pairs for ``state.turn``; ``[]`` => the turn passes."""
        return _legal_moves(state, dice)

    @staticmethod
    def is_terminal(state: EnvState) -> bool:
        return _is_terminal(state)

    @staticmethod
    def outcome(state: EnvState) -> Outcome | None:
        """Winner + magnitude once terminal, else ``None``."""
        return _outcome(state)
