"""Game environment: state, move generation, afterstate enumeration, encoding.

``encode`` is added with the encoding module; everything else is exported here.
"""

from .apply import apply_submove
from .board import initial_board
from .dice import (
    WEIGHTED_ROLLS,
    DiceSource,
    ManualDiceSource,
    RandomDiceSource,
    ReplayDiceSource,
    roll_dice,
    weighted_rolls,
)
from .encoding import ENCODING_VERSION, N_FEATURES, encode
from .env import Env
from .movegen import legal_moves, legal_orderings, move_dice
from .outcome import is_terminal, outcome
from .types import BAR, OFF, Dice, EnvState, Move, Outcome, Player, SubMove, WinKind

__all__ = [
    "BAR",
    "ENCODING_VERSION",
    "N_FEATURES",
    "OFF",
    "WEIGHTED_ROLLS",
    "Dice",
    "DiceSource",
    "Env",
    "EnvState",
    "ManualDiceSource",
    "Move",
    "Outcome",
    "Player",
    "RandomDiceSource",
    "ReplayDiceSource",
    "SubMove",
    "WinKind",
    "apply_submove",
    "encode",
    "initial_board",
    "is_terminal",
    "legal_moves",
    "legal_orderings",
    "move_dice",
    "outcome",
    "roll_dice",
    "weighted_rolls",
]
