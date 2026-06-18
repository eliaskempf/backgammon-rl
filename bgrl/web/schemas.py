"""The REST API contract, as pydantic models.

These models *are* the stable contract the disposable frontend binds to (the plan's
load-bearing piece). FastAPI uses them for request validation and to emit OpenAPI
docs. The view models (``StateView``/``MoveView``/``OutcomeView``) are built from
frozen env objects by :mod:`bgrl.web.views`; everything is absolute-coordinate and
never mover-relative (CLAUDE.md §6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

Color = Literal["white", "black"]
WinKindName = Literal["single", "gammon", "backgammon"]


# --- shared view objects -------------------------------------------------------


class CheckerCounts(BaseModel):
    """Per-colour checker counts (used for both the bar and the off tray)."""

    white: int
    black: int


class CubeView(BaseModel):
    """Reserved cube state (always centered in v1, but carried in the contract)."""

    value: int
    owner: Color | None


class StateView(BaseModel):
    """JSON projection of an ``EnvState`` in absolute coordinates."""

    board: list[int]  # length 24, signed (positive = white, negative = black)
    bar: CheckerCounts
    off: CheckerCounts
    turn: Color
    cube: CubeView


class SubmoveView(BaseModel):
    """One checker movement; ``src``/``dst`` use the env sentinels (BAR=-1, OFF=-2).

    ``die`` is the die value this submove consumes — populated on responses (so the
    UI can move a checker "by the left die"); ignored on request bodies.
    """

    src: int
    dst: int
    die: int | None = None


class MoveView(BaseModel):
    """A full legal play. ``id`` indexes the session's cached legal list for this roll."""

    id: int
    submoves: list[SubmoveView]
    notation: str
    afterstate: StateView


class OutcomeView(BaseModel):
    """How a finished game ended (winner + magnitude)."""

    winner: Color
    kind: WinKindName


class CheckpointInfo(BaseModel):
    """A selectable opponent checkpoint and a little of its metadata for the picker."""

    name: str
    trained_with: str | None = None
    games_trained: int | None = None
    created_at: str | None = None
    win_rate: float | None = None  # eval win rate vs `eval_opponent`, if recorded
    eval_opponent: str | None = None  # who `win_rate` was measured against (e.g. "pubeval")


# --- requests ------------------------------------------------------------------


class NewGameRequest(BaseModel):
    human_color: Color = "white"
    opponent: str = "random"
    seed: int | None = None
    manual_dice: bool = False  # human supplies every roll (both seats); see RollRequest


class GameIdRequest(BaseModel):
    game_id: str


class RollRequest(BaseModel):
    """A roll request that may carry the human-supplied dice for a manual-dice game.

    ``dice`` is required (and validated 1..6) only when the game was created with
    ``manual_dice=True``; for the default auto-rolled game it is omitted and the server
    draws the roll itself. Used by both ``/roll`` (human) and ``/agent_move`` (agent).
    """

    game_id: str
    dice: tuple[int, int] | None = None

    @field_validator("dice")
    @classmethod
    def _dice_in_range(cls, dice: tuple[int, int] | None) -> tuple[int, int] | None:
        if dice is not None and not all(1 <= d <= 6 for d in dice):
            raise ValueError("each die must be in 1..6")
        return dice


class MoveRequest(BaseModel):
    """Submit a human move by ``move_id`` (primary) or by an explicit submove list."""

    game_id: str
    move_id: int | None = None
    submoves: list[SubmoveView] | None = None


# --- responses -----------------------------------------------------------------


class NewGameResponse(BaseModel):
    game_id: str
    state: StateView
    human_color: Color
    opponent: str
    to_act: Color
    needs_roll: bool
    manual_dice: bool  # echoes the request so the UI knows whether to prompt for dice
    can_undo: bool = False  # whether a prior human decision exists to revert to


class RollResponse(BaseModel):
    dice: tuple[int, int]
    to_act: Color
    auto_pass: bool  # true when the roll had no legal move and the server passed
    n_legal: int
    state: StateView
    needs_roll: bool
    terminal: bool
    outcome: OutcomeView | None
    can_undo: bool = False


class LegalMovesResponse(BaseModel):
    dice: tuple[int, int] | None
    moves: list[MoveView]


class MoveResponse(BaseModel):
    ok: bool
    state: StateView
    to_act: Color
    needs_roll: bool
    terminal: bool
    outcome: OutcomeView | None
    # The human mover's win probability after this move per the opponent value net
    # (already "your win chance"); null when the opponent has no net (e.g. random).
    win_prob: float | None = None
    can_undo: bool = False


class AgentMoveResponse(BaseModel):
    move: MoveView | None  # None when the agent was forced to pass
    dice: tuple[int, int]
    state: StateView
    to_act: Color
    needs_roll: bool
    terminal: bool
    outcome: OutcomeView | None
    # The agent mover's win probability after its move per its own value net; the UI
    # shows the complement (``1 - win_prob``) so the readout stays "your win chance".
    win_prob: float | None = None
    can_undo: bool = False


class UndoResponse(BaseModel):
    """The position restored after reverting to the human's previous decision."""

    state: StateView
    to_act: Color
    dice: tuple[int, int] | None  # the same roll the human originally had
    needs_roll: bool
    terminal: bool
    outcome: OutcomeView | None
    moves: list[MoveView]  # the re-enumerated legal moves for the restored roll
    can_undo: bool = False


class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]


class ExportMatResponse(BaseModel):
    """A played game serialized to Jellyfish ``.mat`` text (gnubg-importable)."""

    filename: str  # suggested download name, e.g. ``game-<id>.mat``
    mat: str  # the full .mat file contents
