"""The REST API contract, as pydantic models.

These models *are* the stable contract the disposable frontend binds to (the plan's
load-bearing piece). FastAPI uses them for request validation and to emit OpenAPI
docs. The view models (``StateView``/``MoveView``/``OutcomeView``) are built from
frozen env objects by :mod:`bgrl.web.views`; everything is absolute-coordinate and
never mover-relative (CLAUDE.md §6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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
    """One checker movement; ``src``/``dst`` use the env sentinels (BAR=-1, OFF=-2)."""

    src: int
    dst: int


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


# --- requests ------------------------------------------------------------------


class NewGameRequest(BaseModel):
    human_color: Color = "white"
    opponent: str = "random"
    seed: int | None = None


class GameIdRequest(BaseModel):
    game_id: str


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


class RollResponse(BaseModel):
    dice: tuple[int, int]
    to_act: Color
    auto_pass: bool  # true when the roll had no legal move and the server passed
    n_legal: int
    state: StateView
    needs_roll: bool
    terminal: bool
    outcome: OutcomeView | None


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


class AgentMoveResponse(BaseModel):
    move: MoveView | None  # None when the agent was forced to pass
    dice: tuple[int, int]
    state: StateView
    to_act: Color
    needs_roll: bool
    terminal: bool
    outcome: OutcomeView | None


class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointInfo]
