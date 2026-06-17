"""FastAPI app factory for the browser play server.

Thin HTTP glue over :class:`~bgrl.web.session.GameSession`: each handler resolves a
session, mutates it under the session lock, and projects the result through
:mod:`bgrl.web.views`. All move legality lives in the env behind the session; the
handlers only translate between HTTP and that boundary, mapping game-state conflicts
to ``409`` and unknown ids to ``404``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from bgrl.env import RandomDiceSource, SubMove
from bgrl.web.agents import UnknownOpponent, list_checkpoints, make_opponent
from bgrl.web.schemas import (
    AgentMoveResponse,
    CheckpointsResponse,
    GameIdRequest,
    LegalMovesResponse,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    NewGameResponse,
    RollResponse,
)
from bgrl.web.session import GameError, GameSession, SessionStore, make_seed_streams
from bgrl.web.views import color_of, move_view, outcome_view, player_of, state_view

DEFAULT_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    checkpoints_dir: Path | str = Path("runs"),
    static_dir: Path | str | None = None,
) -> FastAPI:
    """Build the play-server app. ``checkpoints_dir`` is scanned for opponent nets."""
    ckpt_dir = Path(checkpoints_dir)
    store = SessionStore()
    app = FastAPI(title="bgrl play server", version="0.1.0")

    def get_session(game_id: str) -> GameSession:
        try:
            return store.get(game_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown game_id") from None

    @app.get("/checkpoints", response_model=CheckpointsResponse)
    def checkpoints() -> CheckpointsResponse:
        return CheckpointsResponse(checkpoints=list_checkpoints(ckpt_dir))

    @app.post("/new_game", response_model=NewGameResponse)
    def new_game(req: NewGameRequest) -> NewGameResponse:
        dice_rng, agent_rng = make_seed_streams(req.seed)
        try:
            opponent = make_opponent(req.opponent, checkpoints_dir=ckpt_dir, rng=agent_rng)
        except UnknownOpponent:
            raise HTTPException(400, f"unknown opponent {req.opponent!r}") from None
        session = store.create(
            opponent=opponent,
            opponent_name=req.opponent,
            human_seat=player_of(req.human_color),
            dice_source=RandomDiceSource(dice_rng),
        )
        return NewGameResponse(
            game_id=session.game_id,
            state=state_view(session.state),
            human_color=req.human_color,
            opponent=req.opponent,
            to_act=color_of(session.to_act),
            needs_roll=session.needs_roll,
        )

    @app.post("/roll", response_model=RollResponse)
    def roll(req: GameIdRequest) -> RollResponse:
        session = get_session(req.game_id)
        with session.lock:
            if not session.human_to_move:
                raise HTTPException(409, "not the human's turn to roll")
            try:
                dice = session.roll()
            except GameError as exc:
                raise HTTPException(409, str(exc)) from None
            auto_pass = not session.legal
            n_legal = len(session.legal)
            if auto_pass:
                session.apply_pass()
            return RollResponse(
                dice=dice,
                to_act=color_of(session.to_act),
                auto_pass=auto_pass,
                n_legal=n_legal,
                state=state_view(session.state),
                needs_roll=session.needs_roll,
                terminal=session.terminal,
                outcome=outcome_view(session.outcome),
            )

    @app.get("/legal_moves", response_model=LegalMovesResponse)
    def legal_moves(game_id: str) -> LegalMovesResponse:
        session = get_session(game_id)
        with session.lock:
            mover = session.state.turn
            moves = [move_view(i, m, a, mover) for i, (m, a) in enumerate(session.legal)]
            return LegalMovesResponse(dice=session.dice, moves=moves)

    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest) -> MoveResponse:
        session = get_session(req.game_id)
        with session.lock:
            if not session.human_to_move:
                raise HTTPException(409, "not the human's turn")
            if session.dice is None:
                raise HTTPException(409, "roll before moving")
            if req.move_id is None and req.submoves is None:
                raise HTTPException(422, "provide move_id or submoves")
            try:
                if req.move_id is not None:
                    chosen = session.move_for_id(req.move_id)
                else:
                    assert req.submoves is not None
                    chosen = session.move_for_submoves(
                        [SubMove(s.src, s.dst) for s in req.submoves]
                    )
                session.apply_move(chosen)
            except GameError as exc:
                raise HTTPException(409, str(exc)) from None
            return MoveResponse(
                ok=True,
                state=state_view(session.state),
                to_act=color_of(session.to_act),
                needs_roll=session.needs_roll,
                terminal=session.terminal,
                outcome=outcome_view(session.outcome),
            )

    @app.post("/agent_move", response_model=AgentMoveResponse)
    def agent_move(req: GameIdRequest) -> AgentMoveResponse:
        session = get_session(req.game_id)
        with session.lock:
            if session.terminal:
                raise HTTPException(409, "game is over")
            if session.human_to_move:
                raise HTTPException(409, "it is the human's turn")
            try:
                session.play_agent()
            except GameError as exc:
                raise HTTPException(409, str(exc)) from None
            last = session.steps[-1]
            mover = last.state.turn
            played = move_view(0, last.move, last.afterstate, mover) if last.move.submoves else None
            return AgentMoveResponse(
                move=played,
                dice=last.dice,
                state=state_view(session.state),
                to_act=color_of(session.to_act),
                needs_roll=session.needs_roll,
                terminal=session.terminal,
                outcome=outcome_view(session.outcome),
            )

    @app.post("/export_mat")
    def export_mat(req: GameIdRequest) -> dict[str, str]:
        get_session(req.game_id)  # 404 if unknown
        raise HTTPException(501, ".mat export lands in WP3 Part B")

    static_path = Path(static_dir) if static_dir is not None else DEFAULT_STATIC_DIR
    if static_path.is_dir():
        # Mounted last so the API routes above take precedence over static files.
        app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

    return app
