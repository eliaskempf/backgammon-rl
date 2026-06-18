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

from bgrl.agents.expectimax_agent import ExpectimaxAgent
from bgrl.agents.value_agent import ValueAgent
from bgrl.env import EnvState, ManualDiceSource, Player, RandomDiceSource, SubMove
from bgrl.serialization import game_to_mat
from bgrl.web.agents import UnknownOpponent, list_checkpoints, make_opponent
from bgrl.web.schemas import (
    AgentMoveResponse,
    CheckpointsResponse,
    ExportMatResponse,
    GameIdRequest,
    LegalMovesResponse,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    NewGameResponse,
    RollRequest,
    RollResponse,
    UndoResponse,
)
from bgrl.web.session import GameError, GameSession, SessionStore, make_seed_streams
from bgrl.web.views import (
    color_of,
    legal_move_views,
    move_view,
    outcome_view,
    player_of,
    state_view,
)

DEFAULT_STATIC_DIR = Path(__file__).parent / "static"


def _win_prob(session: GameSession, afterstate: EnvState) -> float | None:
    """Mover's win probability for ``afterstate`` if the opponent has a value net.

    ``None`` for a netless opponent (e.g. random), in which case the UI hides the
    estimate. The mover here is the player who produced ``afterstate`` (its ``turn``
    is the opponent), so this is "your win chance" after a human move and the
    agent's win chance after an agent move — the caller frames it.
    """
    opponent = session.opponent
    if isinstance(opponent, ValueAgent | ExpectimaxAgent):
        return opponent.win_prob(afterstate)
    return None


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

    def prime_dice(session: GameSession, dice: tuple[int, int] | None) -> None:
        """Queue the human-entered roll for a manual game; reject a dice/mode mismatch."""
        if session.is_manual:
            if dice is None:
                raise HTTPException(422, "manual-dice game requires explicit dice for each roll")
            session.supply_dice(dice)
        elif dice is not None:
            raise HTTPException(409, "game is not in manual-dice mode; dice are auto-rolled")

    @app.get("/checkpoints", response_model=CheckpointsResponse)
    def checkpoints() -> CheckpointsResponse:
        return CheckpointsResponse(checkpoints=list_checkpoints(ckpt_dir))

    @app.post("/new_game", response_model=NewGameResponse)
    def new_game(req: NewGameRequest) -> NewGameResponse:
        dice_rng, agent_rng = make_seed_streams(req.seed)
        try:
            opponent = make_opponent(
                req.opponent,
                checkpoints_dir=ckpt_dir,
                rng=agent_rng,
                plies=req.expectimax_plies,
                top_k=req.expectimax_top_k,
            )
        except UnknownOpponent:
            raise HTTPException(400, f"unknown opponent {req.opponent!r}") from None
        dice_source = ManualDiceSource() if req.manual_dice else RandomDiceSource(dice_rng)
        session = store.create(
            opponent=opponent,
            opponent_name=req.opponent,
            human_seat=player_of(req.human_color),
            dice_source=dice_source,
        )
        return NewGameResponse(
            game_id=session.game_id,
            state=state_view(session.state),
            human_color=req.human_color,
            opponent=req.opponent,
            to_act=color_of(session.to_act),
            needs_roll=session.needs_roll,
            manual_dice=req.manual_dice,
            can_undo=session.can_undo,
        )

    @app.post("/roll", response_model=RollResponse)
    def roll(req: RollRequest) -> RollResponse:
        session = get_session(req.game_id)
        with session.lock:
            if not session.human_to_move:
                raise HTTPException(409, "not the human's turn to roll")
            prime_dice(session, req.dice)
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
                can_undo=session.can_undo,
            )

    @app.get("/legal_moves", response_model=LegalMovesResponse)
    def legal_moves(game_id: str) -> LegalMovesResponse:
        session = get_session(game_id)
        with session.lock:
            state, dice = session.state, session.dice
            moves = legal_move_views(state, dice, session.legal) if dice is not None else []
            return LegalMovesResponse(dice=dice, moves=moves)

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
            # session.state is now the human's afterstate, so this is "your win chance".
            return MoveResponse(
                ok=True,
                state=state_view(session.state),
                to_act=color_of(session.to_act),
                needs_roll=session.needs_roll,
                terminal=session.terminal,
                outcome=outcome_view(session.outcome),
                win_prob=_win_prob(session, session.state),
                can_undo=session.can_undo,
            )

    @app.post("/agent_move", response_model=AgentMoveResponse)
    def agent_move(req: RollRequest) -> AgentMoveResponse:
        session = get_session(req.game_id)
        with session.lock:
            if session.terminal:
                raise HTTPException(409, "game is over")
            if session.human_to_move:
                raise HTTPException(409, "it is the human's turn")
            prime_dice(session, req.dice)
            try:
                session.play_agent()
            except GameError as exc:
                raise HTTPException(409, str(exc)) from None
            last = session.steps[-1]
            if last.move.submoves:
                played = move_view(0, last.move, last.afterstate, last.state, last.dice)
                win_prob = _win_prob(session, last.afterstate)  # the agent's win chance
            else:
                played = None  # forced pass
                win_prob = None
            return AgentMoveResponse(
                move=played,
                dice=last.dice,
                state=state_view(session.state),
                to_act=color_of(session.to_act),
                needs_roll=session.needs_roll,
                terminal=session.terminal,
                outcome=outcome_view(session.outcome),
                win_prob=win_prob,
                can_undo=session.can_undo,
            )

    @app.post("/undo", response_model=UndoResponse)
    def undo(req: GameIdRequest) -> UndoResponse:
        session = get_session(req.game_id)
        with session.lock:
            try:
                session.undo()
            except GameError as exc:
                raise HTTPException(409, str(exc)) from None
            state, dice = session.state, session.dice
            moves = legal_move_views(state, dice, session.legal) if dice is not None else []
            return UndoResponse(
                state=state_view(state),
                to_act=color_of(session.to_act),
                dice=dice,
                needs_roll=session.needs_roll,
                terminal=session.terminal,
                outcome=outcome_view(session.outcome),
                moves=moves,
                can_undo=session.can_undo,
            )

    @app.post("/export_mat", response_model=ExportMatResponse)
    def export_mat(req: GameIdRequest) -> ExportMatResponse:
        session = get_session(req.game_id)
        with session.lock:
            # The env always opens with WHITE, so WHITE is .mat player 1; name the
            # seats from who sat where (human seat vs. the chosen opponent).
            if session.human_seat is Player.WHITE:
                white_name, black_name = "human", session.opponent_name
            else:
                white_name, black_name = session.opponent_name, "human"
            mat = game_to_mat(
                session.steps,
                session.outcome,
                white_name=white_name,
                black_name=black_name,
            )
        return ExportMatResponse(filename=f"game-{session.game_id}.mat", mat=mat)

    static_path = Path(static_dir) if static_dir is not None else DEFAULT_STATIC_DIR
    if static_path.is_dir():
        # Mounted last so the API routes above take precedence over static files.
        app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

    return app
