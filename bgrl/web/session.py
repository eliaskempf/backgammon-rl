"""Server-side game state — the legality boundary for the web API.

A :class:`GameSession` holds one in-progress game (state, the live roll and its
cached legal moves, the recorded trajectory, the opponent agent) and is the *only*
place moves are validated and applied. It mirrors :func:`bgrl.game.play_game`'s
per-ply protocol (roll → enumerate legal afterstates → apply → flip), so the env
stays the sole authority on legality; the HTTP layer never reasons about it.

Unlike self-play, recording is always on here — the recorded ``Step`` trajectory is
what WP3 Part B serialises to ``.mat``.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import replace

import numpy as np

from bgrl.agents.base import Agent
from bgrl.env import (
    Dice,
    DiceSource,
    Env,
    EnvState,
    ManualDiceSource,
    Move,
    Outcome,
    Player,
    SubMove,
)
from bgrl.game import Step

PASS = Move(submoves=())  # the empty move applied on a forced pass (matches game.py)


class GameError(Exception):
    """The request conflicts with the current game state (maps to HTTP 409)."""


class IllegalMove(GameError):
    """The requested move is not in the legal set for the current roll."""


class GameSession:
    """One in-progress browser game: human in ``human_seat`` vs ``opponent``."""

    def __init__(
        self,
        game_id: str,
        *,
        opponent: Agent,
        opponent_name: str,
        human_seat: Player,
        dice_source: DiceSource,
    ) -> None:
        self.game_id = game_id
        self.opponent = opponent
        self.opponent_name = opponent_name
        self.human_seat = human_seat
        self.dice_source = dice_source
        self.state: EnvState = Env.initial_state()
        self.dice: Dice | None = None
        self.legal: list[tuple[Move, EnvState]] = []
        self.steps: list[Step] = []
        self.outcome: Outcome | None = None
        self.last_used: float = time.monotonic()
        # Sync handlers run in a threadpool; serialise mutations on one game.
        self.lock = threading.RLock()

    # --- queries ---------------------------------------------------------------

    @property
    def terminal(self) -> bool:
        return self.outcome is not None

    @property
    def to_act(self) -> Player:
        return self.state.turn

    @property
    def needs_roll(self) -> bool:
        return not self.terminal and self.dice is None

    @property
    def human_to_move(self) -> bool:
        return not self.terminal and self.state.turn is self.human_seat

    @property
    def is_manual(self) -> bool:
        """True when the human supplies every roll (both seats) via :meth:`supply_dice`."""
        return isinstance(self.dice_source, ManualDiceSource)

    @property
    def can_undo(self) -> bool:
        """True iff there is a prior human decision (a non-pass human step) to revert to."""
        return any(s.state.turn is self.human_seat and s.move.submoves for s in self.steps)

    # --- mutations -------------------------------------------------------------

    def supply_dice(self, dice: Dice) -> None:
        """Queue a human-entered roll for the next :meth:`roll`; manual games only."""
        if not isinstance(self.dice_source, ManualDiceSource):
            raise GameError("game is not in manual-dice mode")
        self.dice_source.push(dice)

    def roll(self) -> Dice:
        """Roll for the current mover and cache the resulting legal moves."""
        if self.terminal:
            raise GameError("game is over")
        if self.dice is not None:
            raise GameError("dice already rolled; play the move first")
        self.dice = self.dice_source.roll()
        self.legal = Env.legal_moves(self.state, self.dice)
        self.last_used = time.monotonic()
        return self.dice

    def move_for_id(self, move_id: int) -> Move:
        if not 0 <= move_id < len(self.legal):
            raise IllegalMove(f"move id {move_id} out of range ({len(self.legal)} legal moves)")
        return self.legal[move_id][0]

    def move_for_submoves(self, submoves: list[SubMove]) -> Move:
        target = Move(submoves=tuple(submoves))
        for move, _ in self.legal:
            if move == target:
                return move
        raise IllegalMove("submoves do not form a legal move for the current roll")

    def apply_move(self, move: Move) -> None:
        """Apply a chosen legal move, recording the step and advancing the game."""
        if self.terminal:
            raise GameError("game is over")
        if self.dice is None:
            raise GameError("must roll before moving")
        for candidate, afterstate in self.legal:
            if candidate == move:
                self._advance(move, afterstate)
                return
        raise IllegalMove("move is not in the legal set for the current roll")

    def apply_pass(self) -> None:
        """Apply a forced pass (the current roll yields no legal move)."""
        if self.dice is None:
            raise GameError("must roll before passing")
        if self.legal:
            raise GameError("there are legal moves; a pass is not allowed")
        self._advance(PASS, replace(self.state, turn=self.state.turn.opponent()))

    def play_agent(self) -> Move | None:
        """Roll if needed and play the opponent's move; ``None`` on a forced pass."""
        if self.terminal:
            raise GameError("game is over")
        if self.dice is None:
            self.roll()
        if not self.legal:
            self.apply_pass()
            return None
        move = self.opponent.act(self.state, self.dice, self.legal)
        self.apply_move(move)
        return move

    def undo(self) -> None:
        """Revert to the human's most recent real decision so they can replay it.

        Scans the recorded steps backward for the latest one where it was the
        human's turn *and* they had a legal move (a forced human pass has nothing to
        redo, so it is skipped). Restores that step's pre-move state and the very
        roll it was played with — the human lands back at "you rolled X, choose your
        move" with the same dice — drops every later step (and any uncommitted
        current roll, which was never recorded), and clears any outcome.

        Raises :class:`GameError` (HTTP 409) when there is no such decision.
        """
        target: int | None = None
        for i in range(len(self.steps) - 1, -1, -1):
            step = self.steps[i]
            if step.state.turn is self.human_seat and step.move.submoves:
                target = i
                break
        if target is None:
            raise GameError("nothing to undo")
        step = self.steps[target]
        self.state = step.state
        self.dice = step.dice
        self.legal = Env.legal_moves(self.state, self.dice)
        self.outcome = Env.outcome(self.state) if Env.is_terminal(self.state) else None
        self.steps = self.steps[:target]
        self.last_used = time.monotonic()

    def _advance(self, move: Move, afterstate: EnvState) -> None:
        assert self.dice is not None  # callers guarantee a live roll
        self.steps.append(Step(self.state, self.dice, move, afterstate))
        self.state = afterstate
        self.dice = None
        self.legal = []
        self.last_used = time.monotonic()
        if Env.is_terminal(self.state):
            self.outcome = Env.outcome(self.state)


class SessionStore:
    """In-memory registry of live games keyed by an opaque ``game_id``.

    Bounded by a TTL and a max count (oldest evicted first); no persistence — a
    server restart drops in-flight games, which is fine for an illustrative tool.
    """

    def __init__(self, *, max_sessions: int = 256, ttl_seconds: float = 3600.0) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()
        self._max = max_sessions
        self._ttl = ttl_seconds

    def create(
        self,
        *,
        opponent: Agent,
        opponent_name: str,
        human_seat: Player,
        dice_source: DiceSource,
    ) -> GameSession:
        game_id = secrets.token_urlsafe(12)
        session = GameSession(
            game_id,
            opponent=opponent,
            opponent_name=opponent_name,
            human_seat=human_seat,
            dice_source=dice_source,
        )
        with self._lock:
            self._reap()
            self._sessions[game_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
            if session is None:
                raise KeyError(game_id)
            session.last_used = time.monotonic()
            return session

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _reap(self) -> None:
        now = time.monotonic()
        for gid in [g for g, s in self._sessions.items() if now - s.last_used > self._ttl]:
            del self._sessions[gid]
        if len(self._sessions) >= self._max:
            by_age = sorted(self._sessions.items(), key=lambda kv: kv[1].last_used)
            for gid, _ in by_age[: len(self._sessions) - self._max + 1]:
                del self._sessions[gid]


def make_seed_streams(seed: int | None) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent (dice, agent) RNG streams from one seed (``None`` = OS entropy)."""
    dice_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(dice_seed), np.random.default_rng(agent_seed)
