"""The reference game driver — plays one game between two agents.

This is the algorithm-agnostic glue that ties the env, the dice source, and the
agent lifecycle together. WP1's training loop is built *around* this (it does not
replace it): passing the **same** learning agent as both seats is exactly
TD-Gammon self-play, and the trainer just calls :func:`play_game` repeatedly.
Living at the package root keeps ``bgrl/training/`` free for WP1.

Per-ply protocol: roll the dice, ask the mover for a move over the precomputed
legal afterstates (an empty legal set is a forced pass — an empty
:class:`~bgrl.env.Move`), apply it, and notify a :class:`LearningAgent` via
``observe_step``. At termination each distinct learning agent gets one
``observe_game_end`` with the absolute outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from bgrl.agents.base import Agent, LearningAgent
from bgrl.env import Dice, DiceSource, Env, EnvState, Move, Outcome, Player

_PASS = Move(submoves=())


@dataclass(frozen=True, slots=True)
class Step:
    """One applied ply, ``state --(dice, move)--> afterstate`` (afterstate.turn = opponent)."""

    state: EnvState
    dice: Dice
    move: Move
    afterstate: EnvState


@dataclass(frozen=True, slots=True)
class GameResult:
    """Outcome of a finished game plus its length; ``steps`` is populated iff recorded."""

    outcome: Outcome | None  # None only if max_plies was hit before terminal
    plies: int
    steps: tuple[Step, ...] = ()


def _afterstate_for(legal: list[tuple[Move, EnvState]], move: Move) -> EnvState:
    for m, after in legal:
        if m == move:
            return after
    raise ValueError("agent returned a move that is not in the legal set")


def play_game(
    white: Agent,
    black: Agent,
    dice: DiceSource,
    *,
    max_plies: int = 10_000,
    record: bool = False,
) -> GameResult:
    """Play one game to termination (or ``max_plies``) and return the result.

    ``dice`` supplies every roll, so the game is fully reproducible given the
    agents and the dice source (use :class:`~bgrl.env.ReplayDiceSource` for CRN).
    ``record=True`` returns the full :class:`Step` trajectory (off by default to
    avoid retaining every position over millions of self-play games).
    """
    agents = {Player.WHITE: white, Player.BLACK: black}
    state = Env.initial_state()
    steps: list[Step] = []
    plies = 0

    while not Env.is_terminal(state) and plies < max_plies:
        mover = state.turn
        agent = agents[mover]
        roll = dice.roll()
        legal = Env.legal_moves(state, roll)
        if legal:
            move = agent.act(state, roll, legal)
            afterstate = _afterstate_for(legal, move)
        else:
            move = _PASS
            afterstate = replace(state, turn=mover.opponent())

        if isinstance(agent, LearningAgent):
            agent.observe_step(state, roll, move, afterstate)
        if record:
            steps.append(Step(state, roll, move, afterstate))

        state = afterstate
        plies += 1

    result = Env.outcome(state)
    if result is not None:
        for agent in _distinct_learners(white, black):
            agent.observe_game_end(result)

    return GameResult(outcome=result, plies=plies, steps=tuple(steps))


def _distinct_learners(white: Agent, black: Agent) -> list[LearningAgent]:
    """Each learning agent once, by identity (self-play shares one object)."""
    learners: list[LearningAgent] = []
    for agent in (white, black):
        if isinstance(agent, LearningAgent) and not any(agent is seen for seen in learners):
            learners.append(agent)
    return learners
