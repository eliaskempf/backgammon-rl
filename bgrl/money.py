"""Money-game play loop with the doubling cube (WP6 B4).

A sibling to :func:`bgrl.game.play_game` — **not** a replacement. Training self-play stays
cubeless and keeps using ``play_game`` (B0: a passed double has no played-out gammon
outcome, which would corrupt the multi-head labels). This loop is for *evaluation and
play*: before each roll the on-roll player may double, the opponent takes or drops, stakes
scale with the cube, and a drop ends the game at the pre-double stake.

The cube lives as a loop-local :class:`~bgrl.nets.equity.CubeContext` and is applied
analytically — it is never fed into the net's encoding (the net stays cubeless, B0). Cube
decisions go through :func:`~bgrl.agents.cube_policy.wants_to_double` /
:func:`~bgrl.agents.cube_policy.wants_to_take`, so an agent's own policy
(:class:`~bgrl.agents.base.CubeCapable`) is honoured and everything else falls back to a
shared :class:`~bgrl.nets.cube.CubeDecider`. The recorded :class:`~bgrl.game.Step` trail
plus the :class:`CubeEvent` list is what the ``.mat`` cube export consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from bgrl.agents.base import Agent
from bgrl.agents.cube_policy import evaluator_for, wants_to_double, wants_to_take
from bgrl.env import DiceSource, Env, Move, Outcome, Player
from bgrl.game import Step
from bgrl.nets.base import ValueNet
from bgrl.nets.cube import CubeDecider
from bgrl.nets.equity import CENTERED_CUBE, CubeContext

_PASS = Move(submoves=())


@dataclass(frozen=True, slots=True)
class CubeEvent:
    """One cube action, taken just before ply ``ply`` by ``doubler``.

    ``taken`` distinguishes an accepted double (the game continues at ``to_value``) from a
    drop (the game ends; ``doubler`` collects ``from_value``).
    """

    ply: int
    doubler: Player
    from_value: int
    to_value: int
    taken: bool


@dataclass(frozen=True, slots=True)
class MoneyGameResult:
    """Outcome of a money game, including the cube history and the signed stake.

    ``outcome`` is ``None`` on a drop (no board terminal) and when ``max_plies`` is hit.
    ``points`` is the signed stake from **WHITE**'s point of view (``+`` = WHITE collects):
    on a played-out game ``win_magnitude * final_cube_value`` for the winner, on a drop the
    pre-double stake for the doubler. ``steps`` is populated iff ``record``.
    """

    outcome: Outcome | None
    dropped: bool
    final_cube_value: int
    points: int
    plies: int
    steps: tuple[Step, ...] = ()
    cube_events: tuple[CubeEvent, ...] = ()


def _may_double(cube: CubeContext, mover: Player) -> bool:
    """Legality gate: the on-roll player may double a centered or self-owned cube."""
    return cube.owner is None or cube.owner is mover


def _signed_points(result: Outcome, cube_value: int, *, jacoby: bool, cube_turned: bool) -> int:
    """Signed (WHITE-POV) stake of a played-out game: magnitude x cube value."""
    kind = int(result.kind)
    if jacoby and not cube_turned:
        kind = 1  # Jacoby rule: undoubled gammons/backgammons score as single games
    points = kind * cube_value
    return points if result.winner is Player.WHITE else -points


def play_money_game(
    white: Agent,
    black: Agent,
    dice: DiceSource,
    *,
    decider: CubeDecider | None = None,
    fallback_net: ValueNet | None = None,
    max_plies: int = 10_000,
    record: bool = True,
    jacoby: bool = False,
) -> MoneyGameResult:
    """Play one money game with the doubling cube; return the result and cube history.

    ``decider`` (default :class:`~bgrl.nets.cube.CubeDecider`) supplies the fallback cube
    policy; ``fallback_net`` gives net-less agents (random, LLM) a reference evaluator for
    that policy (else they never double / always take). ``jacoby`` toggles the Jacoby rule
    (default off, matching the cubeless training labels). No learning hooks fire — training
    must not run through this loop (B0).
    """
    decider = decider or CubeDecider()
    agents = {Player.WHITE: white, Player.BLACK: black}
    evaluators = {
        Player.WHITE: evaluator_for(white, fallback_net=fallback_net),
        Player.BLACK: evaluator_for(black, fallback_net=fallback_net),
    }
    state = Env.initial_state()
    cube = CENTERED_CUBE
    steps: list[Step] = []
    cube_events: list[CubeEvent] = []
    plies = 0

    while not Env.is_terminal(state) and plies < max_plies:
        mover = state.turn
        opp = mover.opponent()

        # --- pre-roll cube decision (only when the mover may legally double) ---
        if _may_double(cube, mover) and wants_to_double(
            agents[mover], state, cube, evaluator=evaluators[mover], decider=decider
        ):
            old_value = cube.value
            takes = wants_to_take(
                agents[opp], state, cube, evaluator=evaluators[opp], decider=decider
            )
            cube_events.append(CubeEvent(plies, mover, old_value, old_value * 2, taken=takes))
            if takes:
                cube = CubeContext(value=old_value * 2, owner=opp)
            else:  # dropped: the game ends, the doubler collects the pre-double stake
                points = old_value if mover is Player.WHITE else -old_value
                return MoneyGameResult(
                    outcome=None,
                    dropped=True,
                    final_cube_value=old_value,
                    points=points,
                    plies=plies,
                    steps=tuple(steps),
                    cube_events=tuple(cube_events),
                )

        # --- normal ply (mirrors bgrl.game.play_game's per-ply protocol) ---
        roll = dice.roll()
        legal = Env.legal_moves(state, roll)
        if legal:
            move = agents[mover].act(state, roll, legal)
            afterstate = next(a for m, a in legal if m == move)
        else:
            move = _PASS
            afterstate = replace(state, turn=opp)
        if record:
            steps.append(Step(state, roll, move, afterstate))
        state = afterstate
        plies += 1

    result = Env.outcome(state)
    points = (
        _signed_points(result, cube.value, jacoby=jacoby, cube_turned=bool(cube_events))
        if result is not None
        else 0
    )
    return MoneyGameResult(
        outcome=result,
        dropped=False,
        final_cube_value=cube.value,
        points=points,
        plies=plies,
        steps=tuple(steps),
        cube_events=tuple(cube_events),
    )
