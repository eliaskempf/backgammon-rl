"""Money-game loop with the doubling cube (:mod:`bgrl.money`)."""

from __future__ import annotations

import numpy as np
import torch

from bgrl.agents import RandomAgent, ValueAgent
from bgrl.env import Move, Outcome, Player, RandomDiceSource, WinKind
from bgrl.money import _signed_points, play_money_game
from bgrl.nets.cube import CubeDecider
from bgrl.nets.value_net import MLPValueNet


class _CubeBot:
    """A CubeCapable agent with a fixed cube policy and greedy first-legal play."""

    def __init__(self, *, double_from_centered: bool, take: bool) -> None:
        self._double = double_from_centered
        self._take = take

    def act(self, state, dice, legal: list[tuple[Move, object]]) -> Move:
        return legal[0][0]

    def should_double(self, state, cube) -> bool:
        return self._double and cube.value == 1 and cube.owner is None

    def should_take(self, state, cube) -> bool:
        return self._take


def _doubler() -> _CubeBot:
    return _CubeBot(double_from_centered=True, take=True)


def _dropper() -> _CubeBot:
    return _CubeBot(double_from_centered=False, take=False)


def _taker() -> _CubeBot:
    return _CubeBot(double_from_centered=False, take=True)


def _dice(seed: int = 0) -> RandomDiceSource:
    return RandomDiceSource(np.random.default_rng(seed))


def test_drop_ends_game_at_pre_double_stake() -> None:
    # WHITE doubles on ply 0; BLACK drops -> WHITE collects the pre-double 1-point stake.
    res = play_money_game(_doubler(), _dropper(), _dice())
    assert res.dropped is True
    assert res.outcome is None
    assert res.final_cube_value == 1
    assert res.points == 1  # +1 from WHITE's POV
    assert len(res.cube_events) == 1
    ev = res.cube_events[0]
    assert (ev.doubler, ev.from_value, ev.to_value, ev.taken) == (Player.WHITE, 1, 2, False)


def test_black_drop_signs_points_negative() -> None:
    # WHITE never doubles; BLACK doubles on ply 1 and WHITE drops -> BLACK collects 1.
    res = play_money_game(_dropper(), _doubler(), _dice())
    assert res.dropped is True
    assert res.points == -1  # BLACK collects -> negative from WHITE's POV
    assert res.cube_events[0].doubler is Player.BLACK


def test_take_doubles_value_transfers_ownership_and_plays_out() -> None:
    # WHITE doubles once (from centered), BLACK takes -> cube=2 owned by BLACK; the game
    # then plays to completion with no further doubles.
    res = play_money_game(_doubler(), _taker(), _dice(7))
    assert res.dropped is False
    assert res.final_cube_value == 2
    assert len(res.cube_events) == 1 and res.cube_events[0].taken is True
    assert res.outcome is not None
    magnitude = int(res.outcome.kind)
    assert abs(res.points) == 2 * magnitude
    expected_sign = 1 if res.outcome.winner is Player.WHITE else -1
    assert res.points == expected_sign * 2 * magnitude


def test_netless_agents_never_double() -> None:
    # RandomAgents have no net and no fallback evaluator -> the cube never turns.
    res = play_money_game(
        RandomAgent(np.random.default_rng(1)), RandomAgent(np.random.default_rng(2)), _dice(3)
    )
    assert res.cube_events == ()
    assert res.final_cube_value == 1
    assert res.outcome is not None


def test_default_value_agents_play_to_completion() -> None:
    # Smoke test of the full B3+B4 path: real nets, the 1-ply evaluator, and the
    # CubeDecider must drive a complete game without error.
    torch.manual_seed(0)
    net = MLPValueNet(hidden=16)
    res = play_money_game(ValueAgent(net), ValueAgent(net), _dice(5), decider=CubeDecider())
    assert res.plies > 0
    assert res.dropped or res.outcome is not None
    assert res.final_cube_value >= 1


def test_jacoby_collapses_only_undoubled_magnitude() -> None:
    gammon_white = Outcome(Player.WHITE, WinKind.GAMMON)
    # Jacoby on + cube never turned: the gammon scores as a single.
    assert _signed_points(gammon_white, 1, jacoby=True, cube_turned=False) == 1
    # Jacoby on but the cube was turned: full magnitude counts (x the cube value).
    assert _signed_points(gammon_white, 2, jacoby=True, cube_turned=True) == 4
    # Jacoby off: full magnitude always.
    assert _signed_points(gammon_white, 1, jacoby=False, cube_turned=False) == 2
    # BLACK winner signs negative from WHITE's POV.
    bg_black = Outcome(Player.BLACK, WinKind.BACKGAMMON)
    assert _signed_points(bg_black, 1, jacoby=False, cube_turned=False) == -3
