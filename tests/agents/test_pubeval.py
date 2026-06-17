"""Tests for the pubeval benchmark opponent (:mod:`bgrl.agents.pubeval_agent`).

The pubeval scoring core (weights + ``setx``) is a verbatim port, so it is correct by
construction; the bespoke, error-prone parts are the ``EnvState -> pos[]`` mapping and
the race test. These are pinned by: the canonical opening ``pos[]`` (hand-derived),
perspective symmetry (a position from WHITE's POV must map identically to its
colour-flipped twin from BLACK's POV), and pubeval beating ``RandomAgent`` decisively
(a mapping/weight bug would tank that win-rate).
"""

from __future__ import annotations

import numpy as np

from bgrl.agents import Agent, PubevalAgent, RandomAgent
from bgrl.agents.pubeval_agent import _is_race, _pubeval_score, _to_pubeval_board
from bgrl.env import Env, EnvState, Player, RandomDiceSource
from bgrl.game import play_game
from bgrl.training.evaluate import play_match


def _color_flip(s: EnvState) -> EnvState:
    """Swap colours and mirror the board (point ``i`` <-> ``23-i``)."""
    return EnvState(
        board=tuple(-s.board[23 - i] for i in range(24)),
        bar=(s.bar[Player.BLACK], s.bar[Player.WHITE]),
        off=(s.off[Player.BLACK], s.off[Player.WHITE]),
        turn=s.turn.opponent(),
        cube_value=s.cube_value,
        cube_owner=s.cube_owner,
    )


def test_opening_board_matches_canonical_pubeval_layout() -> None:
    """The opening maps to pubeval's canonical opening, identically for either mover."""
    state = Env.initial_state()
    expected = [0] * 28
    # Mover (computer) POV: 2 on 24, 5 on 13, 3 on 8, 5 on 6; opponent: 2 on 1, 5 on 12,
    # 3 on 17, 5 on 19. (Tesauro's standard opening encoding.)
    expected[24], expected[13], expected[8], expected[6] = 2, 5, 3, 5
    expected[1], expected[12], expected[17], expected[19] = -2, -5, -3, -5

    assert _to_pubeval_board(state, Player.WHITE) == expected
    assert _to_pubeval_board(state, Player.BLACK) == expected  # opening is symmetric


def test_board_mapping_is_perspective_symmetric() -> None:
    """``board(s, WHITE) == board(color_flip(s), BLACK)`` over real game positions."""
    rng = np.random.default_rng(0)
    result = play_game(PubevalAgent(), RandomAgent(rng), RandomDiceSource(rng), record=True)
    sampled = [Env.initial_state(), *(step.afterstate for step in result.steps)]
    for s in sampled:
        assert _to_pubeval_board(s, Player.WHITE) == _to_pubeval_board(_color_flip(s), Player.BLACK)


def test_is_race_detects_contact() -> None:
    """Opening has contact; a passed-by position is a race; bar men force contact."""
    assert not _is_race(Env.initial_state())

    # WHITE all in its home (0..5), BLACK all in its home (18..23): out of contact.
    board = [0] * 24
    board[0], board[18] = 15, -15
    race_state = EnvState(board=tuple(board), bar=(0, 0), off=(0, 0), turn=Player.WHITE)
    assert _is_race(race_state)

    on_bar = EnvState(board=tuple(board), bar=(1, 0), off=(0, 0), turn=Player.WHITE)
    assert not _is_race(on_bar)


def test_all_men_off_scores_as_a_win() -> None:
    pos = [0] * 28
    pos[26] = 15  # mover has borne off everything
    assert _pubeval_score(False, pos) == _pubeval_score(True, pos) > 1e6


def test_is_an_agent_and_deterministic() -> None:
    agent = PubevalAgent()
    assert isinstance(agent, Agent)
    state = Env.initial_state()
    legal = Env.legal_moves(state, (3, 1))
    assert agent.act(state, (3, 1), legal) == agent.act(state, (3, 1), legal)


def test_plays_a_full_legal_game() -> None:
    rng = np.random.default_rng(1)
    result = play_game(PubevalAgent(), PubevalAgent(), RandomDiceSource(rng))
    assert result.outcome is not None


def test_beats_random_decisively() -> None:
    rng = np.random.default_rng(0)
    result = play_match(PubevalAgent(), RandomAgent(rng), pairs=50, rng=rng)
    assert result.win_rate_a > 0.75, f"pubeval should crush random, got {result.win_rate_a:.3f}"
