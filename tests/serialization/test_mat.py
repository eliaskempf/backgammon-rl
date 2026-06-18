"""Pure (no-gnubg) tests for the Jellyfish ``.mat`` writer.

These pin the structure and notation our writer produces; the live round-trip *through*
gnubg lives in ``test_gnubg.py`` (skipped when gnubg is absent).
"""

import re

import numpy as np

from bgrl.agents import RandomAgent
from bgrl.env import BAR, OFF, EnvState, Move, Player, RandomDiceSource, SubMove, apply_submove
from bgrl.game import Step, play_game
from bgrl.serialization.mat import game_to_mat, match_to_mat, point_number


def _random_game(seed):
    """A deterministic recorded self-play game (RandomAgent vs RandomAgent)."""
    dice = RandomDiceSource(np.random.default_rng(seed))
    white = RandomAgent(np.random.default_rng(1000 + seed))
    black = RandomAgent(np.random.default_rng(2000 + seed))
    res = play_game(white, black, dice, record=True)
    return res.steps, res.outcome


def _step(turn, board_counts, dice, submoves, *, bar=(0, 0), off=(0, 0)):
    """Build one hand-crafted Step (the writer only reads state/dice/move)."""
    board = [0] * 24
    for i, v in board_counts.items():
        board[i] = v
    state = EnvState(board=tuple(board), bar=bar, off=off, turn=turn)
    move = Move(submoves=tuple(SubMove(s, d) for s, d in submoves))
    after = state
    for sm in move.submoves:
        after = apply_submove(after, turn, sm)
    return Step(state, dice, move, after)


# --- point numbering (Jellyfish: each side counts 1..24 from its own home) ----------


def test_point_number_is_per_player():
    assert point_number(0, Player.WHITE) == 1
    assert point_number(23, Player.WHITE) == 24
    assert point_number(23, Player.BLACK) == 1
    assert point_number(0, Player.BLACK) == 24
    # absolute index 5 (WHITE 6-point) is BLACK's 19-point
    assert point_number(5, Player.WHITE) == 6
    assert point_number(5, Player.BLACK) == 19


# --- notation tokens -----------------------------------------------------------------


def test_hit_is_marked_with_star():
    # WHITE on the 8-point (idx 7) lands on a lone BLACK blot on the 5-point (idx 4).
    step = _step(Player.WHITE, {7: 1, 4: -1}, (3, 5), [(7, 4)])
    mat = game_to_mat([step], None)
    assert "8/5*" in mat
    assert "  1) 53:" in mat  # dice rendered higher-first


def test_no_star_when_destination_is_empty_or_owned():
    step = _step(Player.WHITE, {7: 1, 4: 2}, (3, 5), [(7, 4)])  # own checkers on dst
    assert "8/5*" not in game_to_mat([step], None)
    assert "8/5" in game_to_mat([step], None)


def test_bar_entry_token():
    # WHITE enters from the bar with a 5 -> absolute 24-5=19 -> WHITE point 20.
    step = _step(Player.WHITE, {}, (5, 2), [(BAR, 19)], bar=(1, 0))
    assert "bar/20" in game_to_mat([step], None)


def test_bear_off_token():
    step = _step(Player.WHITE, {2: 1}, (3, 1), [(2, OFF)], off=(14, 0))
    assert "3/off" in game_to_mat([step], None)


def test_black_moves_use_black_numbering_in_the_right_column():
    white = _step(Player.WHITE, {7: 1}, (1, 2), [(7, 6)])  # WHITE 8/7-ish, left column
    black = _step(Player.BLACK, {16: -1}, (2, 1), [(16, 18)])  # BLACK 8/6, right column
    mat = game_to_mat([white, black], None)
    line = next(line for line in mat.splitlines() if line.lstrip().startswith("1)"))
    assert "8/6" in line.split("  ")[-1]  # BLACK's token sits in the right cell


def test_doubles_expand_to_four_submoves():
    # idx 7->point 8, idx 5->point 6, idx 3->point 4 (WHITE numbering).
    step = _step(Player.WHITE, {7: 2, 5: 2}, (2, 2), [(7, 5), (7, 5), (5, 3), (5, 3)])
    mat = game_to_mat([step], None)
    assert "8/6 8/6 6/4 6/4" in mat


# --- whole-match structure -----------------------------------------------------------


def test_match_header_and_player_names():
    steps, outcome = _random_game(0)
    mat = game_to_mat(steps, outcome, white_name="alice", black_name="bob")
    assert mat.startswith(" 0 point match\n")  # leading space matches gnubg's export
    assert "\n Game 1\n" in mat
    assert "alice : 0" in mat and "bob : 0" in mat  # gnubg's "<name> : <score>" form
    assert mat.endswith("\n")


def test_player_name_colon_is_sanitised_for_gnubg():
    # A colon in a name (e.g. an "llm:model" agent label) gives gnubg's importer a second
    # ':' on the names line and crashes its analysis; the writer must neutralise it.
    # Slashes are fine, so a checkpoint-path / model-slug name is preserved.
    steps, outcome = _random_game(0)
    mat = game_to_mat(steps, outcome, white_name="llm:dry", black_name="anthropic/claude")
    names_line = next(ln for ln in mat.splitlines() if "claude" in ln)
    assert "llm:dry" not in names_line
    assert "llm-dry : 0" in names_line
    assert "anthropic/claude : 0" in names_line


def test_win_line_reflects_outcome_magnitude():
    steps, outcome = _random_game(0)  # BLACK wins a single
    mat = game_to_mat(steps, outcome)
    assert outcome is not None
    pts = int(outcome.kind)
    assert f"Wins {pts} point{'s' if pts != 1 else ''}" in mat


def test_unfinished_game_omits_win_line():
    steps, _ = _random_game(0)
    assert "Wins" not in game_to_mat(steps[:6], None)


def test_columns_never_merge_and_each_cell_is_one_play():
    # Regression: a wide left cell must keep a whitespace gap from the right cell, and
    # every cell holds exactly one dice token (one ':'), never two plays run together.
    steps, outcome = _random_game(0)
    mat = game_to_mat(steps, outcome)
    move_lines = [ln for ln in mat.splitlines() if re.match(r"^\s*\d+\) ", ln)]
    assert len(move_lines) == (len(steps) + 1) // 2
    for ln in move_lines:
        content = re.match(r"^\s*\d+\) (.*)$", ln).group(1)
        cells = re.split(r"\s{2,}", content.rstrip())
        assert 1 <= len(cells) <= 2
        for cell in cells:
            assert cell.count(":") == 1, f"merged/odd cell {cell!r} in {ln!r}"


def test_first_move_dice_is_higher_first():
    steps, outcome = _random_game(0)
    mat = game_to_mat(steps, outcome)
    first = next(ln for ln in mat.splitlines() if re.match(r"^\s*1\) ", ln))
    hi, lo = re.search(r"\b(\d)(\d):", first).groups()
    assert int(hi) >= int(lo)


def test_match_to_mat_emits_multiple_game_blocks():
    g0 = _random_game(0)
    g1 = _random_game(1)
    mat = match_to_mat([g0, g1])
    assert "\n Game 1\n" in mat
    assert "\n Game 2\n" in mat
