"""Unit tests for the pure env -> API view projections."""

from bgrl.env import BAR, OFF, Env, EnvState, Move, Outcome, Player, SubMove, WinKind
from bgrl.web.views import (
    color_of,
    legal_move_views,
    move_notation,
    move_view,
    player_of,
    point_number,
    state_view,
)

WHITE = Player.WHITE
BLACK = Player.BLACK


def test_color_player_roundtrip():
    assert color_of(WHITE) == "white"
    assert color_of(BLACK) == "black"
    assert player_of("white") is WHITE
    assert player_of("black") is BLACK


def test_point_number_endpoints_and_symmetry():
    # WHITE bears off past index 0: index 0 is the ace (1) point, index 23 is the 24.
    assert point_number(0, WHITE) == 1
    assert point_number(23, WHITE) == 24
    # BLACK bears off past index 23: mirror image.
    assert point_number(23, BLACK) == 1
    assert point_number(0, BLACK) == 24
    # A point and its mirror share the same number across players.
    for p in range(24):
        assert point_number(p, WHITE) == point_number(23 - p, BLACK)


def test_state_view_initial_position():
    sv = state_view(Env.initial_state())
    assert len(sv.board) == 24
    assert sv.turn == "white"
    assert sv.bar.white == 0 and sv.bar.black == 0
    assert sv.off.white == 0 and sv.off.black == 0
    assert sv.cube.value == 1 and sv.cube.owner is None
    # 15 checkers per side on the board at the start (none on bar/off).
    assert sum(c for c in sv.board if c > 0) == 15
    assert sum(-c for c in sv.board if c < 0) == 15


def test_state_view_projects_bar_off_and_cube():
    board = [0] * 24
    board[23] = -1  # one black checker on absolute point 23
    state = EnvState(
        board=tuple(board),
        bar=(2, 0),
        off=(13, 14),
        turn=BLACK,
        cube_value=2,
        cube_owner=WHITE,
    )
    sv = state_view(state)
    assert sv.bar.white == 2 and sv.bar.black == 0
    assert sv.off.white == 13 and sv.off.black == 14
    assert sv.turn == "black"
    assert sv.cube.value == 2 and sv.cube.owner == "white"
    assert sv.board[23] == -1


def test_move_notation_bar_entry_and_bear_off():
    # WHITE bearing a checker off from absolute index 3 (its 4 point).
    white_move = Move(submoves=(SubMove(3, OFF),))
    assert move_notation(white_move, WHITE) == f"{point_number(3, WHITE)}/off"

    # BLACK entering from the bar and moving another checker.
    black_move = Move(submoves=(SubMove(BAR, 20), SubMove(5, 2)))
    expected = f"bar/{point_number(20, BLACK)} {point_number(5, BLACK)}/{point_number(2, BLACK)}"
    assert move_notation(black_move, BLACK) == expected


def test_move_notation_forced_pass():
    assert move_notation(Move(submoves=()), WHITE) == "(no move)"


def test_move_view_carries_id_submoves_and_afterstate():
    state = Env.initial_state()
    dice = (3, 1)
    legal = Env.legal_moves(state, dice)
    move, afterstate = legal[0]
    mv = move_view(7, move, afterstate, state, dice)
    assert mv.id == 7
    assert len(mv.submoves) == len(move.submoves)
    assert mv.afterstate.turn == "black"  # afterstate hands the turn to the opponent
    assert isinstance(mv.notation, str) and mv.notation
    # Every submove is labelled with the die it consumes (here a 3-1, one each).
    assert sorted(sm.die for sm in mv.submoves) == sorted(dice[: len(mv.submoves)])
    # With no orderings passed, only the canonical ordering is exposed.
    assert [[(s.src, s.dst) for s in o] for o in mv.orderings] == [
        [(s.src, s.dst) for s in mv.submoves]
    ]


def test_legal_move_views_expose_every_ordering():
    # Bear-off position (cf. bug-examples/3.png): the play that bears off the 6-point and
    # plays 4/1 is reachable in both orders, so its MoveView must carry both orderings, one
    # of which starts by bearing off (SubMove(5, OFF)).
    board = [0] * 24
    board[3], board[4], board[5] = 1, 12, 1
    board[23] = -15
    state = EnvState(board=tuple(board), bar=(0, 0), off=(1, 0), turn=WHITE)
    dice = (6, 3)
    views = legal_move_views(state, dice, Env.legal_moves(state, dice))
    # ``id`` indexes the legal list so submission by id still works.
    assert [v.id for v in views] == list(range(len(views)))
    bear_first = [v for v in views if any(o[0].src == 5 and o[0].dst == OFF for o in v.orderings)]
    assert bear_first, "no ordering lets the 6-point checker bear off first"
    multi = [v for v in views if len(v.orderings) > 1]
    assert multi and all(
        len({tuple((s.src, s.dst) for s in o) for o in v.orderings}) == len(v.orderings)
        for v in multi
    )


def test_outcome_view():
    from bgrl.web.views import outcome_view

    assert outcome_view(None) is None
    ov = outcome_view(Outcome(winner=WHITE, kind=WinKind.GAMMON))
    assert ov is not None
    assert ov.winner == "white" and ov.kind == "gammon"
