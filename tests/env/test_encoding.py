"""Encoding shape, version, reserved slots, and Tesauro threshold spot-checks."""

import numpy as np

from bgrl.env import Env, EnvState, Player, encode
from bgrl.env.encoding import ENCODING_VERSION, N_FEATURES

WHITE = Player.WHITE


def test_shape_dtype_version():
    s = Env.initial_state()
    f = encode(s, s.turn)
    assert f.shape == (N_FEATURES,)
    assert f.dtype == np.float32
    assert N_FEATURES == 200
    assert ENCODING_VERSION == 1
    assert np.all(f >= 0.0)
    assert np.all(np.isfinite(f))


def test_reserved_cube_slots_are_zero():
    s = Env.initial_state()
    f = encode(s, s.turn)
    assert f[198] == 0.0
    assert f[199] == 0.0


def test_side_to_move_flag():
    s = Env.initial_state()  # WHITE to move
    f = encode(s, WHITE)  # perspective == side to move
    assert f[196] == 1.0
    assert f[197] == 0.0
    g = encode(s, WHITE.opponent())  # perspective != side to move
    assert g[196] == 0.0
    assert g[197] == 1.0


def test_tesauro_thresholds():
    board = [0] * 24
    board[0] = 1  # 1 checker
    board[1] = 2  # 2 checkers
    board[2] = 3  # 3 checkers
    board[3] = 5  # 5 checkers -> 4th unit (5-3)/2 == 1.0
    board[18] = -11  # park BLACK (not asserted here)
    s = EnvState(board=tuple(board), bar=(0, 0), off=(4, 0), turn=WHITE)
    f = encode(s, WHITE)  # WHITE perspective: rel index == absolute index
    assert np.array_equal(f[0:4], [1, 0, 0, 0])
    assert np.array_equal(f[4:8], [1, 1, 0, 0])
    assert np.array_equal(f[8:12], [1, 1, 1, 0])
    assert np.array_equal(f[12:16], [1, 1, 1, 1])
