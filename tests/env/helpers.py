"""Shared test helpers: reachable-state generation, the oracle bridge, and mirror."""

import random

from bgrl._vendor import gym_backgammon_ref as ref
from bgrl.env import Env, EnvState, Player, legal_moves

WHITE, BLACK = Player.WHITE, Player.BLACK


class OracleUnsupported(Exception):
    """The vendored reference hit one of its internal asserts on this position."""


def reachable_states(seed, n, max_plies=300):
    """Sample ``n`` ``(state, dice)`` pairs along random legal rollouts from start.

    Only reachable (hence legal) positions are produced — never fabricated boards,
    which could be unreachable/illegal.
    """
    rng = random.Random(seed)
    out = []
    s = Env.initial_state()
    plies = 0
    while len(out) < n:
        if Env.is_terminal(s) or plies >= max_plies:
            s = Env.initial_state()
            plies = 0
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        out.append((s, dice))
        moves = legal_moves(s, dice)
        if not moves:
            s = EnvState(board=s.board, bar=s.bar, off=s.off, turn=s.turn.opponent())
        else:
            s = moves[rng.randrange(len(moves))][1]
        plies += 1
    return out


def to_ref(state):
    """Build a vendored ``Backgammon`` instance holding ``state``'s position."""
    g = ref.Backgammon()
    board = []
    for v in state.board:
        if v > 0:
            board.append((v, ref.WHITE))
        elif v < 0:
            board.append((-v, ref.BLACK))
        else:
            board.append((0, None))
    g.board = board
    g.bar = [state.bar[0], state.bar[1]]
    g.off = [state.off[0], state.off[1]]
    g.players_positions = g.get_players_positions()
    return g


def oracle_after_keys(state, dice):
    """Afterstate keys ``(board, bar, off)`` the reference reaches for ``state``.

    Raises :class:`OracleUnsupported` if the reference trips an internal assert.
    Note WHITE is driven with negated rolls (the reference's convention).
    """
    player = ref.WHITE if state.turn is WHITE else ref.BLACK
    roll = (-dice[0], -dice[1]) if state.turn is WHITE else (dice[0], dice[1])
    try:
        plays = to_ref(state).get_valid_plays(player, roll)
    except Exception as e:  # the reference asserts on rare states
        raise OracleUnsupported from e
    keys = set()
    for play in plays:
        g = to_ref(state)
        try:
            g.execute_play(player, play)
        except Exception as e:  # the reference asserts on rare states
            raise OracleUnsupported from e
        board = tuple(
            (c if col == ref.WHITE else (-c if col == ref.BLACK else 0)) for (c, col) in g.board
        )
        keys.add((board, (g.bar[0], g.bar[1]), (g.off[0], g.off[1])))
    return keys


def mirror(state):
    """Colour-and-axis mirror: the identical game viewed from the other side."""
    board = tuple(-state.board[23 - i] for i in range(24))
    return EnvState(
        board=board,
        bar=(state.bar[1], state.bar[0]),
        off=(state.off[1], state.off[0]),
        turn=state.turn.opponent(),
    )
