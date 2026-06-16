"""Legal-move + afterstate enumeration (depth-first).

``legal_moves`` returns ``(Move, afterstate)`` pairs de-duplicated by afterstate.
The DFS makes the hard rules fall out by construction:

* "use the maximum number of dice possible" -> keep only maximal-length branches;
* doubles -> up to four submoves (the die multiset has four entries);
* bar checkers must re-enter first -> :func:`_submoves_for_die` yields only bar
  entries while the bar is non-empty;
* bear-off exact vs. overshoot -> :func:`_can_bear_off_from`;
* the forced-higher-die rule (you must play the higher die when only one die is
  playable) -> a post-filter, since pure max-length keeps both single-die plays.

The canonical :class:`Move` kept per afterstate is the lexicographically smallest
*executable* submove sequence reaching it, so the web UI / gnubg export always
get a concrete legal ordering.
"""

from __future__ import annotations

from .apply import apply_submove
from .board import (
    NUM_POINTS,
    all_home,
    can_land,
    count_at,
    direction,
    entry_point,
)
from .types import BAR, OFF, Dice, EnvState, Move, Player, SubMove

_AfterKey = tuple[tuple[int, ...], tuple[int, int], tuple[int, int]]


def _has_farther_checker(board: tuple[int, ...], mover: Player, p: int) -> bool:
    """True if ``mover`` has a checker farther from bearing off than point ``p``."""
    if mover is Player.WHITE:
        return any(board[i] > 0 for i in range(p + 1, 6))
    return any(board[i] < 0 for i in range(18, p))


def _can_bear_off_from(state: EnvState, mover: Player, p: int, die: int) -> bool:
    """Whether ``mover`` may bear a checker off ``p`` with ``die`` (all home)."""
    pip = (p + 1) if mover is Player.WHITE else (NUM_POINTS - p)
    if die == pip:
        return True
    if die > pip:  # overshoot: only from the farthest occupied home point
        return not _has_farther_checker(state.board, mover, p)
    return False


def _submoves_for_die(state: EnvState, mover: Player, die: int) -> list[SubMove]:
    """All legal single submoves for ``mover`` using one die value (1..6)."""
    board = state.board
    if state.bar[mover] > 0:  # must re-enter from the bar before any other move
        ep = entry_point(mover, die)
        return [SubMove(BAR, ep)] if can_land(board, ep, mover) else []

    moves: list[SubMove] = []
    step = direction(mover)
    home_ok: bool | None = None
    for p in range(NUM_POINTS):
        if count_at(board, p, mover) == 0:
            continue
        dst = p + step * die
        if 0 <= dst < NUM_POINTS:
            if can_land(board, dst, mover):
                moves.append(SubMove(p, dst))
        else:  # past the home edge -> a bear-off (only if everyone is home)
            if home_ok is None:
                home_ok = all_home(state, mover)
            if home_ok and _can_bear_off_from(state, mover, p, die):
                moves.append(SubMove(p, OFF))
    return moves


def _seq_key(submoves: tuple[SubMove, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((sm.src, sm.dst) for sm in submoves)


def legal_moves(state: EnvState, dice: Dice) -> list[tuple[Move, EnvState]]:
    """Return ``(Move, afterstate)`` pairs for ``state.turn`` given ``dice``.

    Afterstates carry ``turn = opponent`` (it is their move next). The list is
    de-duplicated by afterstate; an empty list means no legal move (the turn
    passes).
    """
    mover = state.turn
    opp = mover.opponent()
    dice_seq = (dice[0],) * 4 if dice[0] == dice[1] else (dice[0], dice[1])

    best: dict[_AfterKey, tuple[int, tuple[SubMove, ...]]] = {}
    max_used = 0

    def visit(cur: EnvState, remaining: tuple[int, ...], path: tuple[SubMove, ...]) -> None:
        nonlocal max_used
        extended = False
        tried: set[int] = set()
        for i, die in enumerate(remaining):
            if die in tried:  # identical dice at this frontier give identical branches
                continue
            tried.add(die)
            for sm in _submoves_for_die(cur, mover, die):
                extended = True
                nxt = apply_submove(cur, mover, sm)
                visit(nxt, remaining[:i] + remaining[i + 1 :], (*path, sm))
        if not extended and path:  # leaf: nothing more can be played on this branch
            key = (cur.board, cur.bar, cur.off)
            n = len(path)
            max_used = max(max_used, n)
            prev = best.get(key)
            if prev is None or n > prev[0] or (n == prev[0] and _seq_key(path) < _seq_key(prev[1])):
                best[key] = (n, path)

    visit(state, dice_seq, ())

    # Forced-higher-die: if no two-dice play exists for a non-double, the higher
    # die must be played whenever it is playable at all.
    if max_used == 1 and dice[0] != dice[1]:
        higher_submoves = _submoves_for_die(state, mover, max(dice))
        if higher_submoves:
            allowed: set[_AfterKey] = set()
            for sm in higher_submoves:
                nxt = apply_submove(state, mover, sm)
                allowed.add((nxt.board, nxt.bar, nxt.off))
            best = {k: v for k, v in best.items() if k in allowed}

    out: list[tuple[Move, EnvState]] = []
    for (board, bar, off), (n, path) in best.items():
        if n != max_used:
            continue
        after = EnvState(
            board=board,
            bar=bar,
            off=off,
            turn=opp,
            cube_value=state.cube_value,
            cube_owner=state.cube_owner,
        )
        out.append((Move(path), after))
    return out
