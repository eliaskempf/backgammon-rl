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


def move_dice(state: EnvState, dice: Dice, move: Move) -> tuple[int, ...]:
    """Die value each submove of ``move`` consumes, in submove order.

    ``move`` must be one of the plays :func:`legal_moves` returns for this exact
    ``(state, dice)``; the result has the same length as ``move.submoves`` (an
    empty move yields ``()``). Doubles map every entry to the doubled value.

    A bear-off reachable by either die via overshoot (:func:`_can_bear_off_from`)
    is genuinely ambiguous, so the dice are assigned by backtracking — at each
    submove try every still-available die that can produce it, recurse, accept the
    first labelling that consumes a die for every submove. A consistent labelling
    always exists for a legal play; otherwise :class:`ValueError` is raised.
    """
    mover = state.turn
    multiset = (dice[0],) * 4 if dice[0] == dice[1] else (dice[0], dice[1])

    def assign(cur: EnvState, i: int, avail: tuple[int, ...]) -> tuple[int, ...] | None:
        if i == len(move.submoves):
            return ()
        sm = move.submoves[i]
        tried: set[int] = set()
        for j, die in enumerate(avail):
            if die in tried:  # identical dice give identical branches
                continue
            tried.add(die)
            if sm in _submoves_for_die(cur, mover, die):
                rest = assign(apply_submove(cur, mover, sm), i + 1, avail[:j] + avail[j + 1 :])
                if rest is not None:
                    return (die, *rest)
        return None

    labels = assign(state, 0, multiset)
    if labels is None:
        raise ValueError("move is not a legal play for this (state, dice)")
    return labels


def _enumerate(state: EnvState, dice: Dice) -> dict[_AfterKey, list[tuple[SubMove, ...]]]:
    """All maximal *executable* submove orderings, grouped by afterstate.

    The depth-first walk only ever extends along legal submoves, so every leaf is an
    executable play; we keep the leaves of maximal length and apply the forced-higher-die
    rule, leaving exactly the afterstates :func:`legal_moves` reports — but with *every*
    ordering that reaches each, not just the canonical one. Each value's orderings are
    sorted by :func:`_seq_key`, so the first is the canonical (lexicographically smallest)
    one and the output is deterministic.
    """
    mover = state.turn
    dice_seq = (dice[0],) * 4 if dice[0] == dice[1] else (dice[0], dice[1])

    paths: dict[_AfterKey, list[tuple[SubMove, ...]]] = {}
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
            max_used = max(max_used, len(path))
            paths.setdefault(key, []).append(path)

    visit(state, dice_seq, ())

    # Keep only maximal-length plays (drop short dead-ends), then forced-higher-die: if no
    # two-dice play exists for a non-double, the higher die must be played when playable.
    best: dict[_AfterKey, list[tuple[SubMove, ...]]] = {}
    for key, plist in paths.items():
        maximal = [p for p in plist if len(p) == max_used]
        if maximal:
            best[key] = sorted(maximal, key=_seq_key)

    if max_used == 1 and dice[0] != dice[1]:
        higher_submoves = _submoves_for_die(state, mover, max(dice))
        if higher_submoves:
            allowed: set[_AfterKey] = set()
            for sm in higher_submoves:
                nxt = apply_submove(state, mover, sm)
                allowed.add((nxt.board, nxt.bar, nxt.off))
            best = {k: v for k, v in best.items() if k in allowed}

    return best


def legal_moves(state: EnvState, dice: Dice) -> list[tuple[Move, EnvState]]:
    """Return ``(Move, afterstate)`` pairs for ``state.turn`` given ``dice``.

    Afterstates carry ``turn = opponent`` (it is their move next). The list is
    de-duplicated by afterstate, each carrying the canonical (lexicographically smallest
    executable) submove ordering; an empty list means no legal move (the turn passes).
    """
    opp = state.turn.opponent()
    out: list[tuple[Move, EnvState]] = []
    for (board, bar, off), orderings in _enumerate(state, dice).items():
        after = EnvState(
            board=board,
            bar=bar,
            off=off,
            turn=opp,
            cube_value=state.cube_value,
            cube_owner=state.cube_owner,
        )
        out.append((Move(orderings[0]), after))  # orderings[0] = canonical, see _enumerate
    return out


def legal_orderings(state: EnvState, dice: Dice) -> dict[_AfterKey, list[tuple[SubMove, ...]]]:
    """Every legal submove ordering for each afterstate of ``(state, dice)``.

    Keyed by the afterstate's ``(board, bar, off)`` (matching :func:`legal_moves`'
    afterstates), each value lists all maximal executable orderings reaching it, sorted
    canonically. Used by the web UI to let a human enter a play's submoves in any legal
    order; agents do not need it (they choose afterstates, not orderings).
    """
    return _enumerate(state, dice)
