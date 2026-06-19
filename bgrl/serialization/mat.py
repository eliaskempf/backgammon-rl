"""Serialize a played game to the Jellyfish ``.mat`` match format for gnubg.

gnubg (our analysis oracle, CLAUDE.md §8) imports ``.mat`` and the format is simple
text. There is **no formal spec** for it (even gnubg's own manual calls it "not
formally defined"), so the exact layout here — column widths, the player-name line,
the win line — is pinned empirically against gnubg's *own* ``export match mat`` output
and confirmed to round-trip via ``import mat`` (WP3 Part B).

Conventions baked in:

* **Point numbering is per-player, 1..24 from each side's own home** — the Jellyfish
  convention. WHITE bears off past absolute index 0 (so point = ``index + 1``); BLACK
  bears off past absolute index 23 (point = ``24 - index``). This mirrors
  :func:`bgrl.web.views.point_number`; it is re-derived here rather than imported
  because ``serialization`` sits *below* ``web`` in the layering (``web`` imports
  ``serialization``), so the dependency may not go the other way.
* **Hits are marked with ``*``** on the destination (gnubg's notation). We replay a
  move's submoves through :func:`bgrl.env.apply_submove` to know, for each submove,
  whether its destination held a lone opponent blot.
* **WHITE always moves first** (the env opens with WHITE to move), so WHITE is always
  ``.mat`` player 1 and turns pair cleanly into rounds (one numbered line per round).
* v1 is cubeless single games, exported as a ``0 point match`` (money session). The
  win line still records the game-end magnitude (single/gammon/backgammon) faithfully
  via the points won (1/2/3), per CLAUDE.md §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bgrl.env import BAR, OFF, EnvState, Outcome, Player, apply_submove

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bgrl.env import Dice, Move
    from bgrl.game import Step
    from bgrl.money import CubeEvent, MoneyGameResult

# Layout constants, pinned to gnubg's own ``export match mat`` so our files round-trip.
# The left (player-1 / WHITE) move cell is padded to this width before the right cell;
# 28 puts the right cell at gnubg's column (its importer is whitespace-tolerant anyway).
_MOVE_CELL_WIDTH = 28
# The "%3d) " move-number prefix; player names / win line are indented to match it.
_INDENT = "     "


def _two_columns(left: str, right: str) -> str:
    """Join a player-1 / player-2 cell pair, always leaving a whitespace gap.

    gnubg's ``.mat`` importer separates the two players' plays on a line by
    whitespace (the second begins with its own ``DD:`` token), so a wide left cell
    must never butt directly against the right one — pad to the column, but keep at
    least two spaces when the left cell overflows it.
    """
    gap = max(2, _MOVE_CELL_WIDTH - len(left))
    return f"{left}{' ' * gap}{right}".rstrip()


def _safe_name(name: str) -> str:
    """Sanitise a player name for the ``.mat`` names line (``name : score``).

    A colon inside the name gives gnubg's importer a second ``:`` token and desyncs
    its parser — concretely it crashes ``analyse`` with a buffer overflow (e.g. an
    agent label like ``llm:anthropic/claude-haiku-4-5``). Slashes/dashes are fine, so
    we only neutralise the colon and trim whitespace; an empty result falls back.
    """
    return name.replace(":", "-").strip() or "player"


def point_number(point: int, mover: Player) -> int:
    """Absolute board index ``0..23`` -> the mover's own 1..24 point number.

    Equals the checker's pip distance home, which is exactly what ``.mat`` records.
    """
    return point + 1 if mover is Player.WHITE else 24 - point


def _square_token(square: int, mover: Player) -> str:
    if square == BAR:
        return "bar"
    if square == OFF:
        return "off"
    return str(point_number(square, mover))


def _submove_tokens(move: Move, mover: Player, state: EnvState) -> list[str]:
    """Render each submove ``src/dst`` (with ``*`` on hits) in play order.

    Replays the submoves from ``state`` so a hit is detected against the board as it
    stood *just before* that submove (a later submove in the same play can hit too).
    """
    tokens: list[str] = []
    cur = state
    opponent_blot = -_sign(mover)  # a lone opponent checker on a point
    for sm in move.submoves:
        hit = sm.dst != OFF and cur.board[sm.dst] == opponent_blot
        star = "*" if hit else ""
        tokens.append(f"{_square_token(sm.src, mover)}/{_square_token(sm.dst, mover)}{star}")
        cur = apply_submove(cur, mover, sm)
    return tokens


def _sign(player: Player) -> int:
    return 1 if player is Player.WHITE else -1


def _dice_str(dice: Dice) -> str:
    """gnubg writes the two pips higher-first, e.g. ``(1, 3)`` -> ``31``."""
    hi, lo = (dice[0], dice[1]) if dice[0] >= dice[1] else (dice[1], dice[0])
    return f"{hi}{lo}"


def _move_cell(step: Step) -> str:
    """One player's play for a round: ``DD: m1 m2`` (``DD:`` alone on a forced pass)."""
    mover = step.state.turn
    head = f"{_dice_str(step.dice)}:"
    tokens = _submove_tokens(step.move, mover, step.state)
    return f"{head} {' '.join(tokens)}".rstrip() if tokens else head


def _wins_line(points: int, winner: Player) -> str:
    """The game-result line, in the winner's column (left for WHITE, right for BLACK)."""
    text = f"Wins {points} point{'s' if points != 1 else ''}"
    left, right = (text, "") if winner is Player.WHITE else ("", text)
    return _INDENT + _two_columns(left, right)


def _win_line_for(outcome: Outcome | None, cube_events: Sequence[CubeEvent]) -> str | None:
    """The win line: a drop awards the doubler the pre-double stake; else the played magnitude.

    ``None`` for an unfinished, undropped game (no win line). On a drop (last cube event not
    taken) the doubler wins ``from_value`` points; otherwise the magnitude (single / gammon /
    backgammon) — the cube value rides on the ``Doubles =>`` tokens, not the win line.
    """
    if cube_events and not cube_events[-1].taken:  # game ended on a drop
        ev = cube_events[-1]
        return _wins_line(ev.from_value, ev.doubler)
    if outcome is not None:
        return _wins_line(int(outcome.kind), outcome.winner)
    return None


def _cells(steps: Sequence[Step], cube_events: Sequence[CubeEvent]) -> list[tuple[Player, str]]:
    """The chronological ``(player, cell)`` actions, interleaving cube actions with moves.

    A double contributes two actions — ``Doubles => N`` for the doubler and ``Takes`` /
    ``Drops`` for the opponent — inserted before the ply they precede. Because the doubler
    is exactly that ply's mover, this preserves the strict WHITE/BLACK alternation the round
    grid relies on (WHITE first), so consecutive cells pair cleanly into numbered rounds. A
    drop's event sits at ``ply == len(steps)`` (after the last recorded move).
    """
    by_ply: dict[int, list[CubeEvent]] = {}
    for ev in cube_events:
        by_ply.setdefault(ev.ply, []).append(ev)

    cells: list[tuple[Player, str]] = []

    def emit_cube(ply: int) -> None:
        for ev in by_ply.get(ply, ()):
            cells.append((ev.doubler, f"Doubles => {ev.to_value}"))
            cells.append((ev.doubler.opponent(), "Takes" if ev.taken else "Drops"))

    for i, step in enumerate(steps):
        emit_cube(i)
        cells.append((step.state.turn, _move_cell(step)))
    emit_cube(len(steps))
    return cells


def _game_block(
    steps: Sequence[Step],
    outcome: Outcome | None,
    *,
    number: int,
    white_name: str,
    black_name: str,
    cube_events: Sequence[CubeEvent] = (),
) -> list[str]:
    """A ``Game N`` block: header, the **player-names line**, moves/cube actions, win line.

    gnubg's importer reads the line right after ``Game N`` as the two players' names, so
    it must sit here (inside the game), not up in the match header. With no ``cube_events``
    this renders exactly the cubeless layout (the cells are just the moves).
    """
    # gnubg's names line is ``<name> : <score>`` per column (score is 0 for a money
    # session); a bare name desyncs its importer and the whole match fails to load.
    names = " " + _two_columns(f"{_safe_name(white_name)} : 0", f"{_safe_name(black_name)} : 0")
    lines = [f" Game {number}", names]
    # WHITE is player 1; moves and cube actions alternate WHITE/BLACK, so pair them by index.
    cells = _cells(steps, cube_events)
    rounds = (len(cells) + 1) // 2
    for r in range(rounds):
        white = cells[2 * r][1]
        black = cells[2 * r + 1][1] if 2 * r + 1 < len(cells) else ""
        lines.append(f"{r + 1:3d}) " + _two_columns(white, black))
    win = _win_line_for(outcome, cube_events)
    if win is not None:
        lines.append(win)
    return lines


def match_to_mat(
    games: Sequence[tuple[Sequence[Step], Outcome | None] | tuple[object, ...]],
    *,
    white_name: str = "White",
    black_name: str = "Black",
    match_length: int = 0,
) -> str:
    """Render one or more games as a single Jellyfish ``.mat`` match (one match/file).

    ``match_length=0`` is a money session. Each game is a ``(steps, outcome)`` pair, or a
    ``(steps, outcome, cube_events)`` triple to record doubling-cube actions (WP6);
    ``outcome=None`` (an unfinished or dropped game) omits / overrides the win line via the
    cube events. WHITE is player 1 in every game (the env always opens with WHITE).
    """
    # Leading space matches gnubg's own export; its importer keys on "point match".
    lines = [f" {match_length} point match"]
    for i, game in enumerate(games, start=1):
        steps, outcome = game[0], game[1]
        cube_events = game[2] if len(game) > 2 else ()
        lines.append("")
        lines.extend(
            _game_block(
                steps,
                outcome,
                number=i,
                white_name=white_name,
                black_name=black_name,
                cube_events=cube_events,
            )
        )
    return "\n".join(lines) + "\n"


def money_game_to_mat(
    result: MoneyGameResult,
    *,
    white_name: str = "White",
    black_name: str = "Black",
) -> str:
    """Render a :class:`~bgrl.money.MoneyGameResult` (doubling cube included) as a ``.mat``.

    The cube history (``Doubles =>`` / ``Takes`` / ``Drops``) is interleaved with the moves
    and a drop is recorded with the doubler winning the pre-double stake — so gnubg imports a
    faithful money game, cube and all. The exact cube-token spelling is the Jellyfish
    convention; like the rest of this module it is pinned by the gnubg round-trip test.
    """
    return match_to_mat(
        [(result.steps, result.outcome, result.cube_events)],
        white_name=white_name,
        black_name=black_name,
    )


def game_to_mat(
    steps: Sequence[Step],
    outcome: Outcome | None,
    *,
    white_name: str = "White",
    black_name: str = "Black",
    match_length: int = 0,
) -> str:
    """Render a single played game as a Jellyfish ``.mat`` match.

    Convenience wrapper over :func:`match_to_mat` for the common one-game case (the web
    ``/export_mat`` endpoint and per-game analysis). ``steps`` is the recorded
    trajectory (``bgrl.game.Step`` from either self-play ``play_game(record=True)`` or a
    web ``GameSession``); ``outcome`` is the final result or ``None`` if still in play.
    """
    return match_to_mat(
        [(steps, outcome)],
        white_name=white_name,
        black_name=black_name,
        match_length=match_length,
    )
