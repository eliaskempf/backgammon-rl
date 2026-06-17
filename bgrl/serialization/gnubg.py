"""Drive GNU Backgammon headlessly to analyse a ``.mat`` match.

gnubg is our analysis oracle (CLAUDE.md §8): it imports our exported ``.mat``, evaluates
each chequer play at a fixed ply, and tells us the equity of the move actually made
versus its own preferred move. The gap is **equity loss** — the standard, scale-free
strength metric (lower = stronger).

Approach: we run gnubg in batch mode with its embedded Python interpreter
(``gnubg -t -q -p <script>``) and have the script walk the analysed match object
(:func:`gnubg.match`) and emit JSON we parse back here. This is more robust than
scraping the text export — the equities come through as numbers, and gnubg's preferred
move is read directly rather than reconstructed. Inputs are passed via environment
variables (gnubg does not reliably forward CLI args to the embedded script's argv).

**Ply convention (CLAUDE.md §9):** we use gnubg's own numbering internally (raw net =
0-ply), so the ``plies`` argument here *is* gnubg's ply count — no translation needed at
this boundary.

gnubg is an optional system dependency; everything here degrades gracefully when it is
absent (see :func:`gnubg_available`). Install on Ubuntu with ``apt-get install gnubg``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Move-equity agreement tolerance: a play whose equity matches gnubg's best within this
# (in normalized equity) counts as "agrees with gnubg" even if the notation differs.
_AGREE_EPS = 1e-4


@dataclass(frozen=True, slots=True)
class MoveAnalysis:
    """gnubg's verdict on one chequer play: what was made vs. what it preferred."""

    game: int  # 0-based game index within the match
    ply: int  # 0-based move index within that game
    player: str  # "White" | "Black" (player 1 / player 2 of the .mat)
    dice: tuple[int, int]
    move_made: str  # gnubg's notation for the move actually played
    best_move: str  # gnubg's notation for its preferred move
    equity_made: float
    equity_best: float

    @property
    def equity_loss(self) -> float:
        """How much equity the played move gave up vs. gnubg's best (>= 0)."""
        return self.equity_best - self.equity_made

    @property
    def agreed(self) -> bool:
        """True if the played move is (equity-)equivalent to gnubg's preferred move."""
        return self.equity_loss <= _AGREE_EPS


@dataclass(frozen=True, slots=True)
class SideSummary:
    """Aggregate strength stats for one side over an analysed match."""

    moves: int
    mean_equity_loss: float
    agreement: float  # fraction of moves matching gnubg's preferred play


def gnubg_available(gnubg_bin: str = "gnubg") -> bool:
    """True if the gnubg binary is on ``PATH`` (gates the live pipeline + its tests)."""
    return shutil.which(gnubg_bin) is not None


def summarize(moves: list[MoveAnalysis]) -> dict[str, SideSummary]:
    """Per-side (and overall) mean equity loss + gnubg-agreement rate."""
    buckets: dict[str, list[MoveAnalysis]] = {"White": [], "Black": [], "overall": []}
    for m in moves:
        buckets[m.player].append(m)
        buckets["overall"].append(m)
    out: dict[str, SideSummary] = {}
    for key, group in buckets.items():
        n = len(group)
        mean_loss = sum(m.equity_loss for m in group) / n if n else 0.0
        agree = sum(1 for m in group if m.agreed) / n if n else 0.0
        out[key] = SideSummary(moves=n, mean_equity_loss=mean_loss, agreement=agree)
    return out


def analyse_mat(
    mat_path: Path | str,
    *,
    plies: int = 2,
    gnubg_bin: str = "gnubg",
    timeout: float = 300.0,
) -> list[MoveAnalysis]:
    """Analyse every chequer play in a ``.mat`` file and return per-move equity loss.

    Runs gnubg headlessly at ``plies`` (gnubg's own numbering, CLAUDE.md §9). Raises
    :class:`RuntimeError` if gnubg is missing or its run fails — callers that want
    graceful skipping should guard with :func:`gnubg_available` first.
    """
    mat_path = Path(mat_path)
    if shutil.which(gnubg_bin) is None:
        raise RuntimeError(f"gnubg not found on PATH (looked for {gnubg_bin!r})")
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "analyse.py"
        out_path = Path(tmp) / "analysis.json"
        script_path.write_text(_GNUBG_SCRIPT)
        env = {
            **os.environ,
            "BGRL_MAT": str(mat_path.resolve()),
            "BGRL_OUT": str(out_path),
            "BGRL_PLIES": str(plies),
        }
        proc = subprocess.run(
            [gnubg_bin, "-t", "-q", "-p", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if not out_path.is_file():
            raise RuntimeError(
                "gnubg analysis produced no output\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        raw = json.loads(out_path.read_text())

    return [
        MoveAnalysis(
            game=r["game"],
            ply=r["ply"],
            player=r["player"],
            dice=(r["dice"][0], r["dice"][1]),
            move_made=r["move_made"],
            best_move=r["best_move"],
            equity_made=r["equity_made"],
            equity_best=r["equity_best"],
        )
        for r in raw
    ]


# Runs inside gnubg's embedded Python (3.12 in current builds; kept dependency-free).
# Imports the match, analyses it, and dumps a flat list of chequer-play records as JSON.
# Schema (pinned against gnubg 1.07): gnubg.match() -> {"games": [{"game": [action, ...]}]},
# each move action has player "X"/"O", dice, and after analysis an "analysis" dict with
# "moves" (candidates sorted best-first, each {"move": [[from,to],...], "score": equity})
# and "imove" (index of the move actually played; points use 0=off, 25=bar).
# Set BGRL_PROBE=1 to dump a shallow view of the structure instead of extracting.
_GNUBG_SCRIPT = r'''
import os, json
import gnubg

mat = os.environ["BGRL_MAT"]
out = os.environ["BGRL_OUT"]
plies = int(os.environ.get("BGRL_PLIES", "2"))

gnubg.command("set analysis chequerplay evaluation plies %d" % plies)
gnubg.command("set analysis cubedecision evaluation plies %d" % plies)
gnubg.command("import mat %s" % mat)
gnubg.command("analyse match")

m = gnubg.match(analysis=1, boards=0)

if os.environ.get("BGRL_PROBE"):
    def shallow(x, depth=0):
        if depth > 5:
            return "..."
        if isinstance(x, dict):
            return dict((str(k), shallow(v, depth + 1)) for k, v in list(x.items())[:40])
        if isinstance(x, (list, tuple)):
            return [shallow(v, depth + 1) for v in list(x)[:4]]
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        return repr(x)[:200]
    open(out, "w").write(json.dumps(shallow(m), indent=2, default=repr))
    raise SystemExit(0)

PLAYER_NAME = {"X": "White", "O": "Black", 0: "White", 1: "Black"}


def move_str(mv):
    # gnubg candidate moves are [[from, to], ...] in 1..24 with 0=off and 25=bar.
    if not mv:
        return ""
    tok = lambda p: "off" if p == 0 else ("bar" if p == 25 else str(p))
    return " ".join("%s/%s" % (tok(a), tok(b)) for a, b in mv)


records = []
games = m["games"] if isinstance(m, dict) else m
for gi, g in enumerate(games):
    actions = g["game"] if isinstance(g, dict) else g
    ply = 0
    for action in actions:
        if (action.get("action") or action.get("type")) != "move":
            continue
        ply += 1
        analysis = action.get("analysis") or {}
        candidates = analysis.get("moves") or []
        imove = analysis.get("imove")
        # Need a ranked candidate list and a known played-move index to score the play.
        if not candidates or imove is None or imove < 0 or imove >= len(candidates):
            continue
        best = candidates[0]
        made = candidates[imove]
        records.append({
            "game": gi,
            "ply": ply - 1,
            "player": PLAYER_NAME.get(action.get("player"), str(action.get("player"))),
            "dice": list(action.get("dice", (0, 0))),
            "move_made": move_str(made.get("move")),
            "best_move": move_str(best.get("move")),
            "equity_made": float(made.get("score")),
            "equity_best": float(best.get("score")),
        })

open(out, "w").write(json.dumps(records))
'''
