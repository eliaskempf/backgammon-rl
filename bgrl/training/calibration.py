"""Per-head calibration diagnostic for the multi-head value net (WP6 A4).

Once WP6 trains all five outcome heads, the question is whether each head's predicted
probability matches the realised frequency — in particular whether the rare
gammon/backgammon heads actually learned or **collapsed to ~0**, the failure mode online
TD risks when the signal is a few percent of games. This module measures that, and it is
the gate for the optional rollout fine-tuning (A5): a flat / collapsed reliability curve
on the backgammon (or gammon) head is what would justify building A5; otherwise it is
skipped.

Method: play cubeless greedy self-play with the (non-learning) net — the deployment
distribution — and for every non-terminal afterstate compare the net's predicted outcome
vector (mover POV, the canonical ``V(a)`` query) against the *realised* cumulative
outcome of that game from the same POV. Aggregated per head as the mean predicted vs mean
realised rate (the collapse signal: predicted ≈ 0 while realised > 0) plus the expected
calibration error (ECE) over probability bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgrl.agents.value_agent import ValueAgent
from bgrl.env import Outcome, Player, RandomDiceSource, WinKind, encode, is_terminal
from bgrl.game import play_game
from bgrl.nets.base import OUTCOME_DIM, ValueNet
from bgrl.nets.equity import flip_outcome

HEAD_NAMES: tuple[str, ...] = ("p_win", "p_win_g", "p_win_bg", "p_lose_g", "p_lose_bg")
"""Outcome-vector head labels, in index order (matches :mod:`bgrl.nets.base`)."""


@dataclass(frozen=True, slots=True)
class HeadCalibration:
    """Calibration of a single outcome head over the sampled positions.

    ``predicted_mean`` and ``realized_mean`` are the head's average prediction and its
    average realised indicator; a large ``realized_mean`` with a near-zero
    ``predicted_mean`` is the collapse signal. ``ece`` is the expected calibration error
    (probability-weighted mean ``|predicted - realised|`` over bins): 0 is perfect.
    """

    name: str
    predicted_mean: float
    realized_mean: float
    ece: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """A full per-head calibration snapshot over ``samples`` afterstates from ``games``."""

    games: int
    samples: int
    heads: tuple[HeadCalibration, ...]

    def to_metrics(self) -> dict[str, float]:
        """Flatten to a ``cal/<head>_<stat>`` dict for CSV / W&B logging."""
        out: dict[str, float] = {}
        for h in self.heads:
            out[f"cal/{h.name}_pred"] = h.predicted_mean
            out[f"cal/{h.name}_real"] = h.realized_mean
            out[f"cal/{h.name}_ece"] = h.ece
        return out


def _realized_outcome(outcome: Outcome, perspective: Player) -> np.ndarray:
    """The game's realised cumulative outcome 5-vector from ``perspective``'s POV.

    From the winner's POV this is ``[1, kind>=GAMMON, kind>=BACKGAMMON, 0, 0]``; from the
    loser's POV it is the perspective flip of that (win heads 0, loss-magnitude heads
    set) — the same involution the TD bootstrap uses, so the label and the target agree.
    """
    won = np.array(
        [
            1.0,
            float(outcome.kind >= WinKind.GAMMON),
            float(outcome.kind >= WinKind.BACKGAMMON),
            0.0,
            0.0,
        ]
    )
    return won if outcome.winner is perspective else flip_outcome(won)


def collect_predictions(
    net: ValueNet,
    *,
    games: int,
    rng: np.random.Generator,
    max_plies: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy self-play ``games`` with ``net``; return ``(predicted, realized)`` arrays.

    Both have shape ``(samples, OUTCOME_DIM)``: one row per non-terminal afterstate, the
    net's prediction (mover POV) and the realised cumulative outcome from that POV.
    Truncated games (no winner within ``max_plies``) are skipped. A non-learning
    :class:`~bgrl.agents.value_agent.ValueAgent` plays both seats, so no learning hooks
    fire. Predictions are batched into a single :meth:`ValueNet.evaluate` call.
    """
    agent = ValueAgent(net)  # rng=None -> deterministic greedy; dice supply the diversity
    feats: list[np.ndarray] = []
    persp: list[Player] = []
    outcomes: list[Outcome] = []
    for _ in range(games):
        result = play_game(agent, agent, RandomDiceSource(rng), max_plies=max_plies, record=True)
        if result.outcome is None:
            continue
        for s in result.steps:
            a = s.afterstate
            if is_terminal(a):
                continue
            feats.append(encode(a, a.turn))
            persp.append(a.turn)
            outcomes.append(result.outcome)

    if not feats:
        empty = np.zeros((0, OUTCOME_DIM))
        return empty, empty
    predicted = np.asarray(net.evaluate(np.stack(feats)), dtype=np.float64)
    realized = np.stack([_realized_outcome(o, p) for o, p in zip(outcomes, persp, strict=True)])
    return predicted, realized


def _ece(predicted: np.ndarray, realized: np.ndarray, bins: int) -> float:
    """Expected calibration error of one head: Σ_b (n_b/n)·|mean_pred_b - mean_real_b|."""
    if predicted.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    # digitize against the interior edges so values map to bins 0..bins-1.
    idx = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    n = predicted.size
    total = 0.0
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            total += (count / n) * abs(float(predicted[mask].mean()) - float(realized[mask].mean()))
    return total


def calibration_report(
    net: ValueNet,
    *,
    games: int,
    rng: np.random.Generator,
    bins: int = 10,
    max_plies: int = 10_000,
) -> CalibrationReport:
    """Play ``games`` self-play games and summarise per-head calibration.

    Returns a :class:`CalibrationReport`; expect ``p_win`` tightest, gammon looser, and
    backgammon the noisiest (its base rate is ~1-2% of games) — that ordering is normal,
    not a failure. A near-zero ``predicted_mean`` against a clearly positive
    ``realized_mean`` on a magnitude head is the collapse that gates A5.
    """
    predicted, realized = collect_predictions(net, games=games, rng=rng, max_plies=max_plies)
    heads = tuple(
        HeadCalibration(
            name=name,
            predicted_mean=float(predicted[:, k].mean()) if predicted.size else 0.0,
            realized_mean=float(realized[:, k].mean()) if realized.size else 0.0,
            ece=_ece(predicted[:, k], realized[:, k], bins) if predicted.size else 0.0,
        )
        for k, name in enumerate(HEAD_NAMES)
    )
    return CalibrationReport(games=games, samples=len(predicted), heads=heads)
