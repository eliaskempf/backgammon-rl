"""Opponent registry: resolve an opponent name to an :class:`Agent`.

``"random"`` is always available (the dummy opponent WP3 builds against before real
checkpoints exist); any other name is a checkpoint file ``<name>.pt`` under the
server's checkpoints directory, loaded as a value net — so the server plays *any*
trained net without knowing which algorithm produced it. ``plies``/``top_k`` choose how
much WP2 expectimax search wraps that net: 0 = the raw 0-ply :class:`ValueAgent`, >=1 =
n-ply lookahead (``top_k`` prunes candidates so 2-ply stays interactive).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bgrl.agents import Agent, ExpectimaxAgent, RandomAgent, ValueAgent
from bgrl.serialization import load_checkpoint, load_net
from bgrl.web.schemas import CheckpointInfo

RANDOM_OPPONENT = "random"


class UnknownOpponent(Exception):
    """The requested opponent is neither ``"random"`` nor a known checkpoint."""


def make_opponent(
    name: str,
    *,
    checkpoints_dir: Path,
    rng: np.random.Generator,
    plies: int = 0,
    top_k: int | None = None,
) -> Agent:
    if name == RANDOM_OPPONENT:
        return RandomAgent(rng)
    path = checkpoints_dir / f"{name}.pt"
    if not path.is_file():
        raise UnknownOpponent(name)
    net = load_net(load_checkpoint(path))
    # Deterministic play (rng omitted) for a stable opponent.
    if plies <= 0:
        return ValueAgent(net)
    return ExpectimaxAgent(net, plies=plies, top_k=top_k)


def list_checkpoints(checkpoints_dir: Path) -> list[CheckpointInfo]:
    """Loadable checkpoints under ``checkpoints_dir`` (silently skips unreadable ones)."""
    if not checkpoints_dir.is_dir():
        return []
    infos: list[CheckpointInfo] = []
    for path in sorted(checkpoints_dir.glob("*.pt")):
        try:
            checkpoint = load_checkpoint(path)
        except Exception:  # a bad/incompatible checkpoint just isn't offered as an opponent
            continue
        metadata = checkpoint.get("metadata") or {}
        infos.append(
            CheckpointInfo(
                name=path.stem,
                trained_with=checkpoint.get("trained_with"),
                games_trained=metadata.get("games_trained"),
                created_at=metadata.get("created_at"),
                win_rate=metadata.get("win_rate"),
                eval_opponent=metadata.get("eval_opponent"),
            )
        )
    return infos
