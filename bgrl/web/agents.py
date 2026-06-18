"""Opponent registry: resolve an opponent name to an :class:`Agent`.

``"random"`` is always available (the dummy opponent WP3 builds against before real
checkpoints exist); any other name is a checkpoint file ``<name>.pt`` under the
server's checkpoints directory, loaded through the WP0 ``load_agent`` factory — so
the server plays *any* trained net without knowing which algorithm produced it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bgrl.agents import Agent, RandomAgent
from bgrl.serialization import load_agent, load_checkpoint
from bgrl.web.schemas import CheckpointInfo

RANDOM_OPPONENT = "random"


class UnknownOpponent(Exception):
    """The requested opponent is neither ``"random"`` nor a known checkpoint."""


def make_opponent(name: str, *, checkpoints_dir: Path, rng: np.random.Generator) -> Agent:
    if name == RANDOM_OPPONENT:
        return RandomAgent(rng)
    path = checkpoints_dir / f"{name}.pt"
    if not path.is_file():
        raise UnknownOpponent(name)
    # Deterministic greedy play (rng omitted) for a stable opponent.
    return load_agent(load_checkpoint(path))


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
