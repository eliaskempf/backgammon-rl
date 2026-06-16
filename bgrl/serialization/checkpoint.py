"""Algorithm-agnostic checkpoint I/O.

A checkpoint is a self-describing dict (saved with :func:`torch.save`) that the
web server (WP3) and eval scripts can load **without knowing which algorithm
produced it** — that is the whole point of :func:`load_agent`. It records enough
to rebuild the network (``net_arch`` + ``weights``) and the versions needed to
keep old files loadable (``format_version``, ``encoding_version``).

Layout::

    {
      "format_version":   int,    # this module's schema (CHECKPOINT_FORMAT_VERSION)
      "net_arch":         dict,   # net.arch_config(): {"class": ..., **ctor kwargs}
      "weights":          dict,   # net.state_dict()
      "encoding_version": int,    # bgrl.env.ENCODING_VERSION at save time
      "outcome_dim":      int,    # OUTCOME_DIM (the fixed outcome-vector length)
      "trained_with":     str,    # "random" | "td_lambda" | ... (informational)
      "metadata":         dict,   # created_at, git_sha, games_trained, notes, ...
    }
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bgrl.agents.base import Agent
from bgrl.agents.value_agent import ValueAgent
from bgrl.env import ENCODING_VERSION
from bgrl.nets.base import OUTCOME_DIM, ValueNet
from bgrl.nets.value_net import MLPValueNet

CHECKPOINT_FORMAT_VERSION = 1

# Maps the ``net_arch["class"]`` tag back to a constructor. New net classes
# (WP1/WP5) register here so old checkpoints keep loading.
NET_REGISTRY: dict[str, type[MLPValueNet]] = {"MLPValueNet": MLPValueNet}


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def save_checkpoint(
    net: nn.Module,
    path: str | Path,
    *,
    trained_with: str,
    metadata: dict | None = None,
) -> None:
    """Serialise ``net`` to ``path`` as a self-describing checkpoint.

    ``net`` must expose ``arch_config()`` (every net in :data:`NET_REGISTRY`
    does). ``created_at`` (UTC) and ``git_sha`` are stamped automatically and
    merged with any caller ``metadata`` (e.g. ``games_trained``, ``notes``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": datetime.now(UTC).isoformat(), "git_sha": _git_sha()}
    if metadata:
        meta.update(metadata)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "net_arch": net.arch_config(),  # type: ignore[operator]  # registered nets provide this
        "weights": net.state_dict(),
        "encoding_version": ENCODING_VERSION,
        "outcome_dim": OUTCOME_DIM,
        "trained_with": trained_with,
        "metadata": meta,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load and validate a checkpoint dict.

    Raises ``ValueError`` on an unsupported ``format_version`` or an
    ``encoding_version`` mismatch (no migration exists yet — a feature-layout
    change would silently corrupt every input, so we fail loudly instead).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    fmt = checkpoint.get("format_version")
    if fmt != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version {fmt!r} (expected {CHECKPOINT_FORMAT_VERSION})"
        )
    enc = checkpoint.get("encoding_version")
    if enc != ENCODING_VERSION:
        raise ValueError(
            f"checkpoint encoding_version {enc!r} != current {ENCODING_VERSION}; "
            "no migration available"
        )
    return checkpoint


def load_net(checkpoint: dict) -> ValueNet:
    """Reconstruct the network from a (validated) checkpoint dict, in eval mode."""
    arch = checkpoint["net_arch"]
    net_class = arch["class"]
    if net_class not in NET_REGISTRY:
        raise ValueError(f"unknown net class {net_class!r}; registered: {sorted(NET_REGISTRY)}")
    net = NET_REGISTRY[net_class].from_config(arch)
    net.load_state_dict(checkpoint["weights"])
    net.eval()
    return net


def load_agent(checkpoint: dict, *, rng: np.random.Generator | None = None) -> Agent:
    """Build a playable agent from a checkpoint — the generic factory WP3 uses.

    Returns a 0-ply :class:`~bgrl.agents.value_agent.ValueAgent` wrapping the
    loaded net (the universal value-based play policy). WP2 extends this to wrap
    the net in n-ply search; ``trained_with`` is informational and does not
    change the play policy.
    """
    return ValueAgent(load_net(checkpoint), rng=rng)
