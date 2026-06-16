"""TD-Gammon-style shallow value network.

Input is ``encode(afterstate, mover_pov)`` (length :data:`~bgrl.env.encoding.N_FEATURES`);
output is a fixed-length outcome vector ``[p_win, p_win_gammon, p_win_bg,
p_lose_gammon, p_lose_bg]`` from the mover's POV. v1 may train only ``p_win``, but
the shape is fixed so the cube and gammon scoring slot in later. Move selection
consumes *equity* (a later module), never these raw outputs.

This module is intentionally minimal — for WP0 it provides a random-weight net to
benchmark forward-pass throughput; the architecture stays swappable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from bgrl.env.encoding import N_FEATURES

OUTCOME_DIM = 5


class MLPValueNet(nn.Module):
    """``N_FEATURES -> hidden -> OUTCOME_DIM`` MLP with sigmoid activations."""

    def __init__(
        self,
        hidden: int = 64,
        n_features: int = N_FEATURES,
        outcome_dim: int = OUTCOME_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, outcome_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.inference_mode()
    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """Batched convenience wrapper: float32 ``(..., N_FEATURES)`` -> ``(..., OUTCOME_DIM)``."""
        self.eval()
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        return self.net(x).numpy()
