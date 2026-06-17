"""Resume correctness for ``scripts/train.py`` / the training loop.

The headline guarantee is **bit-exact** resume: continuing a run from a saved
checkpoint + RNG state produces weights identical to an uninterrupted run. This works
because TD(λ) traces are zero at every game boundary (``episode_end``), so the net
weights + the dice-stream RNG state are the entire resumable state — no trainer state to
serialise. The slow test exercises the real SIGTERM → checkpoint → ``--resume`` path of
the script end to end.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from bgrl.agents.td_agent import TDAgent
from bgrl.nets.value_net import MLPValueNet
from bgrl.serialization import load_checkpoint
from bgrl.training.loop import train

_REPO = Path(__file__).resolve().parents[2]


def _train(net: MLPValueNet, games: int, rng: np.random.Generator) -> None:
    train(TDAgent(net, lam=0.7, lr=0.1, gamma=1.0), games=games, rng=rng)


def test_resume_is_bit_exact() -> None:
    """Segmented training (M, then resume for K) equals one continuous M+K run."""
    torch.set_num_threads(1)  # single-thread determinism underpins the guarantee
    m, k, hidden = 40, 40, 8

    torch.manual_seed(0)
    net_cont = MLPValueNet(hidden=hidden)
    _train(net_cont, m + k, np.random.default_rng(0))

    # First segment: M games, then snapshot exactly what a checkpoint persists.
    torch.manual_seed(0)
    net_seg = MLPValueNet(hidden=hidden)
    gen = np.random.default_rng(0)
    _train(net_seg, m, gen)
    saved_weights = {key: val.clone() for key, val in net_seg.state_dict().items()}
    saved_rng_state = gen.bit_generator.state

    # Resume: load the weights into a fresh net + a fresh agent (traces zero = a real
    # game boundary), restore the generator, continue for K games.
    net_res = MLPValueNet(hidden=hidden)
    net_res.load_state_dict(saved_weights)
    gen_res = np.random.default_rng(0)
    gen_res.bit_generator.state = saved_rng_state
    _train(net_res, k, gen_res)

    for resumed, continuous in zip(net_res.parameters(), net_cont.parameters(), strict=True):
        assert torch.equal(resumed, continuous)


def test_rng_state_survives_torch_save(tmp_path: Path) -> None:
    """A numpy Generator state round-trips through torch.save (how we stash it)."""
    gen = np.random.default_rng(123)
    gen.random(7)  # advance off the seeded start
    state = gen.bit_generator.state

    path = tmp_path / "state.pt"
    torch.save({"rng": state}, path)
    restored = torch.load(path, weights_only=False)["rng"]

    a, b = np.random.default_rng(), np.random.default_rng()
    a.bit_generator.state = state
    b.bit_generator.state = restored
    assert np.array_equal(a.random(16), b.random(16))


@pytest.mark.slow
def test_sigterm_checkpoints_and_resumes(tmp_path: Path) -> None:
    """SIGTERM mid-run writes latest.pt and exits 0; --resume finishes the run."""

    def run(args: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                "scripts/train.py",
                "--hidden",
                "8",
                "--eval-every",
                "0",
                "--save-every",
                "500",
                "--out-dir",
                str(tmp_path),
                "--seed",
                "0",
                *args,
            ],
            cwd=_REPO,
        )

    # Start a run far too long to finish; wait for the first latest.pt, then SIGTERM.
    proc = run(["--games", "1000000"])
    latest = tmp_path / "latest.pt"
    deadline = time.monotonic() + 60
    while not latest.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert latest.exists(), "latest.pt was not written before timeout"

    proc.terminate()  # SIGTERM on POSIX
    assert proc.wait(timeout=30) == 0, "clean exit expected after SIGTERM"
    done = int(load_checkpoint(latest)["metadata"]["games_trained"])
    assert 0 < done < 1_000_000

    # Resume to a small target just past where we stopped; it must complete.
    target = done + 1000
    assert run(["--games", str(target), "--resume"]).wait(timeout=120) == 0
    assert (tmp_path / "final.pt").exists()
    assert int(load_checkpoint(latest)["metadata"]["games_trained"]) == target
