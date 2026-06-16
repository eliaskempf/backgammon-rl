"""Machine fingerprint for cross-host benchmark comparison.

Everything is best-effort and guarded so a headless cluster node without psutil
(or git) still produces a usable record.
"""

from __future__ import annotations

import os
import platform
import subprocess


def _cpu_model() -> str | None:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        return None


def _version(module: str) -> str | None:
    try:
        return getattr(__import__(module), "__version__", None)
    except Exception:
        return None


def _torch_threads() -> int | None:
    try:
        import torch

        return torch.get_num_threads()
    except Exception:
        return None


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


def machine_fingerprint() -> dict:
    """A JSON-serialisable description of the host the benchmark ran on."""
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cores_logical": os.cpu_count(),
        "cores_physical": _physical_cores(),
        "ram_gb": _ram_gb(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "torch_num_threads": _torch_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "git_sha": _git_sha(),
    }
