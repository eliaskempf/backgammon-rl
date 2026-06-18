"""Serve the browser play UI.

Thin wrapper: wires a checkpoints directory into the FastAPI app and runs uvicorn.
All logic lives in :mod:`bgrl.web`.

    uv run python scripts/play_web.py            # serves the committed opponent ladder
    uv run python scripts/play_web.py --checkpoints-dir runs/sweep/lr0.1_lam0.7_h64_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

import bgrl.web
from bgrl.web import create_app

# The curated, committed opponent ladder shipped with the package (see
# scripts/curate_web_opponents.py). Resolved from the package so the default works
# regardless of the current working directory.
DEFAULT_CHECKPOINTS_DIR = Path(bgrl.web.__file__).parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the bgrl browser play UI.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="directory scanned for *.pt opponent checkpoints",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(checkpoints_dir=args.checkpoints_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
