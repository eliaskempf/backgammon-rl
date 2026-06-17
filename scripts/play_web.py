"""Serve the browser play UI.

Thin wrapper: wires a checkpoints directory into the FastAPI app and runs uvicorn.
All logic lives in :mod:`bgrl.web`.

    uv run python scripts/play_web.py --checkpoints-dir runs/wp1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from bgrl.web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the bgrl browser play UI.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("runs"),
        help="directory scanned for *.pt opponent checkpoints",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(checkpoints_dir=args.checkpoints_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
