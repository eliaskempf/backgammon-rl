# Web play UI — development & build (WP3 Part A)

The play server is a FastAPI app (`bgrl/web/`) that serves a disposable frontend and
exposes a stable REST API. **All move legality comes from the backend `Env`** — the
frontend only renders state and collects intent.

## Run the server

```bash
uv sync --group web                       # install fastapi + uvicorn (web group)
uv run python scripts/play_web.py --checkpoints-dir runs/wp1
# open http://127.0.0.1:8000
```

With no checkpoints in `--checkpoints-dir`, only the built-in `random` opponent is
offered. Any `<name>.pt` checkpoint in that directory becomes a selectable opponent
(loaded via the WP0 `load_agent` factory).

## Frontend (Vite + React + TypeScript)

Source lives in `frontend/`; the **built bundle is committed** to `bgrl/web/static/`
so the Python package serves a ready UI without Node. Node is only needed to rebuild.

Requires Node 18+ and npm:

```bash
# Ubuntu: sudo apt-get install -y nodejs npm   (or use nvm)
cd frontend
npm install
npm run dev      # hot-reload dev server on :5173, proxies API calls to :8000
npm run build    # type-checks and emits the production bundle to ../bgrl/web/static/
```

Commit the regenerated `bgrl/web/static/` after `npm run build`. The frontend is
disposable: it may be replaced freely as long as the REST contract in
`bgrl/web/schemas.py` holds.

## REST API

`POST /new_game`, `POST /roll`, `GET /legal_moves`, `POST /move`, `POST /agent_move`,
`GET /checkpoints`, `POST /export_mat` (501 until WP3 Part B). Interactive docs at
`/docs` when the server is running. The incremental human move is built client-side by
filtering the full legal-move list by the chosen submove prefix; the move is submitted
by `move_id` and re-validated server-side.
