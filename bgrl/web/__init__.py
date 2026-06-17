"""Web server + browser play UI (WP3 Part A).

A thin FastAPI app serves a disposable frontend and exposes a stable REST API; the
backend ``Env`` is the sole authority on move legality. Public surface is the app
factory and the server-side session types.
"""

from bgrl.web.app import create_app
from bgrl.web.session import GameSession, SessionStore

__all__ = ["GameSession", "SessionStore", "create_app"]
