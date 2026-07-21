"""Shared test helpers for server integration tests.

Import from this module (``from helpers import bearer_header, ...``) — do NOT
import from ``conftest``, which is a pytest plugin file and not a regular
Python module.
"""

from sqlalchemy import delete

from db import SessionLocal
from models import APIKey, RefreshTokenJti, RequestLog, User


def clean_auth_state():
    """Wipe users, API keys, refresh-token JTIs, and request logs."""
    with SessionLocal() as session:
        session.execute(delete(RequestLog))
        session.execute(delete(RefreshTokenJti))
        session.execute(delete(APIKey))
        session.execute(delete(User))
        session.commit()


def bearer_header(token: str) -> dict:
    """Return ``{"Authorization": "Bearer <token>"}``."""
    return {"Authorization": f"Bearer {token}"}


def register_first_admin(client, email="admin@example.com", password="hunter123") -> str:
    """Register the first admin user; return the access token."""
    resp = client.post(
        "/auth/register",
        json={"name": "Admin", "email": email, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def disable_app_rate_limiters(app):
    """Disable rate limiters on the app and the auth-router module reference.

    Must be called *after* the app is loaded (via ``load_app``), because
    ``importlib.reload(server_main)`` can rebind ``rate_limit`` to a fresh
    module, leaving any pre-load toggle on a different object.
    """
    app.state.limiter.enabled = False
    app.state.limiter.reset()

    import routers.auth as rauth

    rauth.limiter.enabled = False
    rauth.limiter.reset()