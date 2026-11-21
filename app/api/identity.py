"""Compute-side identity resolution for gateway-fronted vs internal-panel access.

The 4090 backend is reached two ways:

* **Via the public gateway** — it authenticates the user and injects
  ``X-Retrival-User-Id`` / ``X-Retrival-User-Tier`` before forwarding. Presence of
  the id header means "gateway already vouched for this user".
* **Directly from the internal admin panel** over the
  private lab overlay — no gateway, so no identity headers. Such LAN-trusted
  callers act as a single implicit ``internal-admin`` (same trust level the other
  admin routes already assume — they have no auth and rely on NAT/LAN isolation).

Public traffic cannot reach compute directly (NAT/W111), so LAN + no-headers is a
safe internal-admin signal, and gateway traffic always carries headers.
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Request

from core.internal_gate import is_private_lan

INTERNAL_ADMIN_ID = "internal-admin"

# Internal (LAN) vs public traffic is separated by LISTENING PORT, not by source IP —
# the lab overlay SNATs peer traffic so source IPs are unreliable (a co-tenant and the
# admin can both appear as the box's own 10.100.100.111). The backend binds TWO ports:
#   * INTERNAL door (INTERNAL_API_PORT, default 8000, 0.0.0.0) — the trusted lab
#     overlay; a private-LAN peer
#     here acts as internal-admin with no auth.
#   * PUBLIC door (PUBLIC_API_PORT, default 8001, loopback-only) — where the reverse xray
#     tunnel delivers public traffic. A request arriving here NEVER resolves to
#     internal-admin: it must carry a gateway-injected identity or it is denied.
# So public callers cannot reach the internal door (only the overlay can), and tunnelled
# traffic lands on the public door which refuses the admin shortcut.


def _read_api_port(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be an integer between 1 and 65535; got {raw!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"{name} must be between 1 and 65535; got {port}")
    return port


INTERNAL_API_PORT = _read_api_port("INTERNAL_API_PORT", 8000)
PUBLIC_API_PORT = _read_api_port("PUBLIC_API_PORT", 8001)
if INTERNAL_API_PORT == PUBLIC_API_PORT:
    raise RuntimeError(
        "INTERNAL_API_PORT and PUBLIC_API_PORT must use different ports to preserve "
        "the internal-admin/public-gateway trust boundary"
    )

LISTEN_ADDRESSES = (
    ("0.0.0.0", INTERNAL_API_PORT),
    ("127.0.0.1", PUBLIC_API_PORT),
)


def _server_port(request: Request) -> int | None:
    scope = getattr(request, "scope", None)
    server = scope.get("server") if isinstance(scope, dict) else None
    return server[1] if server and len(server) >= 2 else None


def _is_public_door(request: Request) -> bool:
    """True if the request arrived on the loopback-only public tunnel listener."""
    return _server_port(request) == PUBLIC_API_PORT


def _is_internal_door(request: Request) -> bool:
    """True only for the explicitly configured trusted-LAN listener."""
    return _server_port(request) == INTERNAL_API_PORT


def _client_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _lan_is_admin(request: Request) -> bool:
    """True iff this request should act as the trusted internal-admin: it must arrive on
    the INTERNAL door from a private, non-loopback LAN peer."""
    return _is_internal_door(request) and is_private_lan(_client_host(request))


def resolve_owner(request: Request, user_id_header: str | None) -> str:
    """Owner id for per-user routes. Gateway id, else internal-door LAN admin, else 401."""
    if user_id_header:
        return user_id_header
    if _lan_is_admin(request):
        return INTERNAL_ADMIN_ID
    raise HTTPException(status_code=401, detail="gateway user identity required")


def resolve_admin(
    request: Request,
    user_id_header: str | None,
    tier_header: str | None,
) -> str:
    """Owner id for admin-only routes; raises 403/401 for non-admins.

    Gateway path: identity present → tier must be ``admin`` (gateway already gated,
    this is defense in depth). Direct LAN path: trusted internal-admin.
    """
    if user_id_header:
        if tier_header == "admin":
            return user_id_header
        raise HTTPException(status_code=403, detail="admin only")
    if _lan_is_admin(request):
        return INTERNAL_ADMIN_ID
    raise HTTPException(status_code=401, detail="admin authentication required")
