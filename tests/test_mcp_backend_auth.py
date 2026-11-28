"""RetrievalClient sk-key wiring: when api_key is set, every request carries
`Authorization: Bearer <key>`; when unset, no auth header (direct-to-4090)."""
import asyncio

import httpx

from mcp_server.backend import RetrievalClient


def _capture():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True})

    return seen, httpx.MockTransport(handler)


def test_api_key_sends_bearer():
    seen, transport = _capture()
    client = RetrievalClient(client=httpx.AsyncClient(transport=transport, base_url="http://gw"),
                             api_key="sk-user-123")
    asyncio.run(client.list_conferences())
    assert seen["auth"] == "Bearer sk-user-123"


def test_no_api_key_no_auth_header():
    seen, transport = _capture()
    client = RetrievalClient(client=httpx.AsyncClient(transport=transport, base_url="http://compute"))
    asyncio.run(client.list_conferences())
    assert seen["auth"] is None
