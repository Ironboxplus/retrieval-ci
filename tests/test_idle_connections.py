"""A peer that connects and never sends a request must not hold a socket forever.

Real uvicorn, real TCP sockets. On 2026-10-07 easytier-core on .111 left 896 such
connections through the :8000 SSH forward; the backend reached its 1024-fd limit and
answered nothing for an hour, because uvicorn arms its keep-alive timer only after a
response.
"""

from __future__ import annotations

import socket
import threading
import time

from fastapi import FastAPI
import pytest
import uvicorn

from app.api import idle_connections


def _serve(http="auto"):
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, http=http,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    return server, thread, port


def _response(sock: socket.socket) -> bytes:
    """Read one small response (headers and body may arrive in separate segments)."""
    sock.settimeout(5)
    data = b""
    while b'{"ok":true}' not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def _closed_within(sock: socket.socket, seconds: float) -> bool:
    sock.settimeout(seconds)
    try:
        return sock.recv(1) == b""
    except socket.timeout:
        return False


@pytest.fixture
def short_first_request_timeout(monkeypatch):
    monkeypatch.setattr(idle_connections, "FIRST_REQUEST_TIMEOUT_SECONDS", 0.5)


def test_plain_uvicorn_keeps_a_silent_connection_open():
    """The defect being fixed, so a uvicorn upgrade that fixes it upstream shows up here."""
    server, thread, port = _serve()
    try:
        with socket.create_connection(("127.0.0.1", port)) as silent:
            assert not _closed_within(silent, 1.5)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_silent_connection_is_closed_after_the_first_request_timeout(short_first_request_timeout):
    server, thread, port = _serve(idle_connections.http_protocol_class())
    try:
        with socket.create_connection(("127.0.0.1", port)) as silent:
            assert _closed_within(silent, 3.0)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_a_request_inside_the_window_is_served_and_kept_alive(short_first_request_timeout):
    server, thread, port = _serve(idle_connections.http_protocol_class())
    try:
        with socket.create_connection(("127.0.0.1", port)) as client:
            time.sleep(0.2)
            client.sendall(b"GET /ping HTTP/1.1\r\nHost: x\r\n\r\n")
            reply = _response(client)
            assert reply.startswith(b"HTTP/1.1 200") and b'{"ok":true}' in reply
            # past the first-request window: uvicorn's own keep-alive (5 s) now governs
            time.sleep(1.0)
            client.sendall(b"GET /ping HTTP/1.1\r\nHost: x\r\n\r\n")
            assert _response(client).startswith(b"HTTP/1.1 200")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
