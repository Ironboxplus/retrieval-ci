"""Close connections that never send a request.

uvicorn arms its keep-alive timer only after a response, so a peer that connects
and sends nothing holds a socket (and a file descriptor) forever. On 2026-10-07
easytier-core on .111 left 896 such connections through the :8000 SSH forward;
the backend reached its 1024-fd limit and answered nothing for an hour.

The protocol below arms uvicorn's own keep-alive timer at connect time instead.
The first received byte cancels it exactly as before (`data_received` calls
`_unset_keepalive_if_required`), so only connections that stay silent close.
"""

from __future__ import annotations

FIRST_REQUEST_TIMEOUT_SECONDS = 30.0


def _with_first_request_timeout(base: type) -> type:
    class FirstRequestTimeoutProtocol(base):
        def connection_made(self, transport) -> None:  # type: ignore[override]
            super().connection_made(transport)
            self.timeout_keep_alive_task = self.loop.call_later(
                FIRST_REQUEST_TIMEOUT_SECONDS, self.timeout_keep_alive_handler)

    FirstRequestTimeoutProtocol.__name__ = f"FirstRequestTimeout{base.__name__}"
    return FirstRequestTimeoutProtocol


def http_protocol_class() -> type:
    """uvicorn's own choice (httptools when installed, else h11) with the timeout."""

    try:
        from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
    except ImportError:
        from uvicorn.protocols.http.h11_impl import H11Protocol

        return _with_first_request_timeout(H11Protocol)
    return _with_first_request_timeout(HttpToolsProtocol)
