"""zstd (and gzip) on the wire, in both directions.

Measured on a real ``index_code`` payload for this repository (650 files, 8,848
chunks, 19.3 MiB of JSON):

    gzip -6    2,714,299 B    7.5x    447 ms
    zstd -1    2,600,437 B    7.8x     46 ms
    zstd -3    2,233,493 B    9.1x     54 ms
    zstd -9    1,882,886 B   10.8x    226 ms

zstd -3 is the operating point: a better ratio than gzip -6 at an eighth of the
CPU. What it buys is transfer time over the LAN overlay and the public tunnel.

**Compatibility is the constraint, not the feature.** Every already-published
client (0.4.8 and earlier) POSTs plain JSON and sends whatever ``Accept-Encoding``
httpx defaults to. So:

  * a request with no ``Content-Encoding`` is passed through untouched — the
    middleware is not even in the path;
  * a response is only compressed with an encoding the client actually listed, so
    a gzip-only client keeps getting gzip and a client that lists neither keeps
    getting identity bytes.

Nothing here is negotiated out-of-band and nothing requires a client upgrade.

What compression does NOT buy is server memory: a compressed body still has to be
expanded before it can be parsed, so the expanded size is what the process holds
either way. That is why the decompressor below is streaming — it feeds the app
incrementally rather than materialising a second full copy of the expanded body
on top of the one Starlette already keeps. The memory fix for the 2026-09-04 OOM
is the streaming indexer, not this.
"""
from __future__ import annotations

import json
import logging
import zlib

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipResponder, IdentityResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

try:  # optional: a deploy without the wheel degrades to gzip, it does not 500
    import zstandard
except ImportError:  # pragma: no cover - exercised only on an incomplete install
    zstandard = None
    logger.warning("zstandard is not installed; zstd request bodies will be "
                   "refused with 415 and responses will fall back to gzip")

# Headers that describe the *encoded* body and become lies once it is decoded.
_BODY_ENCODING_HEADERS = (b"content-encoding", b"content-length")


class _BadlyCompressedBody(Exception):
    """A client sent a body that does not decode. That is a 400, not a 500."""

    def __init__(self, encoding: str, cause: Exception | str) -> None:
        super().__init__(f"could not decode the {encoding} request body: {cause}")


class _Decoder:
    """Incremental decoder with a uniform surface over zlib and zstandard."""

    __slots__ = ("_obj", "_errors", "encoding")

    def __init__(self, encoding: str, obj, errors: tuple[type[BaseException], ...]) -> None:
        self.encoding = encoding
        self._obj = obj
        self._errors = errors

    def feed(self, chunk: bytes) -> bytes:
        try:
            return self._obj.decompress(chunk)
        except self._errors as exc:
            raise _BadlyCompressedBody(self.encoding, exc) from exc

    def finish(self) -> bytes:
        try:
            tail = self._obj.flush()
        except self._errors as exc:
            raise _BadlyCompressedBody(self.encoding, exc) from exc
        if not self._obj.eof:
            # A truncated upload decodes into valid-looking partial JSON, which the
            # app would report as a schema error against a body the client never
            # sent. Name the real cause instead.
            raise _BadlyCompressedBody(
                self.encoding, "the stream ended in the middle of a frame "
                               "(truncated or interrupted upload)")
        return tail


def _decoder_for(encoding: str) -> _Decoder | None:
    if encoding == "zstd":
        if zstandard is None:
            return None
        return _Decoder(encoding, zstandard.ZstdDecompressor().decompressobj(),
                        (zstandard.ZstdError, ValueError))
    if encoding == "gzip":
        return _Decoder(encoding, zlib.decompressobj(16 + zlib.MAX_WBITS), (zlib.error,))
    if encoding == "deflate":
        return _Decoder(encoding, zlib.decompressobj(), (zlib.error,))
    return None


def supported_request_encodings() -> tuple[str, ...]:
    return ("zstd", "gzip", "deflate") if zstandard is not None else ("gzip", "deflate")


def _accepts(accept_encoding: str, token: str) -> bool:
    """Token-aware Accept-Encoding test, so ``zstd;q=0`` is not read as a yes."""
    for part in accept_encoding.split(","):
        name, _, params = part.strip().partition(";")
        if name.strip().lower() != token:
            continue
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    return float(value.strip()) > 0
                except ValueError:
                    return True
        return True
    return False


def _advertising_send(send: Send) -> Send:
    """Wrap `send` so every response carries `Accept-Encoding`.

    RFC 7694 defines Accept-Encoding as a RESPONSE header meaning "I accept these
    content codings in requests". It is how a client learns, from any ordinary
    reply, that it may compress the next upload — without a capability endpoint,
    a version handshake, or a wasted probe request. A backend without this
    middleware simply never sends it, and the client keeps sending plain JSON.
    """
    advertised = ", ".join(supported_request_encodings()).encode()

    async def send_advertising(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers") or [])
            if not any(key.lower() == b"accept-encoding" for key, _ in headers):
                headers.append((b"accept-encoding", advertised))
                message = {**message, "headers": headers}
        await send(message)

    return send_advertising


async def _send_error(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


class CompressedRequestMiddleware:
    """Decode ``Content-Encoding`` request bodies as they arrive.

    Pure ASGI rather than ``BaseHTTPMiddleware`` on purpose: the latter reads the
    whole body into the middleware before the app sees any of it, which would put a
    second full copy of a large upload in memory.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        send = _advertising_send(send)
        encoding = Headers(scope=scope).get("content-encoding", "").strip().lower()
        if not encoding or encoding == "identity":
            # The path every published client takes. Not one byte is touched.
            await self.app(scope, receive, send)
            return

        decoder = _decoder_for(encoding)
        if decoder is None:
            await _send_error(
                send, 415,
                f"unsupported Content-Encoding {encoding!r}; this server accepts "
                f"{', '.join(supported_request_encodings())}, or an uncompressed body")
            return

        # Both headers describe the encoded bytes; downstream would use them to
        # truncate the decoded body. Starlette reads the stream to completion and
        # does not require Content-Length.
        decoded_scope = dict(scope)
        decoded_scope["headers"] = [
            (key, value) for key, value in scope["headers"]
            if key.lower() not in _BODY_ENCODING_HEADERS
        ]

        failure: list[_BadlyCompressedBody] = []

        async def receive_decoded() -> Message:
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    return message
                chunk = message.get("body", b"")
                more_body = message.get("more_body", False)
                try:
                    decoded = decoder.feed(chunk) if chunk else b""
                    if not more_body:
                        decoded += decoder.finish()
                except _BadlyCompressedBody as exc:
                    # Recorded as well as raised: FastAPI catches everything thrown
                    # while it reads the body and re-raises it as a flat 400 "There
                    # was an error parsing the body", so the exception alone never
                    # reaches the handler below.
                    failure.append(exc)
                    raise
                if decoded or not more_body:
                    return {"type": "http.request", "body": decoded,
                            "more_body": more_body}
                # This input chunk did not complete a block. Pull the next one
                # rather than handing the app an empty message per frame.

        started = False
        replaced = False

        async def send_filtered(message: Message) -> None:
            nonlocal started, replaced
            if replaced:
                return  # the app's own error response; ours already went out
            if failure and not started:
                # Say which upload failed and how, instead of FastAPI's generic
                # parse error — a truncated 19 MiB body is not a schema problem.
                replaced = True
                logger.warning("rejecting %s %s: %s", scope.get("method"),
                               scope.get("path"), failure[0])
                await _send_error(send, 400, str(failure[0]))
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(decoded_scope, receive_decoded, send_filtered)
        except _BadlyCompressedBody as exc:
            # Reached only when the app lets the exception through (i.e. not FastAPI).
            if started or replaced:
                # Headers are already on the wire; a second start would be a
                # protocol violation. Let the server abort the connection.
                raise
            logger.warning("rejecting %s %s: %s", scope.get("method"),
                           scope.get("path"), exc)
            await _send_error(send, 400, str(exc))


class _ZstdResponder(IdentityResponder):
    """zstd counterpart of Starlette's GZipResponder.

    Subclassed rather than reimplemented so the Content-Length/Vary/streaming
    bookkeeping stays byte-for-byte identical to the gzip path that already ships.
    """

    content_encoding = "zstd"

    def __init__(self, app: ASGIApp, minimum_size: int, level: int = 3) -> None:
        super().__init__(app, minimum_size)
        self._zstd = zstandard.ZstdCompressor(level=level)
        self._compressor = self._zstd.compressobj()
        self._emitted_header = False

    def apply_compression(self, body: bytes, *, more_body: bool) -> bytes:
        if not self._emitted_header and not more_body:
            # The whole response in one message — every JSON API reply here. One-shot
            # compression writes the decompressed size into the frame header, which
            # a streaming compressobj cannot, so simple one-shot client decoders
            # (`ZstdDecompressor.decompress`) work as well as streaming ones.
            self._emitted_header = True
            return self._zstd.compress(body)
        out = self._compressor.compress(body)
        if not more_body:
            return out + self._compressor.flush(zstandard.COMPRESSOBJ_FLUSH_FINISH)
        if not self._emitted_header:
            self._emitted_header = True
            # IdentityResponder decides the response is compressed by comparing the
            # first chunk against the original. zstd buffers, so a small first chunk
            # would compress to b"" — indistinguishable from "not compressed" — and
            # the remaining chunks would then go out compressed with no
            # Content-Encoding set. Flushing a block forces the frame header out.
            out += self._compressor.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK)
        return out


class CompressedResponseMiddleware:
    """Compress responses with the best encoding the client actually asked for.

    zstd is preferred over gzip when both are offered; a client that offers neither
    (every published 0.4.x MCP client without the ``zstandard`` extra) is served
    identity bytes exactly as before.
    """

    def __init__(self, app: ASGIApp, minimum_size: int = 500,
                 zstd_level: int = 3, gzip_level: int = 6) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.zstd_level = zstd_level
        self.gzip_level = gzip_level

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        accept = Headers(scope=scope).get("Accept-Encoding", "")
        responder: ASGIApp
        if zstandard is not None and _accepts(accept, "zstd"):
            responder = _ZstdResponder(self.app, self.minimum_size, self.zstd_level)
        elif _accepts(accept, "gzip"):
            responder = GZipResponder(self.app, self.minimum_size,
                                      compresslevel=self.gzip_level)
        else:
            responder = IdentityResponder(self.app, self.minimum_size)
        await responder(scope, receive, send)
