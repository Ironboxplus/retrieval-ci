"""Pure HTTP client for the retrival FastAPI backend.

This module has NO dependency on the `mcp` package — it is the testable core of
the MCP server. It calls the already-running backend (default http://127.0.0.1:8000)
so the MCP process reuses the warm BGE-M3 model and never loads a second copy on
the GPU. Everything here is plain async + httpx, exercised by tests/test_mcp_server.py
via httpx.MockTransport.

Formatters return compact, agent-friendly text (title + paper_id + venue/year +
score + snippet) rather than the full payload — token-efficient citations in the
spirit of Microsoft FastContext / Augment's "return the slice, not the dump".
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union
from urllib.parse import quote

import httpx

logger = logging.getLogger("mcp_server.client")

# Search/rerank can take tens of seconds on large match-doc requests (rerank ~60s
# for big paper_id inputs, see W102). Layered timeouts: connect/pool fail fast so a
# stuck proxy cannot pin a tool call for the full bulk-read window; read/write stay
# generous for paper artifacts and bulk uploads (hang forensics 2026-07-23).
DEFAULT_TIMEOUT_CONNECT_S = 5.0
DEFAULT_TIMEOUT_POOL_S = 5.0
DEFAULT_TIMEOUT_READ_S = 180.0
DEFAULT_TIMEOUT_WRITE_S = 180.0
DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_READ_S  # legacy float alias


def make_http_timeout(
    *,
    connect: float = DEFAULT_TIMEOUT_CONNECT_S,
    read: float = DEFAULT_TIMEOUT_READ_S,
    write: float = DEFAULT_TIMEOUT_WRITE_S,
    pool: float = DEFAULT_TIMEOUT_POOL_S,
) -> httpx.Timeout:
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


def _coerce_timeout(timeout: float | httpx.Timeout) -> httpx.Timeout:
    if isinstance(timeout, httpx.Timeout):
        return timeout
    value = float(timeout)
    # Preserve custom bulk-read budgets while still failing connect/pool fast.
    return make_http_timeout(read=value, write=value)


_PAPER_ARTIFACT_ATTEMPTS = 3
_PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0.5
_PAPER_ARTIFACT_RANGE_BYTES = 4 * 1024 * 1024
_PAPER_STREAM_CHUNK_BYTES = 64 * 1024
_PAPER_BATCH_CONCURRENCY = 2
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)


def _paper_artifact_filename(paper_id: str, suffix: str) -> str:
    """Turn a canonical paper ID into a portable local artifact filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("._") or "paper"
    return f"{stem}{suffix}"


def _paper_artifact_destination(
    paper_id: str,
    suffix: str,
    save_path: str | None,
    *,
    overwrite: bool,
) -> Path:
    """Resolve a caller-local file/directory target and enforce overwrite policy."""
    raw_path = save_path or "."
    expanded = Path(raw_path).expanduser()
    path_text = str(raw_path)
    directory_target = (
        save_path is None
        or expanded.is_dir()
        or path_text.endswith(("/", "\\"))
        or (not expanded.exists() and not expanded.suffix)
    )
    destination = expanded / _paper_artifact_filename(paper_id, suffix) \
        if directory_target else expanded
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass overwrite=True to replace it."
        )
    return destination


def _commit_temporary_file(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically publish a complete sibling file with optional replacement."""
    if overwrite:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass overwrite=True to replace it."
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Filesystem cannot guarantee overwrite=False for destination {destination} "
            "because atomic hard-link publication is unavailable. Choose a local "
            "NTFS/ext4-style path or pass overwrite=True explicitly."
        ) from exc


def _retryable_paper_artifact_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and 500 <= exc.response.status_code < 600
    )


def _paper_artifact_validator(headers) -> tuple[str, str] | None:
    """Prefer a strong ETag, falling back to Last-Modified for If-Range."""
    etag = headers.get("etag")
    if etag and not etag.startswith("W/"):
        return "etag", etag
    last_modified = headers.get("last-modified")
    if last_modified:
        return "last-modified", last_modified
    return None


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    match = _CONTENT_RANGE_PATTERN.fullmatch((value or "").strip())
    if match is None:
        raise ValueError(f"Invalid PDF Content-Range header: {value!r}")
    start, end, total = map(int, match.groups())
    if end < start or total <= end:
        raise ValueError(f"Invalid PDF Content-Range bounds: {value!r}")
    return start, end, total


def _truncate_temporary_file(path: Path, size: int) -> None:
    with path.open("r+b") as output:
        output.truncate(size)


async def _append_stream(response: httpx.Response, destination: Path) -> int:
    written = 0
    with destination.open("ab") as output:
        async for chunk in response.aiter_bytes(_PAPER_STREAM_CHUNK_BYTES):
            if not chunk:
                continue
            output.write(chunk)
            written += len(chunk)
    return written


async def _stream_ranged_artifact(
    open_stream,
    temporary: Path,
    *,
    label: str,
    range_bytes: int,
    signature: bytes | None,
    signature_error: str,
) -> int:
    """Download one artifact with validated byte ranges and per-range retries."""
    temporary.write_bytes(b"")
    offset = 0
    total_size: int | None = None
    validator: tuple[str, str] | None = None
    while total_size is None or offset < total_size:
        requested_end = offset + range_bytes - 1
        if total_size is not None:
            requested_end = min(requested_end, total_size - 1)
        checkpoint = offset
        received_full_response = False

        for attempt in range(1, _PAPER_ARTIFACT_ATTEMPTS + 1):
            ranged_response = False
            declared_end: int | None = None
            try:
                async with open_stream(
                    f"bytes={offset}-{requested_end}",
                    validator[1] if validator else None,
                ) as response:
                    response.raise_for_status()
                    response_validator = _paper_artifact_validator(response.headers)
                    if validator is None:
                        validator = response_validator
                    elif response_validator != validator:
                        raise ValueError(
                            f"{label} changed during ranged download: expected "
                            f"{validator[0]} {validator[1]!r}, received "
                            f"{response_validator!r}."
                        )
                    if response.status_code == 200:
                        if offset != 0:
                            raise ValueError(
                                f"Server stopped honoring ranges for {label} after a "
                                "partial transfer."
                            )
                        content_length = response.headers.get("content-length")
                        expected = int(content_length) if content_length else None
                        written = await _append_stream(response, temporary)
                        if expected is not None and written != expected:
                            raise httpx.RemoteProtocolError(
                                f"Incomplete {label} response: expected {expected} bytes, "
                                f"received {written}."
                            )
                        offset = written
                        total_size = written
                        received_full_response = True
                    elif response.status_code == 206:
                        ranged_response = True
                        start, end, response_total = _parse_content_range(
                            response.headers.get("content-range"))
                        declared_end = end
                        if start != offset or end > requested_end:
                            raise ValueError(
                                f"{label} Content-Range does not match the requested byte range: "
                                f"requested bytes={offset}-{requested_end}, received "
                                f"bytes={start}-{end}/{response_total}."
                            )
                        if total_size is not None and response_total != total_size:
                            raise ValueError(
                                f"{label} size changed during ranged download: "
                                f"expected {total_size}, received {response_total}."
                            )
                        total_size = response_total
                        expected = end - start + 1
                        written = await _append_stream(response, temporary)
                        if written != expected:
                            raise httpx.RemoteProtocolError(
                                f"Incomplete {label} range: expected {expected} bytes, "
                                f"received {written}."
                            )
                        offset = end + 1
                    else:
                        raise ValueError(
                            f"Unexpected {label} response status {response.status_code}; "
                            "expected 200 or 206."
                        )
                break
            except httpx.HTTPError as exc:
                partial_size = temporary.stat().st_size
                valid_partial = (
                    ranged_response
                    and declared_end is not None
                    and checkpoint < partial_size <= declared_end + 1
                    and (total_size is None or partial_size <= total_size)
                )
                if valid_partial:
                    offset = partial_size
                    logger.warning(
                        "Artifact transfer interrupted for %s range=%d-%d after %d "
                        "new byte(s): %s; resuming at byte %d",
                        label, checkpoint, requested_end,
                        partial_size - checkpoint, exc, offset,
                    )
                    break
                _truncate_temporary_file(temporary, checkpoint)
                offset = checkpoint
                if not _retryable_paper_artifact_error(exc) \
                        or attempt == _PAPER_ARTIFACT_ATTEMPTS:
                    raise
                delay = _PAPER_ARTIFACT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Artifact transfer interrupted for %s range=%d-%d "
                    "(attempt %d/%d): %s; retrying in %.1fs",
                    label, checkpoint, requested_end, attempt,
                    _PAPER_ARTIFACT_ATTEMPTS, exc, delay,
                )
                await asyncio.sleep(delay)
            except BaseException:
                _truncate_temporary_file(temporary, checkpoint)
                raise
        else:
            raise RuntimeError(f"{label} range transfer exhausted without a result")

        if received_full_response:
            break

    if total_size is None or offset != total_size:
        raise ValueError(
            f"Incomplete {label} transfer: {offset}/{total_size} bytes."
        )
    with temporary.open("rb") as downloaded:
        if signature is not None and downloaded.read(len(signature)) != signature:
            raise ValueError(signature_error)
    return offset


async def _stream_pdf_ranges(open_stream, temporary: Path, paper_id: str) -> int:
    """Download a PDF with validated byte ranges, retrying only the failed range."""
    return await _stream_ranged_artifact(
        open_stream,
        temporary,
        label=f"PDF for paper '{paper_id}'",
        range_bytes=_PAPER_ARTIFACT_RANGE_BYTES,
        signature=b"%PDF-",
        signature_error=(
            f"Response for paper '{paper_id}' is not a PDF (missing %PDF- header)."
        ),
    )


async def _stream_markdown_ranges(open_stream, temporary: Path, paper_id: str) -> int:
    """Stream an extracted Markdown file without buffering it into JSON."""
    return await _stream_ranged_artifact(
        open_stream,
        temporary,
        label=f"Markdown for paper '{paper_id}'",
        range_bytes=_PAPER_ARTIFACT_RANGE_BYTES,
        signature=None,
        signature_error="",
    )


def _validate_paper_batch(
    paper_ids: list[str],
    artifacts: str,
) -> tuple[list[str], str]:
    if not 1 <= len(paper_ids) <= 10:
        raise ValueError("paper_ids must contain between 1 and at most 10 papers")
    normalized_ids = [str(paper_id).strip() for paper_id in paper_ids]
    if any(not paper_id for paper_id in normalized_ids):
        raise ValueError("paper_ids cannot contain blank values")
    normalized_artifacts = artifacts.strip().lower()
    if normalized_artifacts not in {"pdf", "markdown", "both"}:
        raise ValueError("artifacts must be one of: pdf, markdown, both")
    return list(dict.fromkeys(normalized_ids)), normalized_artifacts


def _paper_batch_destination(save_dir: str) -> Path:
    destination = Path(save_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Batch destination is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _unique_paper_artifact_filename(
    paper_id: str,
    suffix: str,
    used_names: set[str],
) -> str:
    candidate = _paper_artifact_filename(paper_id, suffix)
    if candidate in used_names:
        stem = candidate[:-len(suffix)] if suffix else candidate
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:12]
        candidate = f"{stem}-{digest}{suffix}"
    if candidate in used_names:
        raise ValueError(
            f"Paper IDs cannot be mapped to unique local filenames: {paper_id!r}"
        )
    used_names.add(candidate)
    return candidate


def _paper_batch_artifact_plan(
    paper_ids: list[str],
    artifacts: str,
    destination_dir: Path,
    *,
    overwrite: bool,
) -> list[tuple[str, str, Path]]:
    artifact_kinds = ("pdf", "markdown") if artifacts == "both" else (artifacts,)
    used_names: set[str] = set()
    plan: list[tuple[str, str, Path]] = []
    for paper_id in paper_ids:
        for artifact_kind in artifact_kinds:
            suffix = ".pdf" if artifact_kind == "pdf" else ".md"
            filename = _unique_paper_artifact_filename(
                paper_id, suffix, used_names)
            plan.append((paper_id, artifact_kind, destination_dir / filename))
    if not overwrite:
        existing = [destination for _, _, destination in plan if destination.exists()]
        if existing:
            raise FileExistsError(
                f"Destination already exists: {existing[0]}. "
                "Pass overwrite=True to replace existing artifacts."
            )
    return plan


def format_paper_batch_download(downloaded: dict[str, Any]) -> str:
    errors = downloaded.get("errors") or []
    summary = (
        f"Downloaded {len(downloaded.get('paper_ids') or [])} paper(s): "
        f"{len(downloaded.get('paths') or [])} artifact(s) saved to "
        f"'{downloaded.get('save_dir')}', {len(errors)} error(s)."
    )
    if not errors:
        return summary
    details = "\n".join(
        f"- {error.get('paper_id')} [{error.get('artifact')}]: {error.get('error')}"
        for error in errors
    )
    return f"{summary}\nFailures:\n{details}"


try:  # optional: without the wheel this client simply never compresses
    import zstandard as _zstd
except ImportError:  # pragma: no cover - only on an incomplete install
    _zstd = None

# Only index_code/index_dir bodies get large; below this a round trip is
# latency-bound and compressing costs more than the bytes it saves.
_ZSTD_MIN_BYTES = 64 * 1024
_ZSTD_LEVEL = 3


def _response_detail(r: httpx.Response) -> str:
    """The reason a FastAPI backend gave for an error status, or '' if none."""
    try:
        detail = r.json().get("detail")
    except (ValueError, AttributeError):
        return r.text.strip()[:500]
    if isinstance(detail, list):  # request validation: one line per bad field
        return "; ".join(
            ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            + f": {item.get('msg', '')}"
            for item in detail if isinstance(item, dict)
        )[:500]
    return str(detail)[:500] if detail else ""


def _raise_for_status(r: httpx.Response) -> None:
    """Mirror of `mcp_plugin.retrieval_mcp._raise_for_status`: the backend's reason
    goes into the message; type, request and response stay those of httpx."""
    if r.is_success:
        return
    detail = _response_detail(r)
    if not detail:
        r.raise_for_status()
    raise httpx.HTTPStatusError(
        f"HTTP {r.status_code} from {r.request.method} {r.request.url.path}: {detail}",
        request=r.request, response=r)


def _compress_body(payload: dict) -> bytes | None:
    """The zstd-framed body, or None when plain JSON is the right thing to send."""
    if _zstd is None:
        return None
    raw = json.dumps(payload).encode()
    if len(raw) < _ZSTD_MIN_BYTES:
        return None
    return _zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(raw)


class RetrievalClient:
    """Thin async wrapper over the retrival REST API.

    Pass ``client`` to inject an httpx.AsyncClient (used by tests with
    MockTransport). Otherwise one is built lazily against ``base_url``.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 client: Optional[httpx.AsyncClient] = None,
                 timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
                 api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = _coerce_timeout(timeout)
        # When the MCP server points at the paid gateway (not the 4090 directly),
        # `api_key` is the user's sk-key and is sent as a Bearer token on every
        # request. Sent per-request so it applies to injected clients too. None =>
        # talking straight to the compute box (no auth), the original behavior.
        self._auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # None = not yet known; set from the backend's `Accept-Encoding` response
        # header (RFC 7694) or from the first rejected compressed body.
        self._server_accepts_zstd: bool | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _send_json(self, method: str, path: str, payload: dict) -> httpx.Response:
        """Mirror of `mcp_plugin.retrieval_mcp._send_json` — same policy, same reasons.

        httpx compresses nothing on the request side (it only decodes responses), so
        the ~19 MiB `index_code` body is encoded here. A backend that predates
        request decompression answers 400/415/422 before the route function runs, so
        the uncompressed resend cannot duplicate a side effect; after that this
        client stops compressing for the rest of its life.
        """
        client = self._http()
        body = _compress_body(payload) if self._server_accepts_zstd is not False else None
        if body is None:
            return await client.request(method, path, json=payload,
                                        headers=self._auth_headers)
        resp = await client.request(
            method, path, content=body,
            headers={**self._auth_headers, "Content-Type": "application/json",
                     "Content-Encoding": "zstd"})
        if "zstd" in (resp.headers.get("accept-encoding") or "").lower():
            self._server_accepts_zstd = True
        elif resp.status_code in (400, 415, 422):
            self._server_accepts_zstd = False
            return await client.request(method, path, json=payload,
                                        headers=self._auth_headers)
        return resp

    async def _post(self, path: str, payload: dict) -> dict:
        logger.info("POST %s keys=%s", path, sorted(payload.keys()))
        resp = await self._send_json("POST", path, payload)
        _raise_for_status(resp)
        return resp.json()

    async def _put(self, path: str, payload: dict) -> dict:
        logger.info("PUT %s keys=%s", path, sorted(payload.keys()))
        resp = await self._send_json("PUT", path, payload)
        _raise_for_status(resp)
        return resp.json()

    async def _get(
        self,
        path: str,
        params: dict | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict:
        logger.info("GET %s", path)
        request_headers = {**self._auth_headers, **(headers or {})}
        resp = await self._http().get(
            path, params=params or {}, headers=request_headers)
        _raise_for_status(resp)
        return resp.json()

    async def _delete(self, path: str, params: dict | None = None) -> dict:
        logger.info("DELETE %s", path)
        resp = await self._http().request(
            "DELETE", path, params=params or {}, headers=self._auth_headers)
        _raise_for_status(resp)
        return resp.json()

    # ---- retrieval over the shared academic corpus ----

    async def search_papers(
        self,
        query: str,
        top_k: int = 20,
        venue: Optional[Union[str, list[str]]] = None,
        year: Optional[Union[int, list[int]]] = None,
        venue_tier: Optional[str] = None,
        title_only: bool = False,
        dedup_by_paper: bool = True,
        method: str = "hybrid",
        paper_ids: Optional[list[str]] = None,
        source_type: Optional[str] = None,
    ) -> dict:
        """Hybrid (BGE-M3 dense+sparse RRF + cross-encoder rerank) search over the
        whole corpus. Results come back grouped by paper, ranked by best chunk."""
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "method": method,
            "title_only": title_only,
            "dedup_by_paper": dedup_by_paper,
        }
        if venue is not None:
            payload["venue"] = venue
        if year is not None:
            payload["year"] = year
        if venue_tier is not None:
            payload["venue_tier"] = venue_tier
        if paper_ids:
            payload["paper_ids"] = paper_ids
        if source_type:
            payload["source_type"] = source_type
        return await self._post("/api/search", payload)

    async def search_within_paper(self, paper_id: str, query: str, top_k: int = 50) -> dict:
        """Search only inside one paper, expanding every matched chunk (not just the
        best per paper) so the caller sees how each part of the paper matches.

        Accepts a `doc:<n>` id for uploads/webpages, which have no paper_id; that
        routes to the doc_ids filter instead. Sending both would AND into the
        empty set, so exactly one goes out."""
        scope = _scope_filter(paper_id)
        payload = {"query": query, "top_k": top_k, "method": "hybrid",
                   "dedup_by_paper": False, **scope}
        return await self._post("/api/search", payload)

    async def document_markdown_size(self, document_id: str) -> int | None:
        """Total Markdown bytes for a document, via a 1-byte Range probe.

        The route registers GET only, so HEAD answers 405; a `bytes=0-0` GET
        returns `Content-Range: bytes 0-0/<total>` and costs one byte instead of
        moving a whole book to find out how big it is. None when unreadable.
        """
        try:
            response = await self._http().get(
                f"/api/documents/{quote(document_id, safe='')}/markdown",
                headers={**self._auth_headers,
                         "Range": "bytes=0-0", "Accept-Encoding": "identity"},
            )
        except httpx.HTTPError:
            return None
        match = _CONTENT_RANGE_PATTERN.match(
            response.headers.get("Content-Range", ""))
        return int(match.group(3)) if match else None

    async def get_paper_markdown(self, paper_id: str) -> dict:
        """Fetch one paper's complete extracted Markdown without retrieval truncation."""
        if _is_document_id(paper_id):
            path = f"/api/documents/{quote(_document_id(paper_id), safe='')}/fulltext"
        else:
            path = f"/api/papers/{quote(paper_id, safe='')}/fulltext"
        idempotency_key = f"paper-fulltext-{uuid.uuid4().hex}"
        for attempt in range(1, _PAPER_ARTIFACT_ATTEMPTS + 1):
            try:
                return await self._get(
                    path, headers={"X-Idempotency-Key": idempotency_key})
            except httpx.HTTPError as exc:
                if not _retryable_paper_artifact_error(exc) \
                        or attempt == _PAPER_ARTIFACT_ATTEMPTS:
                    raise
                delay = _PAPER_ARTIFACT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Markdown transfer interrupted for paper=%s (attempt %d/%d): "
                    "%s; retrying in %.1fs",
                    paper_id, attempt, _PAPER_ARTIFACT_ATTEMPTS, exc, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Markdown transfer exhausted without a result")

    async def save_paper_markdown(
        self,
        paper_id: str,
        save_path: str | None = None,
        *,
        overwrite: bool = False,
    ) -> dict:
        """Stream complete Markdown and save it atomically on the MCP host."""
        destination = _paper_artifact_destination(
            paper_id, ".md", save_path, overwrite=overwrite)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        if _is_document_id(paper_id):
            markdown_path = f"/api/documents/{quote(_document_id(paper_id), safe='')}/markdown"
        else:
            markdown_path = f"/api/papers/{quote(paper_id, safe='')}/markdown"
        idempotency_key = f"paper-markdown-{uuid.uuid4().hex}"

        def open_stream(range_header: str, if_range: str | None):
            headers = {
                **self._auth_headers,
                "Accept-Encoding": "identity",
                "Range": range_header,
                "X-Idempotency-Key": idempotency_key,
            }
            if if_range:
                headers["If-Range"] = if_range
            return self._http().stream("GET", markdown_path, headers=headers)

        try:
            size = await _stream_markdown_ranges(open_stream, temporary, paper_id)
            _commit_temporary_file(temporary, destination, overwrite=overwrite)
            return {"path": str(destination), "size": size}
        finally:
            temporary.unlink(missing_ok=True)

    async def download_paper_pdf(
        self,
        paper_id: str,
        save_path: str | None = None,
        *,
        overwrite: bool = False,
    ) -> dict:
        """Download an original paper PDF with validated HTTP ranges."""
        destination = _paper_artifact_destination(
            paper_id, ".pdf", save_path, overwrite=overwrite)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        encoded_paper_id = quote(paper_id, safe="")

        def open_stream(range_header: str, if_range: str | None):
            headers = {
                **self._auth_headers,
                "Accept-Encoding": "identity",
                "Range": range_header,
            }
            if if_range:
                headers["If-Range"] = if_range
            return self._http().stream(
                "GET", f"/api/papers/{encoded_paper_id}/pdf", headers=headers)

        try:
            size = await _stream_pdf_ranges(open_stream, temporary, paper_id)
            _commit_temporary_file(temporary, destination, overwrite=overwrite)
            return {"path": str(destination), "size": size}
        finally:
            temporary.unlink(missing_ok=True)

    async def download_papers(
        self,
        paper_ids: list[str],
        save_dir: str = ".",
        *,
        artifacts: str = "pdf",
        overwrite: bool = False,
    ) -> dict:
        """Download up to ten papers through resumable per-artifact transfers."""
        normalized_ids, normalized_artifacts = _validate_paper_batch(
            paper_ids, artifacts)
        destination_dir = _paper_batch_destination(save_dir)
        plan = _paper_batch_artifact_plan(
            normalized_ids,
            normalized_artifacts,
            destination_dir,
            overwrite=overwrite,
        )
        semaphore = asyncio.Semaphore(_PAPER_BATCH_CONCURRENCY)

        async def download_one(
            paper_id: str,
            artifact_kind: str,
            destination: Path,
        ) -> tuple[str | None, dict[str, str] | None]:
            async with semaphore:
                try:
                    if artifact_kind == "pdf":
                        saved = await self.download_paper_pdf(
                            paper_id,
                            save_path=str(destination),
                            overwrite=overwrite,
                        )
                    else:
                        saved = await self.save_paper_markdown(
                            paper_id,
                            save_path=str(destination),
                            overwrite=overwrite,
                        )
                    return str(saved["path"]), None
                except Exception as exc:  # noqa: BLE001 - report per-artifact failures
                    return None, {
                        "paper_id": paper_id,
                        "artifact": artifact_kind,
                        "error": str(exc),
                    }

        results = await asyncio.gather(*(
            download_one(paper_id, artifact_kind, destination)
            for paper_id, artifact_kind, destination in plan
        ))
        return {
            "paper_ids": normalized_ids,
            "artifacts": normalized_artifacts,
            "save_dir": str(destination_dir),
            "paths": [path for path, _ in results if path is not None],
            "errors": [error for _, error in results if error is not None],
        }

    async def match_document(self, text: str, top_k: int = 20,
                             agg: str = "coverage") -> dict:
        """Given a chunk of text, find the corpus papers most similar in content
        (功能2 / match-doc). Each query chunk is retrieved independently then
        aggregated per paper by coverage (default) or maxsim."""
        payload = {"input_type": "text", "text": text, "top_k": top_k, "agg": agg}
        return await self._post("/api/match-doc", payload)

    async def match_paper(self, paper_id: str, top_k: int = 20,
                          agg: str = "coverage") -> dict:
        """Queue a durable exact content-neighbour match."""
        payload = {"input_type": "paper_id", "paper_id": paper_id,
                   "top_k": top_k, "agg": agg}
        return await self._post("/api/match-doc/jobs", payload)

    async def match_paper_status(self, task_id: int, *, cancel: bool = False) -> dict:
        """Read one durable match task, optionally cancelling it first.

        Uses the owner-scoped match-job routes rather than generic /api/tasks:
        task ids are sequential and /api/tasks has no owner concept, so it would
        expose (and cancel) other users' jobs of every type.
        """
        if cancel:
            return await self._post(f"/api/match-doc/jobs/{task_id}/cancel", {})
        return await self._get(f"/api/match-doc/jobs/{task_id}")

    async def embed(self, texts: list[str], include_sparse: bool = False) -> dict:
        """Embed texts with the backend's shared BGE-M3 (option B): the code KB and
        ACE playbook reuse the warm model over HTTP instead of loading their own.
        Returns {"model","dim","dense":[[...]], "sparse": ... | None}."""
        return await self._post(
            "/api/embed", {"texts": texts, "include_sparse": include_sparse})

    async def list_conferences(self) -> dict:
        """The venue registry (titles, tiers, years available)."""
        return await self._get("/api/conferences")

    async def corpus_stats(self) -> dict:
        """Corpus size: papers, chunks, vector index size, embedding model."""
        return await self._get("/api/stats")

    async def record_journal(
        self,
        *,
        project_slug: str,
        body: str,
        client: str = "mcp",
        source_kind: str = "memory",
        title: str | None = None,
        summary: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        dedupe_policy: str = "append",
    ) -> dict:
        payload = {
            "project_slug": project_slug,
            "client": client,
            "source_kind": source_kind,
            "body": body,
            "tags": tags or [],
            "metadata": metadata or {},
            "dedupe_policy": dedupe_policy,
        }
        if title is not None:
            payload["title"] = title
        if summary is not None:
            payload["summary"] = summary
        return await self._post("/api/journal/entries", payload)

    async def search_journal(
        self,
        *,
        project_slug: str,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        dedup_by_entry: bool = True,
        include_chunks: bool = False,
        chunk_limit: int | None = None,
        path_glob: str | None = None,
    ) -> dict:
        payload = {
            "project_slug": project_slug,
            "query": query,
            "top_k": top_k,
            "mode": mode,
            "dedup_by_entry": dedup_by_entry,
            "include_chunks": include_chunks,
        }
        if chunk_limit is not None:
            payload["chunk_limit"] = chunk_limit
        if path_glob:
            payload["path_glob"] = path_glob
        return await self._post("/api/journal/search", payload)

    async def recent_journal_entries(self, *, project_slug: str, limit: int = 10) -> dict:
        return await self._get("/api/journal/entries",
                               {"project_slug": project_slug, "limit": limit})

    async def reconcile_journal_docs(self, *, project_slug: str, live_titles: list[str],
                                     max_depth: int | None = None,
                                     patterns: list[str] | None = None) -> dict:
        return await self._post("/api/journal/docs/reconcile", {
            "project_slug": project_slug,
            "live_titles": live_titles,
            "max_depth": max_depth,
            "patterns": patterns,
        })

    async def submit_journal_bulk_index(self, **payload: Any) -> dict:
        return await self._post("/api/journal/bulk-index", payload)

    async def journal_bulk_index_job(self, job_id: str) -> dict:
        return await self._get(f"/api/journal/bulk-index/jobs/{quote(job_id, safe='')}")

    # ---- index inventory (full loop: client reports what it indexed, where) ----

    async def report_index(
        self,
        *,
        host: str,
        project: str,
        files: list[dict],
        root_dir: str | None = None,
        replace: bool = True,
        patterns: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict:
        return await self._post("/api/index/report", {
            "host": host, "project": project, "root_dir": root_dir,
            "files": files, "replace": replace,
            "patterns": patterns, "max_depth": max_depth,
        })

    async def index_inventory(
        self,
        *,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
        detail: bool = False,
    ) -> dict:
        params = {k: v for k, v in
                  {"host": host, "project": project, "kind": kind}.items() if v is not None}
        # The endpoint defaults to a SUMMARY (no per-file tree); ask for detail only when the
        # caller needs the per-file content hashes (the incremental-index diff baseline).
        if detail:
            params["detail"] = True
        return await self._get("/api/index/inventory", params)

    async def scope_overlap(
        self,
        *,
        project: str,
        host: str | None = None,
        kind: str | None = None,
        respect_gitignore: bool | None = None,
    ) -> dict:
        """Parent/child overlap for a candidate index scope — a KB-scale conflict list
        computed server-side (not the whole per-file inventory).

        ``respect_gitignore`` is the filter the candidate will index under; the backend
        needs it to tell a redundant child from one reaching files the parent skipped."""
        params = {k: v for k, v in
                  {"project": project, "host": host, "kind": kind,
                   "respect_gitignore": respect_gitignore}.items() if v is not None}
        return await self._get("/api/index/scope-overlap", params)

    async def upsert_index_policy(self, **policy: Any) -> dict:
        return await self._post("/api/index/policy", policy)

    async def get_index_policy(self, *, host: str, project: str,
                               kind: str = "code") -> dict:
        return await self._get("/api/index/policy", {
            "host": host, "project": project, "kind": kind})

    # ---- ACE playbook (server-side, per-user) ----

    async def playbook_record(self, *, project_slug: str, content: str,
                              section: str = "insights", domain: str | None = None) -> dict:
        return await self._post("/api/playbook/bullets", {
            "project_slug": project_slug, "content": content,
            "section": section, "domain": domain,
        })

    async def playbook_search(self, *, project_slug: str, query: str,
                              top_k: int = 5, domain: str | None = None) -> dict:
        return await self._post("/api/playbook/search", {
            "project_slug": project_slug, "query": query,
            "top_k": top_k, "domain": domain,
        })

    async def playbook_list(self, *, project_slug: str | None = None,
                            domain: str | None = None, limit: int = 200) -> dict:
        params = {k: v for k, v in
                  {"project_slug": project_slug, "domain": domain, "limit": limit}.items()
                  if v is not None}
        return await self._get("/api/playbook/bullets", params)

    # ---- curated public code (server-side mirrors, indexing, and search) ----

    async def public_code_sources(self) -> list[dict]:
        return await self._get("/api/public-code/sources")

    async def add_public_code_source(
        self,
        *,
        source_id: str,
        domain: str,
        upstream_url: str,
        relative_path: str,
        default_ref: str = "HEAD",
        proxy_profile: str | None = None,
        credential_profile: str | None = None,
        language_overrides: dict[str, str] | None = None,
        update_policy: str = "manual",
        update_interval_minutes: int | None = None,
        index_now: bool = True,
    ) -> dict:
        return await self._post("/api/public-code/sources", {
            "source_id": source_id,
            "domain": domain,
            "upstream_url": upstream_url,
            "relative_path": relative_path,
            "default_ref": default_ref,
            "proxy_profile": proxy_profile,
            "credential_profile": credential_profile,
            "language_overrides": language_overrides or {},
            "update_policy": update_policy,
            "update_interval_minutes": update_interval_minutes,
            "index_now": index_now,
        })

    async def check_public_code_updates(
        self, *, source_ids: list[str], domains: list[str]
    ) -> dict:
        return await self._post("/api/public-code/sources/check-updates", {
            "source_ids": source_ids,
            "domains": domains,
        })

    async def index_public_code_sources(
        self, *, source_ids: list[str], domains: list[str]
    ) -> dict:
        return await self._post("/api/public-code/sources/index", {
            "source_ids": source_ids,
            "domains": domains,
        })

    async def set_public_code_update_policy(
        self,
        *,
        source_id: str,
        update_policy: str,
        update_interval_minutes: int | None,
    ) -> dict:
        encoded_source_id = quote(source_id, safe="")
        return await self._put(
            f"/api/public-code/sources/{encoded_source_id}/update-policy",
            {
                "update_policy": update_policy,
                "update_interval_minutes": update_interval_minutes,
            },
        )

    async def create_public_code_import(self, body: dict) -> dict:
        return await self._post("/api/public-code/imports", body)

    async def upload_public_code_import_part(
        self, import_id: str, offset: int, data: bytes,
    ) -> dict:
        resp = await self._http().put(
            f"/api/public-code/imports/{quote(import_id, safe='')}/data",
            params={"offset": offset}, content=data,
            headers={**self._auth_headers, "Content-Type": "application/octet-stream"},
            timeout=httpx.Timeout(600.0, connect=30.0))
        _raise_for_status(resp)
        return resp.json()

    async def public_code_import(self, import_id: str) -> dict:
        return await self._get(f"/api/public-code/imports/{quote(import_id, safe='')}")

    async def complete_public_code_import(self, import_id: str) -> dict:
        return await self._post(
            f"/api/public-code/imports/{quote(import_id, safe='')}/complete", {})

    async def public_code_job(self, job_id: str) -> dict:
        return await self._get(f"/api/public-code/jobs/{quote(job_id, safe='')}")

    async def public_code_jobs(self, *, limit: int = 50) -> list[dict]:
        return await self._get("/api/public-code/jobs", {"limit": limit})

    async def search_public_code(
        self,
        *,
        query: str,
        source_ids: list[str],
        domains: list[str],
        top_k: int = 10,
        mode: str = "hybrid",
        language: str | None = None,
        rerank: bool = False,
    ) -> dict:
        return await self._post("/api/public-code/search", {
            "query": query,
            "source_ids": source_ids,
            "domains": domains,
            "top_k": top_k,
            "mode": mode,
            "language": language,
            "rerank": rerank,
        })

    # ---- code KB (server-side; client chunks locally, backend embeds + stores) ----
    async def code_index(self, *, project: str, host: str, root_dir: str,
                         files: list[dict], incremental: bool = False,
                         manifest: list[str] | None = None,
                         removed_paths: list[str] | None = None,
                         report: dict | None = None,
                         client_job_id: str | None = None) -> dict:
        # Async since W156: returns 202 {job_id, status, ...} — poll code_index_job.
        # incremental (W158-inc): only `files` (changed) are re-embedded; `manifest` (all
        # current paths) drives removed-file reconciliation server-side.
        # report (W159): catalog records the backend writes on job completion, so the
        # client can submit and return without blocking on the embed to report.
        payload: dict[str, Any] = {
            "project": project, "host": host, "root_dir": root_dir, "files": files,
            "incremental": incremental,
        }
        if client_job_id is not None:
            payload["client_job_id"] = client_job_id
        if manifest is not None:
            payload["manifest"] = manifest
        if removed_paths:
            payload["removed_paths"] = removed_paths
        if report is not None:
            payload["report"] = report
        return await self._post("/api/code/index", payload)

    async def code_delete(self, *, project: str) -> dict:
        """Clear one code project entirely: drop its vectors + inventory rows. `project`
        (the indexed dir's abspath) goes as a QUERY param — a path segment can't hold a
        scope with slashes / a drive colon (W165). Returns {deleted, catalog_removed}."""
        return await self._delete("/api/code", {"project": project})

    async def delete_journal_docs(self, *, project_slug: str) -> dict:
        """Clear a journal doc scope entirely: drop all source_kind="doc" entries + vectors
        AND their inventory rows (memory entries kept). `project_slug` (the indexed dir's
        abspath) goes as a QUERY param — a path segment can't hold slashes / a drive colon.
        Returns {removed_entries, deleted_vectors, catalog_removed}."""
        return await self._delete("/api/journal/docs", {"project_slug": project_slug})

    async def code_index_job(self, job_id: str) -> dict:
        return await self._get(f"/api/code/jobs/{job_id}")

    async def code_index_jobs(self, *, project: str | None = None,
                              active_only: bool = False, limit: int = 20) -> dict:
        params: dict[str, Any] = {"limit": limit, "active_only": active_only}
        if project is not None:
            params["project"] = project
        return await self._get("/api/code/jobs", params)

    async def code_search(self, *, query: str, project: str | None = None,
                          top_k: int = 8, mode: str = "hybrid", rerank: bool = False,
                          language: str | None = None, path_glob: str | None = None,
                          max_per_file: int | None = None) -> dict:
        payload: dict = {"query": query, "top_k": top_k, "mode": mode, "rerank": rerank}
        if project:
            payload["project"] = project
        if language:
            payload["language"] = language
        if path_glob:
            payload["path_glob"] = path_glob
        if max_per_file:
            payload["max_per_file"] = max_per_file
        return await self._post("/api/code/search", payload)

    async def federated_code_search(self, *, query: str, projects: list[str] | None = None,
                                    include_public: bool = False,
                                    source_ids: list[str] | None = None,
                                    domains: list[str] | None = None,
                                    top_k: int = 10, mode: str = "hybrid",
                                    language: str | None = None,
                                    path_glob: str | None = None,
                                    max_per_file: int | None = None,
                                    rerank: bool = False) -> dict:
        payload: dict = {
            "query": query, "top_k": top_k, "mode": mode, "rerank": rerank,
            "include_public": include_public,
            "projects": projects or [], "source_ids": source_ids or [], "domains": domains or [],
        }
        if language:
            payload["language"] = language
        if path_glob:
            payload["path_glob"] = path_glob
        if max_per_file is not None:
            payload["max_per_file"] = max_per_file
        return await self._post("/api/code/federated-search", payload)

    async def code_duplicates(self, *, project: str, threshold: float = 0.92,
                              min_shared_chunks: int = 2, max_pairs: int = 50,
                              max_examples: int = 3,
                              exclude_globs: list[str] | None = None) -> dict:
        payload: dict = {
            "project": project, "threshold": threshold,
            "min_shared_chunks": min_shared_chunks, "max_pairs": max_pairs,
            "max_examples": max_examples,
        }
        # None means "server defaults"; [] means "scan everything" and must survive.
        if exclude_globs is not None:
            payload["exclude_globs"] = exclude_globs
        return await self._post("/api/code/duplicates", payload)


# --------------------------------------------------------------------------- #
# Compact formatters (pure) — turn API JSON into token-efficient agent text.
# --------------------------------------------------------------------------- #

def _snippet(text: str, limit: int = 280) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# Mirrors mcp_plugin/retrieval_mcp.py. Uploads and webpages carry no paper_id —
# `papers` dispatches payload profiles on source_type and only paper/review get
# that key (core/storage/papers_payload.py) — so their handle is the numeric
# documents.id, emitted `doc:`-prefixed and routed back by _scope_filter.
_DOC_ID_PREFIX = "doc:"
_INLINE_DOCUMENT_MARKDOWN_MAX_BYTES = 400_000


def _is_document_id(raw: str) -> bool:
    """True when this id addresses a documents.id rather than a paper_id.

    A bare run of digits counts too: measured against production 2026-09-21,
    0 of 106,156 paper_ids are all-digits, so the namespaces cannot collide.
    """
    value = str(raw or "").strip()
    return value.startswith(_DOC_ID_PREFIX) or value.isdigit()


def _document_id(raw: str) -> str:
    value = str(raw or "").strip()
    if value.startswith(_DOC_ID_PREFIX):
        return value[len(_DOC_ID_PREFIX):].strip()
    return value


def _scope_filter(paper_id: str) -> dict[str, list[str]]:
    """Server-side scope for one id. Exactly one key — they AND if both are sent."""
    if _is_document_id(paper_id):
        return {"doc_ids": [_document_id(paper_id)]}
    return {"paper_ids": [str(paper_id).strip()]}


def _result_id(p: dict) -> str:
    """The id a follow-up call should use — never an empty string for an upload."""
    pid = str(p.get("paper_id") or "").strip()
    if pid:
        return pid
    doc_id = str(p.get("doc_id") or "").strip()
    return f"{_DOC_ID_PREFIX}{doc_id}" if doc_id else ""


def format_public_code_sources(sources: list[dict]) -> str:
    indexed = sum(1 for source in sources if source.get("active_revision"))
    lines = [f"Public code sources: {len(sources)} configured, {indexed} indexed."]
    for source in sources:
        revision = str(source.get("active_revision") or "none")[:10]
        policy = str(source.get("update_policy") or "manual")
        interval = source.get("update_interval_minutes")
        if policy == "scheduled" and interval:
            policy = f"scheduled/{interval}m"
        state = "error" if source.get("last_check_error") else (
            "active" if source.get("active_revision") else "configured")
        lines.append(
            f"- {source.get('source_id')} [{source.get('domain')}] {state}; "
            f"{int(source.get('active_manifest_file_count') or 0)} files, "
            f"{int(source.get('active_chunk_count') or 0)} chunks, "
            f"{int(source.get('active_vector_count') or 0)} vectors; "
            f"rev={revision}; policy={policy}; path={source.get('relative_path')}"
        )
        if source.get("last_check_error"):
            lines.append(f"  check_error: {source['last_check_error']}")
    return "\n".join(lines)


def format_public_code_job(job: dict) -> str:
    completed = int(job.get("completed_sources") or 0)
    total = int(job.get("total_sources") or 0)
    progress = float(job.get("progress") or 0.0)
    line = (
        f"Public code job {job.get('job_id')} [{job.get('status')}] "
        f"{job.get('operation')} {completed}/{total} sources ({progress:.0%}); "
        f"phase={job.get('phase')}; files={int(job.get('files_done') or 0)}/"
        f"{int(job.get('files_total') or 0)}; chunks={int(job.get('chunks_done') or 0)}"
    )
    current = job.get("current_source_id") or job.get("current_path")
    if job.get("current_path"):
        current = f"{job.get('current_source_id') or '?'}:{job['current_path']}"
    if current:
        line += f"; current={current}"
    if job.get("message"):
        line += f"; message={job['message']}"
    errors = job.get("errors") or {}
    if errors:
        line += "; errors=" + "; ".join(
            f"{key}: {value}" for key, value in errors.items())
    return line


def _public_code_search_diagnostics(response: dict) -> list[str]:
    """Render retrieval-mode and rerank state without hiding a strict miss."""

    lines: list[str] = []
    requested_mode = response.get("requested_mode")
    effective_mode = response.get("effective_mode")
    if requested_mode is not None or effective_mode is not None:
        lines.append(
            f"Search mode: requested={requested_mode!r}, effective={effective_mode!r}."
        )

    candidate_count = response.get("candidate_count")
    rerank_candidate_count = response.get("rerank_candidate_count")
    if candidate_count is not None:
        if response.get("rerank_applied"):
            lines.append(
                "Rerank: applied to "
                f"{rerank_candidate_count or 0} of {candidate_count} retrieved candidate(s)."
            )
        elif rerank_candidate_count:
            lines.append(
                "Rerank: not applied after considering "
                f"{rerank_candidate_count} of {candidate_count} retrieved candidate(s)."
            )
        else:
            lines.append(f"Rerank: not applied; {candidate_count} retrieved candidate(s).")

    notice = response.get("notice")
    if notice:
        lines.append(f"Search diagnostic: {notice}")
    return lines


def _number_code_lines(code: str, start_line) -> str:
    """Annotate source lines without changing stored code or inventing a location."""
    if not code or type(start_line) is not int or start_line < 1:
        return code
    lines = code.replace("\r\n", "\n").rstrip("\r").split("\n")
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{start_line + offset:>{width}} | {line}"
                     for offset, line in enumerate(lines))


def format_public_code_search(response: dict, *, show_line_numbers: bool = True) -> str:
    hits = response.get("hits") or []
    selected = response.get("selected_source_ids") or []
    header = (
        f"Public code matches for {response.get('query')!r}: {len(hits)} hit(s)"
        + (f" across {', '.join(selected)}" if selected else "")
        + "."
    )
    diagnostics = _public_code_search_diagnostics(response)
    if not hits:
        return "\n".join([header, *diagnostics])
    lines = [header]
    if diagnostics:
        lines.extend(["", *diagnostics])
    lines.append("")
    for index, hit in enumerate(hits, 1):
        revision = str(hit.get("revision") or "")[:10]
        label = f" [{hit.get('label')}]" if hit.get("label") else ""
        score_parts = [f"score={float(hit.get('score') or 0.0):.3f}"]
        if hit.get("retrieval_score") is not None:
            score_parts.append(f"retrieval_score={float(hit['retrieval_score']):.3f}")
        if hit.get("reranker_score") is not None:
            score_parts.append(f"reranker_score={float(hit['reranker_score']):.3f}")
        lines.append(
            f"{index}. {hit.get('source_id')}@{revision} "
            f"{hit.get('path')}:{hit.get('start_line')}-{hit.get('end_line')}"
            f"{label} {' '.join(score_parts)}"
        )
        lines.append(f"```{hit.get('language') or ''}")
        body = str(hit.get("code") or hit.get("content") or "").rstrip("\n")
        lines.append(_number_code_lines(body, hit.get("start_line")) if show_line_numbers else body)
        lines.append("```")
    return "\n".join(lines)


def _is_journal_summary_chunk(chunk: dict) -> bool:
    index = chunk.get("chunk_index")
    return index == 0 or str(index).strip() == "0"


def _journal_display_chunks(hit: dict, *, max_chunks: int = 3) -> list[dict]:
    """Pick concrete journal chunks to show in MCP output.

    The journal API returns ``best_chunk`` plus, when requested, ``chunks``. For
    indexed Markdown docs the best chunk can be the synthetic chunk-0 summary
    ("journal.md source_kind..."), which is useless as handoff context. Prefer
    returned body chunks while preserving backend order, then fall back to the
    summary if that is all the backend returned.
    """
    candidates: list[dict] = []
    seen: set[str] = set()

    def add(chunk: dict | None) -> None:
        if not isinstance(chunk, dict):
            return
        content = str(chunk.get("content") or "")
        if not content.strip():
            return
        key = str(chunk.get("chunk_id") or chunk.get("chunk_index") or content)
        if key in seen:
            return
        seen.add(key)
        candidates.append(chunk)

    for chunk in hit.get("chunks") or []:
        add(chunk)
    if not candidates:
        add(hit.get("best_chunk") or {})

    body_chunks = [chunk for chunk in candidates if not _is_journal_summary_chunk(chunk)]
    return (body_chunks or candidates)[:max_chunks]


def _journal_chunk_label(chunk: dict) -> str:
    parts: list[str] = []
    if chunk.get("chunk_index") is not None:
        parts.append(f"chunk {chunk.get('chunk_index')}")
    heading = str(chunk.get("heading") or "").strip()
    if heading and not _is_journal_summary_chunk(chunk):
        parts.append(heading)
    return " · ".join(parts)


def format_search_results(resp: dict, max_snippet: int = 280) -> str:
    papers = resp.get("papers") or []
    if not papers:
        return "No matching papers found."
    lines = [f"Found {len(papers)} papers (ranked by best-chunk score):", ""]
    for i, p in enumerate(papers, 1):
        meta = p.get("metadata") or {}
        venue = meta.get("venue") or p.get("venue") or "?"
        year = meta.get("year") or p.get("year") or "?"
        title = p.get("doc_title") or p.get("title") or "(untitled)"
        score = p.get("max_score", p.get("score", 0.0))
        pid = _result_id(p)
        lines.append(f"{i}. {title}  [{venue} {year}]  score={score:.3f}")
        lines.append(f"   paper_id: {pid}")
        chunks = p.get("chunks") or []
        if chunks:
            lines.append(f"   ▸ {_snippet(chunks[0].get('content', ''), max_snippet)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_match_results(resp: dict, max_snippet: int = 220) -> str:
    papers = resp.get("papers") or []
    if not papers:
        return "No similar papers found."
    lines = [f"Top {len(papers)} content-nearest papers:", ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title") or "(untitled)"
        venue = p.get("venue") or "?"
        year = p.get("year") or "?"
        score = p.get("score", 0.0)
        coverage = p.get("coverage", 0)
        pid = p.get("paper_id") or ""
        lines.append(f"{i}. {title}  [{venue} {year}]  score={score:.3f}  coverage={coverage}")
        lines.append(f"   paper_id: {pid}")
        matched = p.get("matched") or []
        if matched:
            lines.append(f"   ▸ {_snippet(matched[0].get('paper_chunk', ''), max_snippet)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_stats(resp: dict) -> str:
    return (
        f"Corpus: {resp.get('total_papers', '?')} papers, "
        f"{resp.get('total_chunks', '?')} chunks, "
        f"{resp.get('vector_index_size', '?')} vectors indexed "
        f"(model: {resp.get('embedding_model', '?')})."
    )


def format_journal_record(resp: dict) -> str:
    entry = resp.get("entry") or {}
    return (
        f"Recorded journal entry {entry.get('entry_id', '?')} "
        f"(chunks={resp.get('chunk_count', 0)}, indexed={resp.get('indexed', False)})."
    )


def format_journal_search(resp: dict, max_snippet: int = 260, max_chunks: int = 3) -> str:
    hits = resp.get("hits") or []
    if not hits:
        # A path_glob that matched 0 docs while the scope HAS docs returns a notice
        # explaining the glob is scope-relative — surface it instead of a bare miss.
        return resp.get("notice") or "No matching journal entries found."
    lines = [f"Found {len(hits)} journal entries:", ""]
    for i, hit in enumerate(hits, 1):
        entry = hit.get("entry") or {}
        title = entry.get("title") or entry.get("entry_id") or "(untitled)"
        lines.append(
            f"{i}. {title} [{entry.get('source_kind', '?')}/{entry.get('client', '?')}] "
            f"score={hit.get('score', 0.0):.3f}"
        )
        lines.append(f"   entry_id: {entry.get('entry_id', '')}")
        for chunk in _journal_display_chunks(hit, max_chunks=max_chunks):
            label = _journal_chunk_label(chunk)
            prefix = f"   ▸ [{label}] " if label else "   ▸ "
            lines.append(prefix + _snippet(chunk.get("content", ""), max_snippet))
        lines.append("")
    return "\n".join(lines).rstrip()


def _project_freshness(project: dict) -> str:
    """' — last indexed <ts>[ @<commit>]' for a project node, or '' if unknown.

    Prefers the server-computed summary fields (last_indexed_at / last_commit) so the compact
    inventory needs no per-file tree; falls back to walking the file nodes for a detail
    response or an older server. A git commit, if the client recorded one in file metadata at
    index time, is surfaced too so you can tell WHICH revision is indexed."""
    latest = project.get("last_indexed_at") or ""
    commit = project.get("last_commit") or ""
    if not latest:  # detail response / older server → derive from the per-file nodes
        for d in project.get("dirs") or []:
            for f in d.get("files") or []:
                ts = f.get("indexed_at") or ""
                if ts > latest:
                    latest = ts
                commit = commit or ((f.get("metadata") or {}).get("git_commit") or "")
    if not latest:
        return ""
    return f" — last indexed {latest[:19].replace('T', ' ')}" + (f" @{commit}" if commit else "")


def _code_snapshot_inventory_note(project: dict) -> str | None:
    state = project.get("code_snapshot_state")
    if not state:
        return None
    line = f"code snapshot: {state}"
    job_id = project.get("code_active_index_job_id")
    if job_id:
        line += f"; job={job_id}"
    if project.get("code_recovery_required"):
        line += "; full recovery index required after interrupted write"
    previous = project.get("code_previous_completed_snapshot_available")
    if previous is False:
        line += "; completed snapshot unavailable"
    elif previous is True and state == "preparing":
        line += "; previous completed snapshot available"
    return line

def _same_scope_path(left: str, right: str) -> bool:
    return left.replace("\\", "/").rstrip("/").lower() == right.replace("\\", "/").rstrip("/").lower()


def _format_inventory_summary(resp: dict, *, project: str | None = None,
                              limit: int = 40) -> str:
    """One line per project, at most `limit`: scopes whose code snapshot is not plainly
    completed first (they need attention), then newest first. Rendering all 305 scopes
    of one user, each with its path twice and a "completed" note, was 56K characters,
    more than MCP clients accept (2026-10-09)."""
    users = resp.get("users") or []
    totals = resp.get("totals") or {}
    if not users:
        return "No indexed files reported yet."
    needle = (project or "").replace("\\", "/").lower()
    rows = []
    for user in users:
        owner = user.get("owner_user_id") or "?"
        for host in user.get("hosts") or []:
            for node in host.get("projects") or []:
                name = str(node.get("project") or "?")
                if needle and needle not in name.replace("\\", "/").lower():
                    continue
                attention = (node.get("code_snapshot_state") not in (None, "completed")
                             or bool(node.get("code_recovery_required")))
                label = host.get("host") or "?"
                if len(users) > 1:
                    label = f"{owner} @ {label}"
                rows.append((attention, node.get("last_indexed_at") or "", label, node))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    shown = rows[:max(1, limit)]
    lines = [f"Inventory: {totals.get('hosts', 0)} host(s), {totals.get('projects', 0)} "
             f"project(s), {totals.get('files', 0)} file(s), {totals.get('chunks', 0)} chunk(s)."]
    if needle:
        lines.append(f"{len(rows)} project(s) match {project!r}.")
    if len(rows) > len(shown):
        lines.append(f"Listing {len(shown)} of {len(rows)}: scopes needing attention, then "
                     "the most recently indexed. Narrow with project=<path fragment> or "
                     "host=, or raise limit.")
    per_host: dict[str, int] = {}
    for row in rows:
        per_host[row[2]] = per_host.get(row[2], 0) + 1
    groups: dict[str, list[dict]] = {}
    for _, _, label, node in shown:
        groups.setdefault(label, []).append(node)
    for label, nodes in groups.items():
        lines.append(f"▸ {label} ({len(nodes)} of {per_host[label]} project(s))")
        for node in nodes:
            name = str(node.get("project") or "?")
            line = f"  · {name} ({node.get('file_count', 0)} files){_project_freshness(node)}"
            root = node.get("root_dir")
            if root and not _same_scope_path(str(root), name):
                line += f" [root {root}]"
            note = _code_snapshot_inventory_note(node)
            if note and (node.get("code_snapshot_state") != "completed"
                         or node.get("code_recovery_required")):
                line += f"; {note}"
            lines.append(line)
    return "\n".join(lines)


def format_inventory(resp: dict) -> str:
    users = resp.get("users") or []
    totals = resp.get("totals") or {}
    if not users:
        return "No indexed files reported yet (run index_code first)."
    lines = [
        f"Index inventory: {totals.get('users', 0)} user(s), {totals.get('hosts', 0)} host(s), "
        f"{totals.get('projects', 0)} project(s), {totals.get('files', 0)} file(s), "
        f"{totals.get('chunks', 0)} chunk(s).",
        "",
    ]
    for u in users:
        lines.append(f"● user {u.get('owner_user_id', '?')}  "
                     f"({u.get('file_count', 0)} files / {u.get('chunk_count', 0)} chunks)")
        for h in u.get("hosts") or []:
            lines.append(f"  ▸ host {h.get('host', '?')}")
            for p in h.get("projects") or []:
                lines.append(f"    · project {p.get('project', '?')} "
                             f"[{p.get('root_dir') or '?'}] "
                             f"({p.get('file_count', 0)} files / {p.get('chunk_count', 0)} chunks)"
                             f"{_project_freshness(p)}")
                if note := _code_snapshot_inventory_note(p):
                    lines.append(f"      {note}")
                for d in p.get("dirs") or []:
                    lines.append(f"      - {d.get('dir', '.')}/  "
                                 f"({d.get('file_count', 0)} files, {d.get('chunk_count', 0)} chunks)")
    return "\n".join(lines)


def format_journal_recent(resp: dict) -> str:
    entries = resp.get("entries") or []
    if not entries:
        return "No recent journal entries found."
    lines = [f"Recent journal entries ({len(entries)} of {resp.get('total', len(entries))}):", ""]
    for i, entry in enumerate(entries, 1):
        title = entry.get("title") or entry.get("summary") or entry.get("entry_id") or "(untitled)"
        lines.append(f"{i}. {title} [{entry.get('source_kind', '?')}/{entry.get('client', '?')}]")
        lines.append(f"   entry_id: {entry.get('entry_id', '')}")
    return "\n".join(lines)


# ---- public-code bundle import: a clean local clone replaces the server's network clone ----
# Byte-identical in mcp_plugin/retrieval_mcp.py and mcp_server/backend.py (asserted by tests).

_PUBLIC_IMPORT_PART_BYTES = 32 * 1024 * 1024
_PUBLIC_IMPORT_PART_ATTEMPTS = 5


def _https_upstream(url: str) -> str:
    """git@github.com:owner/repo.git -> https://github.com/owner/repo.git; others unchanged."""
    match = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", url.strip())
    return f"https://{match.group(1)}/{match.group(2)}" if match else url.strip()


def _upstream_key(url: str) -> str:
    """host/owner/repo, so https/ssh and with/without .git name one upstream."""
    url = _https_upstream(url).rstrip("/")
    url = url[:-4] if url.endswith(".git") else url
    rest = url.partition("://")[2] or url
    host, _, path = rest.partition("/")
    return f"{host.lower()}/{path}"


def _run_git_text(repo: str, *arguments: str, timeout: float = 120) -> tuple[int, str]:
    # stdin=DEVNULL: an MCP stdio server's stdin is the JSON-RPC pipe (W159)
    out = subprocess.run(["git", "-C", repo, *arguments], stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=timeout)
    return out.returncode, (out.stdout if out.returncode == 0 else out.stderr).strip()


def _inspect_public_clone(path: str) -> dict:
    """The clone's HEAD, branch and origin; ValueError unless it is published upstream code.

    Only a clean HEAD that is on an origin branch is accepted, so the public corpus never
    receives local edits or unpublished commits. A --depth 1 clone or a bare mirror is fine."""
    code, bare = _run_git_text(path, "rev-parse", "--is-bare-repository")
    if code != 0:
        raise ValueError(f"{path} is not a git repository: {bare}")
    repo = path
    if bare != "true":
        repo = _run_git_text(path, "rev-parse", "--show-toplevel")[1]
        code, dirty = _run_git_text(repo, "status", "--porcelain", "--untracked-files=no")
        if code != 0 or dirty:
            raise ValueError(
                f"{repo} has uncommitted changes to tracked files; only published upstream "
                "code is accepted, so commit nothing local and stash or discard them")
    code, head = _run_git_text(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if code != 0:
        raise ValueError(f"{repo} has no HEAD commit: {head}")
    code, origin = _run_git_text(repo, "remote", "get-url", "origin")
    if code != 0 or not origin:
        raise ValueError(f"{repo} has no origin remote, so its upstream is unknown")
    if bare != "true":
        code, containing = _run_git_text(repo, "branch", "-r", "--contains", "HEAD")
        if code != 0 or not any(line.strip().startswith("origin/")
                                for line in containing.splitlines()):
            raise ValueError(
                f"HEAD {head[:12]} of {repo} is not on any origin branch; only published "
                "upstream code is accepted (fetch, then check out an origin branch)")
    code, branch = _run_git_text(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    return {"repo": repo, "head": head, "origin": _https_upstream(origin),
            "branch": branch if code == 0 else None}


def _resolve_public_import_source(clone: dict, sources: list[dict], *, domain: str | None,
                                  source_id: str | None) -> dict:
    """Reuse the registered source with the same upstream; otherwise describe a new one."""
    key = _upstream_key(clone["origin"])
    for source in sources:
        if _upstream_key(source.get("upstream_url") or "") != key:
            continue
        for name, given in (("source_id", source_id), ("domain", domain)):
            if given and given != source.get(name):
                raise ValueError(
                    f"{clone['origin']} is already registered as {source['source_id']} in "
                    f"domain {source['domain']}; omit {name} or pass {source.get(name)!r}")
        ref = source.get("default_ref") or "HEAD"
        if ref != "HEAD" and ref.removeprefix("refs/heads/") != clone["branch"]:
            raise ValueError(
                f"{source['source_id']} indexes branch {ref}; check out that branch "
                f"(the clone is on {clone['branch'] or 'a detached HEAD'})")
        return {"source_id": source["source_id"], "domain": source["domain"],
                "relative_path": source["relative_path"], "default_ref": ref,
                "upstream_url": source["upstream_url"], "new": False}
    if not domain:
        known = sorted({str(source.get("domain")) for source in sources})
        raise ValueError(
            f"{clone['origin']} is not a registered public source; pass domain to add it "
            f"(existing domains: {', '.join(known)})")
    name = source_id or key.rsplit("/", 1)[-1]
    return {"source_id": name, "domain": domain, "relative_path": f"{domain}/{name}",
            "default_ref": "HEAD", "upstream_url": clone["origin"], "new": True}


def _create_public_bundle(clone: dict, target: str) -> tuple[int, str]:
    """Bundle HEAD (and its branch) of the clone; returns (size, sha256)."""
    refs = ["HEAD"] + ([clone["branch"]] if clone["branch"] else [])
    code, message = _run_git_text(clone["repo"], "bundle", "create", target, *refs,
                                  timeout=3600)
    if code != 0:
        raise ValueError(f"git bundle create failed: {message}")
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return os.path.getsize(target), digest.hexdigest()


async def _run_public_import(status: dict, clone: dict, source: dict, *,
                             create: Callable[[dict], Awaitable[dict]],
                             put: Callable[[str, int, bytes], Awaitable[dict]],
                             get: Callable[[str], Awaitable[dict]],
                             complete: Callable[[str], Awaitable[dict]]) -> None:
    """Bundle, upload in resumable parts, complete. Progress lands in `status`."""
    directory = tempfile.mkdtemp(prefix="retrieval-import-")
    target = os.path.join(directory, f"{source['source_id']}.bundle")
    try:
        status["phase"] = "bundling"
        size, sha256 = await asyncio.to_thread(_create_public_bundle, clone, target)
        status.update(phase="uploading", bytes_total=size, bytes_done=0)
        created = await create({
            "source_id": source["source_id"], "domain": source["domain"],
            "relative_path": source["relative_path"], "upstream_url": source["upstream_url"],
            "default_ref": source["default_ref"], "head_commit": clone["head"],
            "size_bytes": size, "sha256": sha256,
            "private_project": clone["repo"],
        })
        import_id = created["import_id"]
        status["import_id"] = import_id
        offset, failures = 0, 0
        with open(target, "rb") as handle:
            while offset < size:
                handle.seek(offset)
                data = await asyncio.to_thread(handle.read, _PUBLIC_IMPORT_PART_BYTES)
                try:
                    offset = int((await put(import_id, offset, data))["received_bytes"])
                    failures = 0
                except httpx.HTTPError:
                    failures += 1
                    if failures >= _PUBLIC_IMPORT_PART_ATTEMPTS:
                        raise
                    await asyncio.sleep(2 ** failures)
                    # resume where the server says it is, not where this side thinks
                    offset = int((await get(import_id))["received_bytes"])
                status["bytes_done"] = offset
        receipt = await complete(import_id)
        status.update(phase="submitted", job_id=receipt["job"]["job_id"])
    except Exception as exc:
        status.update(phase="failed", error=str(exc))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _format_public_import(handle: str, status: dict) -> str:
    total = int(status.get("bytes_total") or 0)
    done = int(status.get("bytes_done") or 0)
    line = (f"Public code import {handle} [{status.get('phase')}] {status.get('source_id')} "
            f"@ {str(status.get('head') or '')[:12]}")
    if total:
        line += f"; uploaded {done / 1e6:.1f}/{total / 1e6:.1f} MB"
    if status.get("import_id"):
        line += f"; import_id={status['import_id']}"
    if status.get("error"):
        line += f"; error={status['error']}"
    return line
