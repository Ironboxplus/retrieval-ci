"""Read-only access to the Cfs volumes on this host's OpenList node.

The 115 volumes (tier1 and medical PDFs, the arXiv daily inbox) are written by the
GitHub Actions node and read here: a pipeline stage lists a volume through the local
node and copies what it needs. The node signs in as a user that may only read and
list (no PUT, rename, move, copy or delete bit), so a bug here cannot write the volume.

Adapted from actions/cfs_upload/openlist.py (Api.call / login / list_tree / read_file,
branch actions/paper-download at 35e6d56), with a streamed read in place of the
whole-body one: a PDF or an arXiv package is written to ``<dest>.part``, its size is
checked, and only then renamed into place.

Configuration, from the backend's environment:

  OPENLIST_URL              node address (default http://127.0.0.1:5244)
  OPENLIST_READER_USER      the read-only user
  OPENLIST_READER_PASS_FILE file holding its password (never the password itself)

The password and the sign-in token are removed from every error message.
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import urllib.parse
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_URL = "http://127.0.0.1:5244"
CHUNK = 1 << 20


class VolumeError(RuntimeError):
    pass


class VolumeNotFound(VolumeError):
    pass


def _read_password(password_file: Path) -> str:
    try:
        password = Path(password_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise VolumeError(f"cannot read the reader password file {password_file}: {exc.strerror or exc}") from None
    if not password:
        raise VolumeError(f"the reader password file {password_file} is empty")
    return password


class VolumeReader:
    def __init__(self, url: str = DEFAULT_URL, user: str = "", password: str = "", *,
                 password_file: Optional[Path] = None, timeout: float = 300.0) -> None:
        if password_file is not None:
            password = _read_password(password_file)
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise VolumeError(f"OPENLIST_URL must be an http(s) URL, got {url!r}")
        self.url = url.rstrip("/")
        self._parts = parts
        self.user = user
        self._password = password
        self._basic = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._token = ""
        self.timeout = timeout

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "VolumeReader":
        env = os.environ if env is None else env
        user = env.get("OPENLIST_READER_USER", "")
        password_file = env.get("OPENLIST_READER_PASS_FILE", "")
        if not user:
            raise VolumeError("OPENLIST_READER_USER is not set (the read-only user of this host's OpenList node)")
        if not password_file:
            raise VolumeError("OPENLIST_READER_PASS_FILE is not set (the file holding that user's password)")
        return cls(env.get("OPENLIST_URL") or DEFAULT_URL, user, password_file=Path(password_file))

    # ------------------------------------------------------------------ plumbing
    def _redact(self, text: str) -> str:
        for secret in (self._password, self._basic, self._token):
            if secret:
                text = text.replace(secret, "***")
        return text

    def _connection(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._parts.scheme == "https" else http.client.HTTPConnection
        return cls(self._parts.hostname, self._parts.port, timeout=self.timeout)

    def _json(self, method: str, path: str, body: Optional[dict] = None, *, retry_login: bool = True) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = self._token
        conn = self._connection()
        try:
            conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                         headers=headers)
            resp = conn.getresponse()
            payload = resp.read()
        except (OSError, http.client.HTTPException) as exc:
            raise VolumeError(self._redact(f"{method} {path}: {type(exc).__name__}: {exc}")) from None
        finally:
            conn.close()
        try:
            doc = json.loads(payload)
        except ValueError:
            raise VolumeError(self._redact(f"{method} {path}: HTTP {resp.status}, not JSON: {payload[:200]!r}")) from None
        code = doc.get("code")
        if code == 401 and retry_login and path != "/api/auth/login":
            self._login()
            return self._json(method, path, body, retry_login=False)
        if code != 200:
            message = self._redact(str(doc.get("message")))
            # "user not found" at sign-in is a credentials problem, not a missing path.
            error = VolumeNotFound if "not found" in message and path != "/api/auth/login" else VolumeError
            raise error(f"{method} {path}: code {code} {message}")
        return doc.get("data") or {}

    def _login(self) -> None:
        self._token = ""
        data = self._json("POST", "/api/auth/login", {"username": self.user, "password": self._password})
        self._token = data["token"]

    # ------------------------------------------------------------------- listing
    def _list_dir(self, path: str) -> list[dict]:
        if not self._token:
            self._login()
        # refresh=False: the answer comes from the node's index, never a call to 115.
        data = self._json("POST", "/api/fs/list", {"path": path, "page": 1, "per_page": 0, "refresh": False})
        return data.get("content") or []

    def list_files(self, directory: str) -> dict[str, int]:
        """Every file below ``directory`` as {path relative to it: size}."""
        root = directory.rstrip("/")
        out: dict[str, int] = {}

        def walk(path: str, prefix: str) -> None:
            try:
                entries = self._list_dir(path)
            except VolumeNotFound as exc:
                raise VolumeNotFound(f"{path} is not on the volume ({exc})") from None
            for entry in entries:
                rel = prefix + entry["name"]
                if entry["is_dir"]:
                    walk(f"{path}/{entry['name']}", rel + "/")
                else:
                    out[rel] = int(entry["size"])

        walk(root, "")
        return out

    # --------------------------------------------------------------------- reads
    def _get(self, path: str) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        conn = self._connection()
        try:
            conn.request("GET", "/dav" + urllib.parse.quote(path), headers={"Authorization": "Basic " + self._basic})
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise VolumeError(self._redact(f"GET {path}: {type(exc).__name__}: {exc}")) from None
        if resp.status != 200:
            resp.read()
            conn.close()
            error = VolumeNotFound if resp.status == 404 else VolumeError
            raise error(f"GET {path}: HTTP {resp.status}")
        return conn, resp

    def fetch(self, path: str, dest: Path, *, size: Optional[int] = None) -> int:
        """Stream ``path`` to ``dest`` and return its size.

        Written to ``<dest>.part`` and renamed only when the byte count equals the
        node's Content-Length (and ``size``, when the caller knows it from a listing);
        otherwise the part is removed, ``dest`` is left as it was, and VolumeError is raised."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        conn, resp = self._get(path)
        try:
            announced = resp.getheader("Content-Length")
            written = 0
            try:
                with open(part, "wb") as out:
                    for block in iter(lambda: resp.read(CHUNK), b""):
                        out.write(block)
                        written += len(block)
            except (OSError, http.client.HTTPException) as exc:
                raise VolumeError(self._redact(
                    f"GET {path}: transfer cut short after {written} bytes ({type(exc).__name__}: {exc})")) from None
            if announced is not None and written != int(announced):
                raise VolumeError(f"GET {path}: incomplete, {written} bytes of the {announced} announced")
            if size is not None and written != size:
                raise VolumeError(f"GET {path}: {written} bytes, expected {size}")
            os.replace(part, dest)
            return written
        finally:
            conn.close()
            part.unlink(missing_ok=True)

    def read_bytes(self, path: str, *, limit: int = 1 << 20) -> bytes:
        """A small file in memory (a heartbeat, a manifest); refuses one larger than ``limit``."""
        conn, resp = self._get(path)
        try:
            announced = resp.getheader("Content-Length")
            if announced is not None and int(announced) > limit:
                raise VolumeError(f"GET {path}: {announced} bytes, larger than the {limit} allowed")
            try:
                data = resp.read(limit + 1)
            except (OSError, http.client.HTTPException) as exc:
                raise VolumeError(self._redact(f"GET {path}: {type(exc).__name__}: {exc}")) from None
            if len(data) > limit:
                raise VolumeError(f"GET {path}: larger than the {limit} bytes allowed")
            return data
        finally:
            conn.close()


__all__ = ["DEFAULT_URL", "VolumeError", "VolumeNotFound", "VolumeReader"]
