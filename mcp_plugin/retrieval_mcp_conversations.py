"""Claude Code / Codex conversations -> Journal docs, glued through the deja-vu CLI.

deja (https://github.com/vshulcz/deja-vu, MIT) owns transcript parsing, project
attribution and secret redaction; this module only consumes its stable ``--json``
surfaces (``deja last`` / ``deja show``) and never reads deja's own index.  Opt-in via
``RETRIEVAL_JOURNAL_CONVERSATIONS`` (``1`` main sessions, ``agents`` also spawned ones).

Cost contract: a sync with no transcript change issues ZERO HTTP requests.  Per-session
state remembers ``updated`` plus where the still-growing last segment starts, so only
changed sessions are re-read and only segments whose bytes changed are re-uploaded.
No inventory/stats/listing endpoint is ever called.

Self-contained: the plugin installs the pinned deja release itself (sha256 from the
release's checksums.txt is pinned below) and puts a ``zstd`` shim backed by the
``zstandard`` dependency on deja's PATH, so nothing has to be installed by hand.  Only
stdlib + the package's own dependencies; HTTP verbs are injected, because the PyPI wheel
ships this file next to ``retrieval_mcp.py`` and must not import from the repo.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import io
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger("retrieval_mcp.conversations")

MODE_ENV = "RETRIEVAL_JOURNAL_CONVERSATIONS"
DEJA_BIN_ENV = "RETRIEVAL_DEJA_BIN"
SUPPORTED_HARNESSES = ("claude", "codex")
DEJA_JSON_SCHEMA_VERSION = 2
# Pinned deja-vu release.  Hashes are copied from the release's checksums.txt; an archive
# that does not match is refused.  Bumping the version means updating every hash.
DEJA_VERSION = "0.23.5"
DEJA_RELEASE_URL = "https://github.com/vshulcz/deja-vu/releases/download/v{version}/{asset}"
DEJA_RELEASE_SHA256 = {
    ("darwin", "amd64"): "2dafee671d194be807616c39524b412175c4cca894918032bd28ad1c7cb2c1a0",
    ("darwin", "arm64"): "e34e0c477a6968cd8fb35274c3a973521aea09c2338bd04f3446c005d77fec4f",
    ("linux", "amd64"): "b8f8b62c96e56fcb0d7c6110d4cc14cffce4b3075e8881ad4112e1da1e8d3e00",
    ("linux", "arm64"): "e78f048d0b932ec5df3f21ef33171c69de48930390e876bc7afffa6f9a0061c4",
    ("windows", "amd64"): "93f5d095dd0affe8dd55b4e8c95e2dfec0a0648d8f6ba41566f8aae222a81d3e",
    ("windows", "arm64"): "cec28c5d313c841471c1f11a6b146db7346a185ab7fa760d7a32dde5128a41f7",
}
DEJA_DOWNLOAD_TIMEOUT_S = 300.0
DEJA_PAGE_LIMIT = 200  # deja show refuses larger windows.

# A segment closes at the first user turn after this many rendered chars, and by force at
# the hard limit.  Both decisions look only at the prefix, so a closed segment's bytes
# never change when the session later grows (that is what makes re-sync cheap).
SEGMENT_SOFT_CHARS = 8000
SEGMENT_HARD_CHARS = 16000
COMMAND_LINE_CHARS = 200
UPLOAD_MAX_FILES = 500
UPLOAD_MAX_BYTES = 8 * 1024 * 1024

STATE_VERSION = 1
# A cold first index of a large ~/.claude + ~/.codex takes ~2 min; show pages are ~1 s.
DEJA_LIST_TIMEOUT_S = 1800.0
DEJA_SHOW_TIMEOUT_S = 300.0
JOB_POLL_FIRST_S = 0.5
JOB_POLL_MAX_S = 5.0
JOB_TIMEOUT_S = 1800.0

# deja redacts GitHub/HF/OpenAI tokens, URL credentials and Bearer values but (measured
# on 0.23.5) not PyPI tokens.
_PYPI_TOKEN = re.compile(r"pypi-[A-Za-z0-9_-]{16,}")
# Text the harness writes under the user role (measured on this machine's Codex 0.154
# rollouts with deja 0.23.5): instruction files, interrupt markers, subagent completion
# notices and goal-mode prompts.  None of it was typed by the person.
_HARNESS_USER_PREFIXES = (
    "# AGENTS.md instructions for", "<INSTRUCTIONS>", "<environment_context>",
    "<user_instructions>", "<turn_aborted>", "<subagent_notification>",
    "<codex_internal_context",
)
# Codex's IDE extension wraps the typed request in the editor state; keep the request.
_IDE_CONTEXT_PREFIX = "# Context from my IDE setup:"
_IDE_REQUEST_MARKER = "## My request for Codex:"
_ISO_TIME = re.compile(
    r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.\d+)?(Z|[+-]\d\d:\d\d)$")
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

PostFn = Callable[[str, dict], Awaitable[dict]]
GetFn = Callable[[str], Awaitable[dict]]


class DejaError(RuntimeError):
    """deja ran (or tried to) and did not produce usable JSON."""


class DejaNotFound(DejaError):
    """No usable deja binary: a missing RETRIEVAL_DEJA_BIN target, an unsupported
    platform, or a failed/refused download of the pinned release."""


# ---------------------------------------------------------------------------- config

def conversation_mode(environ: dict[str, str] | None = None) -> str:
    """``off`` (unset/0), ``main`` (1) or ``agents`` (agents)."""
    raw = ((environ if environ is not None else os.environ).get(MODE_ENV) or "").strip().lower()
    if raw == "agents":
        return "agents"
    if raw in {"1", "true", "yes", "on", "main"}:
        return "main"
    return "off"


def deja_binary(environ: dict[str, str] | None = None) -> str | None:
    """The operator's explicit ``RETRIEVAL_DEJA_BIN``, if it names a file; else None.

    Without an override the pinned release is installed by :func:`ensure_deja`; a
    ``deja`` on PATH is deliberately not used, so the JSON contract is the pinned one.
    """
    env = environ if environ is not None else os.environ
    explicit = (env.get(DEJA_BIN_ENV) or "").strip()
    return explicit if explicit and os.path.isfile(explicit) else None


def deja_platform() -> tuple[str, str] | None:
    """(os, arch) in deja's release naming, or None when it publishes no build."""
    system = {"win32": "windows", "linux": "linux", "darwin": "darwin"}.get(sys.platform)
    machine = platform.machine().lower()
    arch = {"amd64": "amd64", "x86_64": "amd64", "x64": "amd64",
            "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    key = (system, arch)
    return key if key in DEJA_RELEASE_SHA256 else None


def _deja_exe_name(system: str) -> str:
    return "deja.exe" if system == "windows" else "deja"


def managed_deja_path(state_dir: str | Path) -> Path | None:
    key = deja_platform()
    if key is None:
        return None
    return Path(state_dir) / "deja" / "bin" / DEJA_VERSION / _deja_exe_name(key[0])


def install_deja_archive(archive: bytes, *, expected_sha256: str, system: str,
                         target: Path) -> Path:
    """Verify a release archive against its pinned hash and install its binary at
    ``target`` (atomically).  A mismatching archive is refused before anything is
    extracted."""
    got = hashlib.sha256(archive).hexdigest()
    if got != expected_sha256:
        raise DejaNotFound(f"deja {DEJA_VERSION} archive sha256 {got} != pinned {expected_sha256}")
    name = _deja_exe_name(system)
    if system == "windows":
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            data = zf.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
            member = tf.getmember(name)
            extracted = tf.extractfile(member) if member.isfile() else None
            if extracted is None:
                raise DejaNotFound(f"deja archive has no regular file {name!r}")
            data = extracted.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, target)
    return target


def ensure_deja(state_dir: str | Path, *, timeout: float = DEJA_DOWNLOAD_TIMEOUT_S) -> str:
    """Path of the pinned deja binary, downloading and verifying it on first use.

    Blocking (runs inside the background sync).  urllib honours the standard
    HTTP(S)_PROXY variables.
    """
    import urllib.request

    key = deja_platform()
    if key is None:
        raise DejaNotFound(f"deja publishes no build for {sys.platform}/{platform.machine()}; "
                           f"set {DEJA_BIN_ENV}")
    target = managed_deja_path(state_dir)
    if target.is_file():
        return str(target)
    system, arch = key
    asset = f"deja-vu_{DEJA_VERSION}_{system}_{arch}.{'zip' if system == 'windows' else 'tar.gz'}"
    url = DEJA_RELEASE_URL.format(version=DEJA_VERSION, asset=asset)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            archive = response.read()
    except OSError as exc:
        raise DejaNotFound(f"could not download deja {DEJA_VERSION} ({url}): {exc}") from exc
    install_deja_archive(archive, expected_sha256=DEJA_RELEASE_SHA256[key], system=system,
                         target=target)
    logger.info("installed deja %s at %s", DEJA_VERSION, target)
    return str(target)


def resolve_deja(state_dir: str | Path, environ: dict[str, str] | None = None) -> str:
    """The deja to run: an explicit override (which must exist) or the pinned install."""
    env = environ if environ is not None else os.environ
    if (env.get(DEJA_BIN_ENV) or "").strip():
        binary = deja_binary(env)
        if binary is None:
            raise DejaNotFound(f"{DEJA_BIN_ENV} does not name a file")
        return binary
    return ensure_deja(state_dir)


# deja decompresses Codex's .jsonl.zst rollouts by running ``zstd -d -c -q`` (stdin to
# stdout).  The shim answers exactly that with the ``zstandard`` package the plugin
# already depends on.  Its directory goes LAST on PATH, so a real zstd still wins.
_ZSTD_SHIM = """import sys

import zstandard


def main() -> int:
    if any(arg not in ("-d", "-c", "-q") for arg in sys.argv[1:]):
        sys.stderr.write("zstd shim: only 'zstd -d -c -q' is supported\\n")
        return 2
    out = sys.stdout.buffer
    reader = zstandard.ZstdDecompressor().stream_reader(sys.stdin.buffer, read_across_frames=True)
    try:
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                return 0
            out.write(chunk)
    except zstandard.ZstdError as exc:
        out.flush()
        sys.stderr.write(f"zstd shim: {exc}\\n")
        return 1


sys.exit(main())
"""


def ensure_zstd_shim(state_dir: str | Path, python: str | None = None) -> Path:
    """Write (or refresh) the ``zstd`` shim and return its directory."""
    python = python or sys.executable
    shim_dir = Path(state_dir) / "deja" / "zstd-shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    script = shim_dir / "zstd_shim.py"
    files = {"zstd_shim.py": _ZSTD_SHIM}
    if os.name == "nt":
        files["zstd.bat"] = f'@"{python}" "{script}" %*\r\n'
    else:
        files["zstd"] = f'#!/bin/sh\nexec "{python}" "{script}" "$@"\n'
    for name, content in files.items():
        path = shim_dir / name
        if not path.is_file() or path.read_bytes() != content.encode("utf-8"):
            tmp = path.with_name(f"{name}.{os.getpid()}.tmp")
            tmp.write_bytes(content.encode("utf-8"))
            os.replace(tmp, path)
        if name == "zstd":
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim_dir


def deja_environment(state_dir: str | Path, binary: str | None = None,
                     base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for every deja subprocess; keeps our index apart from the user's own.

    ``DEJA_INCLUDE_SUBAGENTS=1`` is forced so the index content does not depend on the
    mode or on a stray user setting; spawned sessions are filtered by ``kind`` instead.
    The zstd shim directory is appended to PATH for Codex's compressed rollouts.
    """
    env = dict(os.environ if base is None else base)
    index_dir = Path(state_dir) / "deja" / "index.db"
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    env.update({
        "DEJA_OFFLINE": "1",
        "DEJA_TUI": "0",
        "DEJA_STORES": ",".join(SUPPORTED_HARNESSES),
        "DEJA_INCLUDE_SUBAGENTS": "1",
        "DEJA_INDEX_DIR": str(index_dir),
    })
    shim_dir = str(ensure_zstd_shim(state_dir))
    entries = [e for e in env.get("PATH", "").split(os.pathsep) if e]
    if shim_dir not in entries:
        env["PATH"] = os.pathsep.join([*entries, shim_dir])
    return env


def deja_project_name(scope: str) -> str | None:
    """deja's project name for a working directory: its last two path segments, or just
    the last when the one before it is a drive (``C:/prd`` -> ``prd``).  A bare drive
    root is named after the drive.  Matching on the deja side is case-insensitive."""
    parts = [p for p in re.split(r"[\\/]+", scope or "") if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if re.fullmatch(r"[A-Za-z]:", parts[-2]):
        return parts[-1]
    return f"{parts[-2]}/{parts[-1]}"


# -------------------------------------------------------------------------- deja CLI

def run_deja(binary: str, args: list[str], env: dict[str, str], timeout: float) -> dict:
    """Run one deja command and return its parsed ``--json`` stdout.

    stderr carries index progress and is only surfaced on failure.  deja truncates some
    strings by bytes (titles), so decode leniently instead of failing the whole sync.
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.run([binary, *args], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=timeout, check=False,
                              creationflags=flags)
    except FileNotFoundError as exc:
        raise DejaNotFound(f"deja binary not found: {binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DejaError(f"deja {args[0]} timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise DejaError(f"deja {args[0]} could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace").strip()[-300:]
        raise DejaError(f"deja {args[0]} exited {proc.returncode}: {tail}")
    try:
        payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise DejaError(f"deja {args[0]} printed non-JSON output") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != DEJA_JSON_SCHEMA_VERSION:
        got = payload.get("schema_version") if isinstance(payload, dict) else type(payload).__name__
        raise DejaError(f"unsupported deja JSON schema_version {got!r} "
                        f"(expected {DEJA_JSON_SCHEMA_VERSION})")
    return payload


@dataclass(frozen=True)
class Deja:
    binary: str
    env: dict[str, str]

    def list_sessions(self, project: str) -> list[dict]:
        payload = run_deja(self.binary, ["last", "100000", "--json", "--project", project],
                           self.env, DEJA_LIST_TIMEOUT_S)
        return [s for s in payload.get("sessions") or []
                if isinstance(s, dict) and s.get("id") and s.get("harness") in SUPPORTED_HARNESSES]

    def show_page(self, harness: str, session_id: str, offset: int) -> tuple[list[dict], int]:
        payload = run_deja(
            self.binary,
            ["show", session_id, "--harness", harness, "--json",
             "--offset", str(offset), "--limit", str(DEJA_PAGE_LIMIT)],
            self.env, DEJA_SHOW_TIMEOUT_S)
        messages = [m for m in (payload.get("session") or {}).get("messages") or []
                    if isinstance(m, dict)]
        total = int((payload.get("window") or {}).get("total") or 0)
        return messages, total


# ------------------------------------------------------------------------- rendering

def _parse_time(value: Any) -> datetime | None:
    match = _ISO_TIME.match(value) if isinstance(value, str) else None
    if not match:
        return None
    year, month, day, hour, minute, second, zone = match.groups()
    tz = timezone.utc if zone == "Z" else timezone(
        (1 if zone[0] == "+" else -1) * timedelta(hours=int(zone[1:3]), minutes=int(zone[4:6])))
    try:
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second),
                        tzinfo=tz).astimezone(timezone.utc)
    except ValueError:
        return None


def message_sha(message: dict) -> str:
    """Identity of a raw deja message, used to detect a rewritten session."""
    blob = "\0".join(str(message.get(k) or "") for k in ("role", "time", "text"))
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()


def _first_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[0].strip()


def _clip(line: str, limit: int) -> str:
    return line if len(line) <= limit else line[:limit - 1] + "…"


def _typed_user_text(text: str) -> str | None:
    """What the person typed in a user message, or None when the harness wrote it."""
    stripped = text.lstrip()
    if stripped.startswith(_HARNESS_USER_PREFIXES):
        return None
    if stripped.startswith(_IDE_CONTEXT_PREFIX):
        _, marker, request = stripped.partition(_IDE_REQUEST_MARKER)
        request = request.strip("\n")
        return request if marker and request.strip() else None
    return text.strip("\n")


def render_message(message: dict) -> str | None:
    """One kept message as markdown, or None when its role is dropped from Journal.

    user/assistant keep full text; a command keeps one ``$ cmd`` line; edit/wrote keep
    only the file path.  files / tool-output / summary (and unknown roles) are noise for
    recall and are the bulk of a transcript's bytes.
    """
    role = message.get("role")
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    if role == "user":
        body = _typed_user_text(text)
        if body is None:
            return None
    elif role == "assistant":
        body = text.strip("\n")
    elif role == "command":
        line = _first_line(text)
        body = _clip(line if line.startswith("$") else f"$ {line}", COMMAND_LINE_CHARS)
    elif role in ("edit", "wrote"):
        body = f"{'edited' if role == 'edit' else 'wrote'}: {_first_line(text)}"
    else:
        return None
    moment = _parse_time(message.get("time"))
    stamp = f" · {moment:%Y-%m-%d %H:%MZ}" if moment else ""
    return f"**{role}**{stamp}\n{body}\n"


# ------------------------------------------------------------------------ segmenting

@dataclass(frozen=True)
class Segment:
    part: int
    title: str
    body: str
    sha: str
    first_offset: int
    last_offset: int
    first_message_sha: str


def _safe_component(value: str) -> str:
    return _UNSAFE_PATH_CHARS.sub("_", value).strip("._") or "unknown"


def segment_title(harness: str, session_id: str, started: str | None, part: int) -> str:
    moment = _parse_time(started)
    day = f"{moment:%Y-%m-%d}" if moment else "0000-00-00"
    return f".conversations/{_safe_component(harness)}/{day}_{_safe_component(session_id)}/{part:03d}.md"


def _redact_extra(text: str) -> str:
    return _PYPI_TOKEN.sub("[redacted:pypi-token]", text)


def _build_segment(harness: str, session_id: str, started: str | None, part: int,
                   rendered: list[str], first_offset: int, last_offset: int,
                   first_message_sha: str) -> Segment:
    # Only facts fixed by the prefix go in the header: deja's `title` is truncated to a
    # terminal width and `updated`/`words` move as the session grows.
    header = (f"# {harness} session {session_id} · part {part}\n"
              f"started: {started or 'unknown'} · messages {first_offset}–{last_offset}\n\n")
    body = _redact_extra(header + "\n".join(rendered))
    return Segment(part=part, title=segment_title(harness, session_id, started, part),
                   body=body, sha=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                   first_offset=first_offset, last_offset=last_offset,
                   first_message_sha=first_message_sha)


def segment_messages(harness: str, session_id: str, started: str | None,
                     messages: Iterable[tuple[int, dict]], first_part: int = 1) -> list[Segment]:
    """Cut ``(offset, message)`` pairs into segments, in order.

    A segment closes when the NEXT kept message arrives and either the rendered text is
    already >= SEGMENT_SOFT_CHARS and that message is a user turn, or it has reached
    SEGMENT_HARD_CHARS.  The last segment is therefore always the open one.
    """
    segments: list[Segment] = []
    rendered: list[str] = []
    size = 0
    first_offset = last_offset = 0
    first_sha = ""
    part = first_part

    def close() -> None:
        nonlocal rendered, size, part
        segments.append(_build_segment(harness, session_id, started, part, rendered,
                                       first_offset, last_offset, first_sha))
        rendered, size, part = [], 0, part + 1

    for offset, message in messages:
        text = render_message(message)
        if text is None:
            continue
        if rendered and (size >= SEGMENT_HARD_CHARS
                         or (size >= SEGMENT_SOFT_CHARS and message.get("role") == "user")):
            close()
        if not rendered:
            first_offset, first_sha = offset, message_sha(message)
        rendered.append(text)
        size += len(text)
        last_offset = offset
    if rendered:
        close()
    return segments


# ------------------------------------------------------------------------ local state

class ConversationState:
    """Per (host, scope) JSON in the plugin state dir; losing it only costs a re-sync."""

    def __init__(self, state_dir: str | Path, host: str, scope: str) -> None:
        digest = hashlib.sha256(f"{host}\0{scope}".encode("utf-8")).hexdigest()[:16]
        self.path = Path(state_dir) / "conversations" / f"{digest}.json"
        self.host, self.scope = host, scope
        self.sessions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if (isinstance(data, dict) and data.get("version") == STATE_VERSION
                and isinstance(data.get("sessions"), dict)):
            self.sessions = data["sessions"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": STATE_VERSION, "host": self.host, "scope": self.scope,
                   "sessions": self.sessions}
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def commit(self, plan: "SessionPlan") -> None:
        self.sessions[plan.key] = plan.new_state


# ---------------------------------------------------------------------------- planning

@dataclass
class SessionPlan:
    key: str
    harness: str
    session_id: str
    new_state: dict
    to_upload: list[Segment]
    rebuilt: bool = False


def session_key(harness: str, session_id: str) -> str:
    return f"{harness}/{session_id}"


def _fetch_messages(deja: Deja, session: dict, start: int) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    offset = start
    while True:
        page, total = deja.show_page(session["harness"], session["id"], offset)
        out.extend((offset + i, m) for i, m in enumerate(page))
        offset += len(page)
        if not page or offset >= total:
            return out


def plan_session(deja: Deja, session: dict, stored: dict | None) -> SessionPlan:
    harness, session_id = session["harness"], session["id"]
    started = session.get("started")
    old_segments: dict[str, str] = dict((stored or {}).get("segments") or {})
    resume = bool(stored and stored.get("open_segment_first_msg_sha"))
    start = int(stored["open_segment_start_offset"]) if resume else 0
    part = int(stored.get("open_part") or 1) if resume else 1
    rebuilt = False

    messages = _fetch_messages(deja, session, start)
    if resume and (not messages or message_sha(messages[0][1]) != stored["open_segment_first_msg_sha"]):
        # The transcript was rewritten under the open segment (compaction, edit): the
        # remembered boundary is meaningless, so rebuild the session from its start.
        rebuilt = True
        messages = _fetch_messages(deja, session, 0)
        part = 1

    segments = segment_messages(harness, session_id, started, messages, first_part=part)
    new_segments = {**old_segments, **{s.title: s.sha for s in segments}}
    opened = segments[-1] if segments else None
    new_state = {
        "updated": session.get("updated"),
        "open_part": opened.part if opened else 1,
        "open_segment_start_offset": opened.first_offset if opened else 0,
        "open_segment_first_msg_sha": opened.first_message_sha if opened else None,
        "segments": new_segments,
    }
    return SessionPlan(
        key=session_key(harness, session_id), harness=harness, session_id=session_id,
        new_state=new_state, rebuilt=rebuilt,
        to_upload=[s for s in segments if old_segments.get(s.title) != s.sha])


def plan_scope_sync(deja: Deja, project: str, state: ConversationState, mode: str,
                    errors: list[str] | None = None) -> tuple[list[SessionPlan], int]:
    """Blocking: decide what to upload.  Returns (plans for changed sessions, sessions seen).

    A session whose ``updated`` matches the stored value is skipped without any deja
    ``show`` call.  One failing session is recorded in ``errors`` and left for next time.
    """
    sessions = deja.list_sessions(project)
    if mode != "agents":
        sessions = [s for s in sessions if not s.get("kind")]
    plans: list[SessionPlan] = []
    for session in sessions:
        stored = state.sessions.get(session_key(session["harness"], session["id"]))
        if (stored is not None and session.get("updated")
                and stored.get("updated") == session["updated"]):
            continue
        try:
            plans.append(plan_session(deja, session, stored))
        except DejaNotFound:
            raise
        except DejaError as exc:
            logger.warning("conversation session %s/%s skipped: %s",
                           session["harness"], session["id"], exc)
            if errors is not None:
                errors.append(str(exc))
    return plans, len(sessions)


# ---------------------------------------------------------------------------- upload

@dataclass
class SyncResult:
    sessions_seen: int = 0
    sessions_changed: int = 0
    sessions_uploaded: int = 0
    segments_uploaded: int = 0
    requests: int = 0
    errors: list[str] = field(default_factory=list)


async def _await_job(get: GetFn, job_id: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    delay = JOB_POLL_FIRST_S
    while True:
        job = await get(f"/api/journal/bulk-index/jobs/{job_id}")
        if job.get("status") in {"completed", "failed"}:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"bulk-index job {job_id} still {job.get('status')} "
                               f"after {timeout_s:.0f}s")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, JOB_POLL_MAX_S)


def _upload_request(scope: str, host: str, harness: str, segments: list[Segment]) -> dict:
    files, reports = [], []
    for seg in segments:
        size = len(seg.body.encode("utf-8"))
        files.append({"title": seg.title, "body": seg.body, "content_hash": seg.sha,
                      "size_bytes": size})
        reports.append({"rel_path": seg.title, "kind": "doc", "collection": "journal_chunks",
                        "chunk_count": 0, "content_hash": seg.sha, "size_bytes": size})
    return {"project_slug": scope, "host": host, "root_dir": scope, "files": files,
            "report_files": reports, "replace": False, "live_titles": [], "patterns": None,
            "max_depth": None, "client": harness}


def _batches(items: list[tuple[SessionPlan, Segment]]) -> Iterable[list[tuple[SessionPlan, Segment]]]:
    batch: list[tuple[SessionPlan, Segment]] = []
    size = 0
    for item in items:
        nbytes = len(item[1].body.encode("utf-8"))
        if batch and (len(batch) >= UPLOAD_MAX_FILES or size + nbytes > UPLOAD_MAX_BYTES):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += nbytes
    if batch:
        yield batch


async def upload_plans(plans: list[SessionPlan], *, scope: str, host: str, state: ConversationState,
                       post: PostFn, get: GetFn, result: SyncResult,
                       job_timeout_s: float = JOB_TIMEOUT_S) -> None:
    """Upload each plan's changed segments; commit a session only after every request
    carrying one of its segments completed, so a failure re-sends on the next sync."""
    failed: set[str] = set()
    outstanding: dict[str, int] = {}
    for plan in plans:
        if not plan.to_upload:
            state.commit(plan)  # content unchanged (only dropped roles grew): no HTTP
        else:
            outstanding[plan.key] = len(plan.to_upload)
    by_key = {plan.key: plan for plan in plans}
    for harness in SUPPORTED_HARNESSES:
        items = [(plan, seg) for plan in plans if plan.harness == harness
                 for seg in plan.to_upload]
        for batch in _batches(items):
            segments = [seg for _, seg in batch]
            try:
                job = await post("/api/journal/bulk-index",
                                 _upload_request(scope, host, harness, segments))
                result.requests += 1
                job_id = job.get("job_id")
                if not job_id:
                    raise RuntimeError("bulk-index returned no job_id")
                final = await _await_job(get, job_id, job_timeout_s)
                if final.get("status") != "completed":
                    raise RuntimeError(f"bulk-index job {job_id} failed: {final.get('error')}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep old state; next sync retries
                logger.warning("conversation upload for %s failed: %s", scope, exc)
                result.errors.append(str(exc))
                failed.update(plan.key for plan, _ in batch)
                continue
            result.segments_uploaded += len(batch)
            for plan, _ in batch:
                outstanding[plan.key] -= 1
                if outstanding[plan.key] == 0 and plan.key not in failed:
                    state.commit(by_key[plan.key])
                    result.sessions_uploaded += 1
            state.save()
    if plans:
        state.save()


async def sync_scope(*, scope: str, host: str, state_dir: str | Path, post: PostFn, get: GetFn,
                     mode: str, binary: str | None = None,
                     environ: dict[str, str] | None = None) -> SyncResult:
    """Sync one scope's conversations into Journal.  Raises DejaNotFound / DejaError when
    deja itself is unusable; per-session and upload failures are collected in ``errors``."""
    if not binary:
        binary = await asyncio.to_thread(resolve_deja, state_dir, environ)
    project = deja_project_name(scope)
    result = SyncResult()
    if project is None:
        return result
    deja = Deja(binary, deja_environment(state_dir, binary))
    state = ConversationState(state_dir, host, scope)
    plans, seen = await asyncio.to_thread(plan_scope_sync, deja, project, state, mode, result.errors)
    result.sessions_seen, result.sessions_changed = seen, len(plans)
    await upload_plans(plans, scope=scope, host=host, state=state, post=post, get=get, result=result)
    return result


# ----------------------------------------------------------------- background kicker

class ConversationSyncer:
    """Single-flight, rate-limited background sync per (host, scope) for the MCP tools.

    ``kick`` never waits for the sync: it starts at most one task per scope, never
    sooner than ``min_interval_s`` after the previous start, and returns a line for the
    tool output only when the last sync failed (None otherwise).
    """

    def __init__(self, *, post: PostFn, get: GetFn, state_dir: Callable[[], str | Path],
                 min_interval_s: float = 60.0) -> None:
        self._post, self._get, self._state_dir = post, get, state_dir
        self.min_interval_s = min_interval_s
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._started_at: dict[tuple[str, str], float] = {}
        self._last: dict[tuple[str, str], dict] = {}
        # One run at a time per process: all scopes share one deja index, and a cold
        # first index must not be started twice.
        self._lock: asyncio.Lock | None = None

    def kick(self, host: str, scope: str, *, environ: dict[str, str] | None = None) -> str | None:
        mode = conversation_mode(environ)
        if mode == "off":
            return None
        key = (host, scope)
        task = self._tasks.get(key)
        running = task is not None and not task.done()
        if not running and time.monotonic() - self._started_at.get(key, -1e9) >= self.min_interval_s:
            self._started_at[key] = time.monotonic()
            self._tasks[key] = asyncio.get_running_loop().create_task(
                self._run(host, scope, mode, environ))
            running = True
        return self._line(key, running)

    async def _run(self, host: str, scope: str, mode: str, environ: dict[str, str] | None) -> None:
        key = (host, scope)
        if self._lock is None:
            self._lock = asyncio.Lock()
        try:
            async with self._lock:
                result = await sync_scope(scope=scope, host=host, state_dir=self._state_dir(),
                                          post=self._post, get=self._get, mode=mode, environ=environ)
            self._last[key] = {"state": "done", "at": time.time(), "result": result}
        except asyncio.CancelledError:
            raise
        except DejaNotFound as exc:
            self._last[key] = {"state": "missing", "at": time.time(), "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a sync problem must never fail a search
            logger.warning("conversation sync for %s failed: %s", scope, exc)
            self._last[key] = {"state": "failed", "at": time.time(), "message": str(exc)}

    def _line(self, key: tuple[str, str], running: bool) -> str | None:
        """Only a problem is worth the model's context; a healthy sync says nothing."""
        last = self._last.get(key)
        if running or last is None:
            return None
        if last["state"] in ("missing", "failed"):
            return f"Conversations sync failed: {last['message'][:200]}"
        result: SyncResult = last["result"]
        if result.errors:
            return f"Conversations sync failed: {result.errors[0][:200]}"
        return None
