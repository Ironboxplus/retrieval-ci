"""Local process control and stdio bridge for the Retrieval MCP daemon.

This module deliberately knows nothing about the 32 Retrieval tools.  It is safe
for every agent's lightweight stdio shim to import: only ``serve`` imports the
full FastMCP tool module in its separate process.  The runtime directory is
per OS user and all mutating files live below it, so stale recovery never touches
an arbitrary process or path outside that namespace.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextvars
import ctypes
import hmac
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import secrets
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit, urlunsplit

try:  # Python 3.10 keeps the supported package floor; 3.11 gained tomllib.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by an isolated import probe.
    import tomli as tomllib


LOG = logging.getLogger("retrieval_mcp.daemon")
_DAEMON_LOG_STREAM: Any | None = None
_WINDOWS_ACL_PRINCIPAL_UNSET = object()
_WINDOWS_ACL_PRINCIPAL: str | None | object = _WINDOWS_ACL_PRINCIPAL_UNSET

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 17871
DEFAULT_RAW_MCP_PORT = 17872
DEFAULT_PLUGIN_BACKEND_URL = "http://10.100.100.111:8000"
DEFAULT_REPOSITORY_BACKEND_URL = "http://127.0.0.1:8000"
MCP_PATH = "/mcp"
HEALTH_PATH = "/health"
SHUTDOWN_PATH = "/shutdown"
STATE_FILE = "daemon.json"
DAEMON_LOCK_DIR = "daemon.lock"
STARTUP_LOCK_DIR = "startup.lock"
LOCK_OWNER_FILE = "owner.json"
STATE_SCHEMA_VERSION = 1
# 2: the raw handshake carries the shim's working directory.  A shim that sends it
# must not reuse a daemon that ignores it — that daemon would keep resolving a
# relative scope against its own directory and answer for the wrong project.
DAEMON_PROTOCOL_SCHEMA_VERSION = 2
STARTUP_TIMEOUT_SECONDS = 20.0
STARTUP_LOCK_WAIT_SECONDS = 25.0
STALE_STARTUP_LOCK_SECONDS = 45.0
HEALTH_TIMEOUT_SECONDS = 2.5
HEALTH_ATTEMPTS = 3
# Startup contenders only need a fast affirmative answer before they decide not
# to take the mutation lock.  A stale/half-started state must not make every
# contender spend the full control-plane retry budget before the owner can
# recover it inside the fence.
STARTUP_REUSE_PROBE_TIMEOUT_SECONDS = 0.25
STOP_TIMEOUT_SECONDS = 10.0

# This is intentionally a small release contract rather than an import of the
# 4k-line tool module.  Every stdio shim needs to compare it before reusing a
# daemon, but must never load the FastMCP registrations just to do so.  The full
# daemon validates its real registrations against this manifest during startup.
TOOL_SURFACE_NAMES = frozenset({
    "search_papers", "search_arxiv", "search_within_paper", "match_document", "match_paper",
    "match_paper_status",
    "get_paper_markdown", "download_paper_pdf", "download_papers", "list_conferences",
    "corpus_stats", "journal_record", "journal_search", "journal_recent", "journal_index_dir",
    "journal_index_status", "delete_journal_docs", "ace_context_aware", "ace_enhance_prompt",
    "ace_smart_generate", "ace_smart_reflect", "ace_playbook", "index_code",
    "index_code_status", "search_code", "find_duplicate_code",
    "delete_code", "index_inventory",
    "list_public_code_sources", "add_public_code_source", "check_public_code_updates",
    "index_public_code_sources", "set_public_code_update_policy", "public_code_job_status",
    "import_public_code_repo",
})


def _tool_name_surface_digest(names: frozenset[str] | set[str]) -> str:
    serialized = json.dumps(sorted(names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


TOOL_SURFACE_NAMES_DIGEST = _tool_name_surface_digest(TOOL_SURFACE_NAMES)


def _canonical_json(value: Any) -> Any:
    """Return a stable JSON-only value without changing schema list semantics."""
    if isinstance(value, dict):
        return {str(key): _canonical_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise DaemonControlError("MCP tool schema contains a non-JSON value")


def _canonical_tool_description(description: str) -> str:
    """Normalize only the non-semantic docstring margin emitted by MCP/Pydantic.

    Python 3.10 with Pydantic 2.13 can preserve a tool docstring's common
    indentation after its first line, while Python 3.13/Pydantic 2.12 emits the
    same text already dedented. ``inspect.cleandoc`` removes only that shared
    margin and trailing blank margin; relative indentation in argument lists and
    examples remains part of the public description contract.
    """
    return inspect.cleandoc(description)


def canonical_tool_surface(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Canonicalize the stable public part of a ``tools/list`` response.

    The singleton's lightweight client must fence a daemon on more than names:
    client caches and generated bindings also rely on the description and JSON
    input schema.  Output/transport implementation fields are deliberately not
    included because the public compatibility contract is name + description +
    inputSchema.
    """
    surface: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        input_schema = getattr(tool, "inputSchema", None)
        if not isinstance(name, str) or not name or name in seen:
            raise DaemonControlError("MCP tools/list has an empty or duplicate name")
        if not isinstance(description, str) or not isinstance(input_schema, dict):
            raise DaemonControlError(f"MCP tools/list has an invalid public schema for {name!r}")
        seen.add(name)
        surface.append({
            "name": name,
            "description": _canonical_tool_description(description),
            "inputSchema": _canonical_json(input_schema),
        })
    return sorted(surface, key=lambda entry: entry["name"])


def _canonical_public_tool_surface_entries(
    tools_or_surface: Sequence[Any],
) -> list[dict[str, Any]]:
    """Canonicalize either SDK tools or an already-serialized tools/list surface.

    The digest helpers are also used by release/contract tests with decoded JSON
    responses.  Normalize that representation through the same description
    path as SDK objects; otherwise a remote Pydantic renderer could bypass the
    cross-runtime normalization merely because its response was deserialized
    first.
    """
    if tools_or_surface and isinstance(tools_or_surface[0], dict) and {
        "name", "description", "inputSchema",
    }.issubset(tools_or_surface[0]):
        decoded_tools = [
            SimpleNamespace(
                name=entry.get("name"),
                description=entry.get("description"),
                inputSchema=entry.get("inputSchema"),
            )
            for entry in tools_or_surface
        ]
        return canonical_tool_surface(decoded_tools)
    return canonical_tool_surface(tools_or_surface)


def canonical_tool_surface_digest(tools_or_surface: Sequence[Any]) -> str:
    """Hash the canonical full ``tools/list`` public schema deterministically."""
    surface = _canonical_public_tool_surface_entries(tools_or_surface)
    serialized = json.dumps(surface, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def tool_surface_field_digests(tools_or_surface: Sequence[Any]) -> dict[str, dict[str, str]]:
    """Return safe per-field fingerprints for a full public ``tools/list`` surface."""
    surface = _canonical_public_tool_surface_entries(tools_or_surface)

    def digest(value: Any) -> str:
        serialized = json.dumps(
            _canonical_json(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return {
        entry["name"]: {
            "description": digest(entry["description"]),
            "inputSchema": digest(entry["inputSchema"]),
        }
        for entry in surface
    }


def _first_tool_surface_diff_path(actual_surface: Sequence[dict[str, Any]]) -> str:
    """Return the first public contract path that differs, never its content."""
    actual = tool_surface_field_digests(actual_surface)
    expected_names = set(TOOL_SURFACE_FIELD_DIGESTS)
    actual_names = set(actual)
    for name in sorted(expected_names - actual_names):
        return f"tools[{name}]"
    for name in sorted(actual_names - expected_names):
        return f"tools[{name}]"
    for name in sorted(expected_names):
        for field in ("description", "inputSchema"):
            if actual[name].get(field) != TOOL_SURFACE_FIELD_DIGESTS[name].get(field):
                return f"tools[{name}].{field}"
    return "tools/list"


# Updated by the full schema parity test whenever a public tool is intentionally
# changed. It remains a literal so the ordinary stdio shim never imports the
# 4k-line FastMCP registration module merely to decide whether reuse is safe.
# Descriptions are normalized with ``inspect.cleandoc`` before hashing, so a
# Pydantic renderer's shared docstring margin cannot fence an equivalent daemon.
TOOL_SURFACE_DIGEST = 'a5c8d1662e5de3b2f1bd12e2d7421f653ed25b5a0f0c2550ceb65db0b33d1a46'

# This mirrors the normalized 34-tool public contract at field granularity. It
# stores only SHA-256 values, so a startup failure can name the first incompatible
# public path without copying descriptions or schema content into a daemon log.
TOOL_SURFACE_FIELD_DIGESTS: dict[str, dict[str, str]] = {
    "ace_context_aware": {"description": "159cf91af3ce3b34e7e5c6d6262923d1e1e90ebd451717daa6989492c7f74dcf", "inputSchema": "dae945cf4046c5ccd97e735775318e5c95223a60866176d13ac74c8b636c1c19"},
    "ace_enhance_prompt": {"description": "a9e2e5fbd14e0d1b9b9c72e04dbad81eaa537d3b2117ad61e1043c181dc83518", "inputSchema": "024e019e7ed408fbb10cec7babb39dddd435106c831ecfc7dded6486d4bfe546"},
    "ace_playbook": {"description": "72f623eb512c322e1465dfb123d03ccee8fd6d409f24bee44370d271481f4a55", "inputSchema": "9bb59473f7650c4197f2ded2cfc6ad16d0636baab3040a83d98792d3924d7fcc"},
    "ace_smart_generate": {"description": "f29687b4547e9fb6e3f2cdac4f625546c2fbb6c3224d24aeeeddb2c834c6b731", "inputSchema": "3bc5fd378a1e9c35068eb02724f1df182cd04cb6d6d7e2a89d2d850e7fde0de3"},
    "ace_smart_reflect": {"description": "70247f5a341d6a1056cbf0008c06d66fce70e83c9f1b254da7dc8e2c727453c0", "inputSchema": "69a233d9528714cc8998ca6f11b3d9539527ad64e14344786c027a1cab84f4f4"},
    "add_public_code_source": {"description": "1fa8f48c98f7fb7c09465c39d4f306d65bcaedbc888d830dcd7eaf5d50858215", "inputSchema": "5506186fdec67551fc0fd0428f4bc1faf5f1f96e7475a79a8c64aaf1647ae4b8"},
    "check_public_code_updates": {"description": "d0c6610ac91a0413f06bc57dbaf000a01c7012a7bd47af26755e60a89f557e57", "inputSchema": "d2a24eec796f7c59a73782b1cfa63e2a46b874e97cf71ca8a76561bfe0b8a85e"},
    "corpus_stats": {"description": "9de5ba1424b90ef70d4ac091834640cc842d9e5fe03d45c1d27fbabcf6d4c440", "inputSchema": "efddc7bd8bbcef73a14eb1ace1ffdaec81e518ef1e13c1e9271d0b8acb694a49"},
    "delete_code": {"description": "85e2b79f64162e280a6f35df9ef3220e7d19196846f3ea739a3e3bc305e02b53", "inputSchema": "7a88742d481c89e6b4ac8ebd6df269cefa3f3eb80accacfc0a4b2507fe175213"},
    "delete_journal_docs": {"description": "9519f06f85335b01de54af00bf56df8f3d3c859ac3ed502eff8cce7865fce490", "inputSchema": "21a0a2542424e177d5ae05e772da6f8726888dad99cb174d2dcbe136dc074cc9"},
    "download_paper_pdf": {"description": "f49788cac8195d61a7f234e8848ff6f63a89f37da9980b564fb070b37a5565f1", "inputSchema": "3c04520c801780f33bbed6abf81c9150dec6abd22f73dfae724d074fda929ce1"},
    "download_papers": {"description": "55835aa00e8c560eee24e32c261f2fa8a5a9fae161b347807f3593de3750f773", "inputSchema": "dfdfa44598021be750138b17c1594d0747e35c589f7b506d8ab6f51b65c2df4c"},
    "find_duplicate_code": {"description": "627e84fb9f79487d91f1dc5b58948a4f69d918e46e4c67e8374f9aaee49eca86", "inputSchema": "c2e57a820aa05c16803f7c7a8ea0fe176677b805712d8f958be4af40be3d09e5"},
    "get_paper_markdown": {"description": "6037380e2cc37f140704a6ff32e367f6be9697adf17a25f7cd9824a63616690a", "inputSchema": "3c04520c801780f33bbed6abf81c9150dec6abd22f73dfae724d074fda929ce1"},
    "index_code": {"description": "03600f332a09e89fab1e2834d4546e5dd0ebf2691fe2d9dc341ba2414e9724d8", "inputSchema": "ee6ae415bbdf74a4eef82a9a11c2c160d917569053d076352f15f5c5ca3e794b"},
    "index_code_status": {"description": "f9e60e4e4251a739f05db66c02eff0ec1c24bafe3eaf36b037e331a8701277ac", "inputSchema": "bc114578d48a2a24e83ffcddea52e26c928129f4b666cac1635072bc6e9c767d"},
    "index_inventory": {"description": "25ffae51c918d5c9e8febb4b345058c4684725007739e3229ba21cab04bcd247", "inputSchema": "527a6ba8df21b10b6ae4f3464ff66c3c6ee6701fb9790136211e35a1b786b122"},
    "index_public_code_sources": {"description": "c61b8a835e64e1871295d9dae77e822ad6b631a627c18511c96d36c196f4bbee", "inputSchema": "d2a24eec796f7c59a73782b1cfa63e2a46b874e97cf71ca8a76561bfe0b8a85e"},
    "journal_index_dir": {"description": "ec3f8687a09b1dd6f453b787bf6e38cac61336ff25dc81c9eec78abbf0202044", "inputSchema": "9ed52c3d871c0885211751331433045be5ec287fd2ec0016c2b6145e63945c16"},
    "journal_index_status": {"description": "7c60b6a474792f11f95cd1355b68ebbe5238e6c779becca5398dd972ed6c694e", "inputSchema": "205ba4d6e177fd73ee0decc8643292df7e1a9c9a4acef7bca686a2725141ff3c"},
    "journal_recent": {"description": "36104dc6c8647e59ffd65f983d3fa4f53eb46a7064344807768cbce041296874", "inputSchema": "1e43234390b5678884e769026673a5c5c37437d8b5455bafe5db854cdb397242"},
    "journal_record": {"description": "fe2e1659bb794868cb1c94d62e27f6d537be5d06564f330eb612786e9d4e1052", "inputSchema": "cc6eb675cd5b73903e7a639a7a0c744a7ff7582ce55b9c21ea190f38ef91301d"},
    "journal_search": {"description": "cbefef896b0da16e71fb0015e1432a4d90ca0e1fe9d4be3ff9260436d4d1e8b1", "inputSchema": "0299393b94385ab8ef478e0043b03b40590c1f361400b7f44353bbe98db6df51"},
    "list_conferences": {"description": "e43b1b42551906d170b96048577ddefc033831ab0b8387c9c24bb0da70b994dc", "inputSchema": "efddc7bd8bbcef73a14eb1ace1ffdaec81e518ef1e13c1e9271d0b8acb694a49"},
    "list_public_code_sources": {"description": "19497f5662fa2aaa1b95b1ce3a1caf000f6481e6fecca5b513eec76181d9857a", "inputSchema": "efddc7bd8bbcef73a14eb1ace1ffdaec81e518ef1e13c1e9271d0b8acb694a49"},
    "match_document": {"description": "e5233119616c509f69f5b6a93dfd1add187e17f974a909821cce3efb82edd81e", "inputSchema": "4e07b41b0df80503526637ca94b65436ee04bb67470d716b6754dc2ccfcbe30a"},
    "match_paper": {"description": "96225104d3ba09e2950b46a793858a6672b2dd26561593dd918b9b2bf8861453", "inputSchema": "b20194242846571ff7bf30c92d8b98db4e2b23e7b7d3859bddc63ff994bab226"},
    "match_paper_status": {"description": "d64a606a832a80ddef9cf027614f723e19c5594e9c9c63c69508594d4d5615fe", "inputSchema": "f729a2e30af43ea722a0480c722904f88f19c515c00a54a799d7db1a7ddf503c"},
    "import_public_code_repo": {"description": "2110cfe2425f7122a4411019596d4953b212a6ad091bd8e1e80aafe14f15b197", "inputSchema": "f6264d6b6d1f2f4ef9a8c922de197ad6a601ecf168931ca57ee8b9782e64c93e"},
    "public_code_job_status": {"description": "2aa90fcae95bac19fc22667cd025be259d597f2dc15bc2b65325bdf962209bfb", "inputSchema": "4e1efe6585eae2a3e4f1c116db5693cb3d556c20c2c9fad79ab39878d6301676"},
    "search_code": {"description": "98bbd8a4561e9fdca80123860b9bcd8095612e5c61df0c1232e2a3fb991e7617", "inputSchema": "732257043079481d69ef710c55e7976231c43fa636170d19d898a2119a600b11"},
    "search_arxiv": {"description": "055b01afc2d6c9310cfe8cc189972a2c77778db8fce882f4ddba585947974e8f", "inputSchema": "1b6355012cee080e5bc062cd978f076224eba35372fce53d1edd8e8fbe0f2097"},
    "search_papers": {"description": "f85d30462faaa0b4885296f027aea42b6bc9e0bb041a7ae39f8ac7ffe1dd7ad8", "inputSchema": "13d70dca40b2d1f28bb5c6fb7e4e0cb401b163fd2606e7c31518f778af7fd76b"},
    "search_within_paper": {"description": "15ba2eeef040521b7d65591c751dc58be07f7c2a09cc11fc198f55db1e030006", "inputSchema": "19ff2d44d3035a7dc77310806928e04a1cc66503827c5cf16aa20da90ff37c8d"},
    "set_public_code_update_policy": {"description": "1b30096b5383d8af64491d497a3445cc065190b5c442b0aac170c35913e01c14", "inputSchema": "0f8bd3c5f9646f03b65adde61483ab1fc8a5ef64c92fe4b18f47e4df04e614b6"},
}


class DaemonControlError(RuntimeError):
    """A safe control-plane operation could not be completed."""


@dataclass(frozen=True)
class BackendProfile:
    """A non-secret backend identity used to fence singleton reuse.

    The daemon is allowed to reuse a process only when the client would send
    requests to the same normalized backend with the same credential.  State
    records fingerprints instead of either value so a state file and daemon log
    cannot disclose a gateway key.
    """

    normalized_url: str
    backend_url_fingerprint: str
    api_key_fingerprint: str
    fingerprint: str

    @property
    def url_fingerprint(self) -> str:
        """Short in-process spelling; persisted state uses the explicit name."""
        return self.backend_url_fingerprint


def _sha256_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _environment_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _normalized_backend_url(raw_url: str) -> str:
    """Canonicalize only the endpoint identity; reject credential-bearing URLs."""
    candidate = raw_url.strip()
    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    if scheme not in {"http", "https"} or not hostname:
        raise DaemonControlError("RETRIEVAL_API_URL must be an absolute http(s) backend URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DaemonControlError(
            "RETRIEVAL_API_URL must not contain credentials, a query, or a fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise DaemonControlError("RETRIEVAL_API_URL contains an invalid port") from exc
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    netloc = f"{host}:{port}" if port is not None else host
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def backend_profile_from_environment(default_url: str) -> BackendProfile:
    """Build the caller's reusable backend identity without retaining its key."""
    raw_url = _environment_value(
        "RETRIEVAL_API_URL",
        "RETRIVAL_API_URL",
        default=default_url,
    )
    assert raw_url is not None
    normalized_url = _normalized_backend_url(raw_url)
    api_key = _environment_value("RETRIEVAL_API_KEY", "RETRIVAL_API_KEY", default="") or ""
    backend_url_fingerprint = _sha256_fingerprint(normalized_url)
    api_key_fingerprint = _sha256_fingerprint(api_key)
    fingerprint = _sha256_fingerprint(
        json.dumps(
            {
                "api_key_fingerprint": api_key_fingerprint,
                "backend_url_fingerprint": backend_url_fingerprint,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return BackendProfile(
        normalized_url=normalized_url,
        backend_url_fingerprint=backend_url_fingerprint,
        api_key_fingerprint=api_key_fingerprint,
        fingerprint=fingerprint,
    )


@dataclass(frozen=True)
class DaemonState:
    pid: int
    instance_token: str
    port: int
    started_at: float
    raw_mcp_port: int | None = None
    package_version: str | None = None
    protocol_schema: int | None = None
    tool_surface_digest: str | None = None
    backend_url_fingerprint: str | None = None
    api_key_fingerprint: str | None = None
    backend_profile_fingerprint: str | None = None

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}{MCP_PATH}"

    @property
    def health_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}{HEALTH_PATH}"

    @property
    def raw_mcp_port_or_default(self) -> int:
        return self.raw_mcp_port if self.raw_mcp_port is not None else _raw_mcp_port()

    @property
    def shutdown_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}{SHUTDOWN_PATH}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "pid": self.pid,
            "instance_token": self.instance_token,
            "port": self.port,
            "url": self.url,
            "started_at": self.started_at,
            "raw_mcp_port": self.raw_mcp_port_or_default,
            "package_version": self.package_version,
            "protocol_schema": self.protocol_schema,
            "tool_surface_digest": self.tool_surface_digest,
            "backend_url_fingerprint": self.backend_url_fingerprint,
            "api_key_fingerprint": self.api_key_fingerprint,
            "backend_profile_fingerprint": self.backend_profile_fingerprint,
        }

    @classmethod
    def from_mapping(cls, raw: object) -> "DaemonState | None":
        if not isinstance(raw, dict):
            return None
        try:
            pid = int(raw["pid"])
            token = str(raw["instance_token"])
            port = int(raw["port"])
            started_at = float(raw.get("started_at", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if pid <= 0 or not token or not (1 <= port <= 65535):
            return None
        package_version = raw.get("package_version")
        if not isinstance(package_version, str) or not package_version:
            package_version = None
        try:
            protocol_schema = int(raw["protocol_schema"])
        except (KeyError, TypeError, ValueError):
            protocol_schema = None
        tool_surface_digest = raw.get("tool_surface_digest")
        if not isinstance(tool_surface_digest, str) or not tool_surface_digest:
            tool_surface_digest = None
        backend_url_fingerprint = raw.get("backend_url_fingerprint")
        if not isinstance(backend_url_fingerprint, str) or not backend_url_fingerprint:
            backend_url_fingerprint = None
        api_key_fingerprint = raw.get("api_key_fingerprint")
        if not isinstance(api_key_fingerprint, str) or not api_key_fingerprint:
            api_key_fingerprint = None
        backend_profile_fingerprint = raw.get("backend_profile_fingerprint")
        if not isinstance(backend_profile_fingerprint, str) or not backend_profile_fingerprint:
            backend_profile_fingerprint = None
        try:
            raw_mcp_port = int(raw.get("raw_mcp_port", _raw_mcp_port()))
        except (TypeError, ValueError):
            raw_mcp_port = _raw_mcp_port()
        if not (1 <= raw_mcp_port <= 65535):
            return None
        # A persisted URL is useful to humans, but never trusted as a network
        # target; all probes are rebuilt from the fixed loopback host and port.
        return cls(
            pid=pid,
            instance_token=token,
            port=port,
            started_at=started_at,
            raw_mcp_port=raw_mcp_port,
            package_version=package_version,
            protocol_schema=protocol_schema,
            tool_surface_digest=tool_surface_digest,
            backend_url_fingerprint=backend_url_fingerprint,
            api_key_fingerprint=api_key_fingerprint,
            backend_profile_fingerprint=backend_profile_fingerprint,
        )


@dataclass(frozen=True)
class _WmiLaunchedDaemon:
    """Minimal Popen-compatible status view for a WMI-created process."""

    pid: int

    def poll(self) -> int | None:
        return None if _pid_exists(self.pid) else 1


def _runtime_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser()
    overridden = os.environ.get("RETRIEVAL_MCP_STATE_DIR")
    if overridden:
        return Path(overridden).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "RetrievalMCP"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "RetrievalMCP"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "retrieval-mcp"


def _is_windows() -> bool:
    return os.name == "nt"


def _is_symlink_or_reparse_point(path: Path) -> bool:
    """Treat Windows junctions as links too, not just POSIX-style symlinks."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    return bool(
        _is_windows()
        and getattr(metadata, "st_file_attributes", 0) & 0x0400  # FILE_ATTRIBUTE_REPARSE_POINT
    )


def _reject_runtime_symlink(path: Path) -> Path:
    """Return an absolute lexical path only when none of its parents is a link."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = candidate
    while True:
        if _is_symlink_or_reparse_point(current):
            raise DaemonControlError(
                "refusing symlink or reparse-point in Retrieval MCP runtime path"
            )
        if current.parent == current:
            break
        current = current.parent
    return candidate


def _current_windows_acl_principal() -> str | None:
    """Resolve the actual token identity, not a possibly stale USERNAME variable."""
    global _WINDOWS_ACL_PRINCIPAL
    if _WINDOWS_ACL_PRINCIPAL is _WINDOWS_ACL_PRINCIPAL_UNSET:
        try:
            result = subprocess.run(
                ["whoami"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            principal = result.stdout.strip() if result.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            principal = ""
        _WINDOWS_ACL_PRINCIPAL = principal or None
    return _WINDOWS_ACL_PRINCIPAL if isinstance(_WINDOWS_ACL_PRINCIPAL, str) else None


def _apply_windows_private_acl(path: Path) -> None:
    """Best-effort owner ACL hardening for the per-user Windows runtime namespace."""
    principal = _current_windows_acl_principal()
    if not principal:
        LOG.warning("Retrieval MCP runtime ACL was not tightened: current Windows identity is unknown")
        return
    try:
        result = subprocess.run(
            [
                "icacls",
                str(path),
                "/grant:r",
                # ``(OI)(CI)`` on a *file* yields an inherit-only ACE and can
                # remove the creator's own read access.  Direct full control is
                # correct for both each concrete state/log file and the runtime
                # directories we explicitly harden before use.  Do not remove
                # inheritance here: an existing lock directory can contain a
                # persisted owner file, and stripping its parent ACL before we
                # grant that nested file would make stale recovery unreadable.
                f"{principal}:F",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            LOG.warning("Retrieval MCP runtime ACL hardening returned icacls=%s", result.returncode)
    except (OSError, subprocess.SubprocessError):
        # ACLs vary across consumer/domain Windows setups.  The directory still
        # remains in the caller's private app-data namespace; fail closed for
        # link traversal, but do not make a local daemon unusable solely because
        # a best-effort ACL tool is unavailable.
        LOG.warning("Retrieval MCP runtime ACL hardening was unavailable")


def _harden_private_file(path: Path) -> None:
    _reject_runtime_symlink(path)
    if _is_windows():
        _apply_windows_private_acl(path)
        return
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        raise DaemonControlError("could not set private Retrieval MCP file permissions") from exc


def _ensure_private_runtime_dir(path: Path) -> Path:
    """Create and harden the user runtime root without following links/junctions."""
    candidate = _reject_runtime_symlink(path)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DaemonControlError("could not create Retrieval MCP runtime directory") from exc
    _reject_runtime_symlink(candidate)
    if _is_windows():
        _apply_windows_private_acl(candidate)
    else:
        try:
            os.chmod(candidate, 0o700)
        except OSError as exc:
            raise DaemonControlError("could not set private Retrieval MCP runtime permissions") from exc
    return candidate


def resolve_runtime_dir(raw: str | None = None) -> Path:
    """Public runtime-dir resolver shared by the thin and full entry points."""
    return _ensure_private_runtime_dir(_runtime_dir(raw))


def _package_version() -> str:
    """Read the installed package version without importing the full tool server."""
    test_override = os.environ.get("RETRIEVAL_MCP_TEST_PACKAGE_VERSION")
    if test_override:
        return test_override
    try:
        return importlib.metadata.version("retrieval-mcp")
    except importlib.metadata.PackageNotFoundError:
        try:
            with open(Path(__file__).with_name("pyproject.toml"), "rb") as handle:
                value = tomllib.load(handle)["project"]["version"]
            return str(value)
        except (KeyError, OSError, tomllib.TOMLDecodeError):
            return "unknown"


def _local_compatibility() -> tuple[str, int, str]:
    return _package_version(), DAEMON_PROTOCOL_SCHEMA_VERSION, TOOL_SURFACE_DIGEST


def _compatibility_mismatches(
    state: "DaemonState",
    backend_profile: BackendProfile | None = None,
) -> list[str]:
    expected_version, expected_protocol, expected_digest = _local_compatibility()
    mismatches: list[str] = []
    if state.package_version != expected_version:
        mismatches.append("package_version")
    if state.protocol_schema != expected_protocol:
        mismatches.append("protocol_schema")
    if state.tool_surface_digest != expected_digest:
        mismatches.append("tool_surface_digest")
    if backend_profile is not None and (
        state.backend_url_fingerprint != backend_profile.backend_url_fingerprint
        or state.api_key_fingerprint != backend_profile.api_key_fingerprint
        or state.backend_profile_fingerprint != backend_profile.fingerprint
    ):
        mismatches.append("backend_profile")
    return mismatches


def _compatibility_reason(
    state: "DaemonState",
    backend_profile: BackendProfile | None = None,
) -> str:
    mismatches = _compatibility_mismatches(state, backend_profile)
    if not mismatches:
        return ""
    return (
        "daemon compatibility mismatch for " + ", ".join(mismatches)
        + "; run retrieval-mcp stop, then retry so the current package can start one daemon"
    )


def redirect_daemon_output_from_environment() -> None:
    """Give an OS-launched daemon a private log before heavy tool imports run.

    Direct ``Popen`` launch already redirects descriptors.  The WMI path has no
    per-process descriptor argument, so it passes a private runtime log path in
    its environment and calls this tiny helper before FastMCP/chonkie imports.
    """
    global _DAEMON_LOG_STREAM
    raw_path = os.environ.get("RETRIEVAL_MCP_DAEMON_LOG")
    if not raw_path or _DAEMON_LOG_STREAM is not None:
        return
    try:
        path = Path(raw_path)
        _DAEMON_LOG_STREAM = _open_private_log(path, text=True)
        sys.stdout = _DAEMON_LOG_STREAM
        sys.stderr = _DAEMON_LOG_STREAM
    except OSError:
        # A daemon must never turn a diagnostics-path problem into unbounded
        # stdout output on the MCP transport; serve uses HTTP, so normal stderr
        # remains an acceptable fallback for this exceptional path.
        return


def _daemon_port() -> int:
    """Return the fixed public port, with a deliberately undocumented test override."""
    raw = os.environ.get("RETRIEVAL_MCP_TEST_PORT")
    if raw is None:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise DaemonControlError("invalid isolated test daemon port") from exc
    if not (1 <= port <= 65535):
        raise DaemonControlError("isolated test daemon port is outside 1..65535")
    return port


def _raw_mcp_port() -> int:
    """Return the private raw JSONL port, with a test-only isolation override."""
    raw = os.environ.get("RETRIEVAL_MCP_TEST_RAW_PORT")
    if raw is None:
        return DEFAULT_RAW_MCP_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise DaemonControlError("invalid isolated raw MCP daemon port") from exc
    if not (1 <= port <= 65535):
        raise DaemonControlError("isolated raw MCP daemon port is outside 1..65535")
    return port


def _state_path(runtime_dir: Path) -> Path:
    return runtime_dir / STATE_FILE


def _daemon_lock_path(runtime_dir: Path) -> Path:
    return runtime_dir / DAEMON_LOCK_DIR


def _startup_lock_path(runtime_dir: Path) -> Path:
    return runtime_dir / STARTUP_LOCK_DIR


def _read_json(path: Path) -> dict[str, Any] | None:
    _reject_runtime_symlink(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_state(path: Path) -> DaemonState | None:
    return DaemonState.from_mapping(_read_json(path))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    parent = _ensure_private_runtime_dir(path.parent)
    path = parent / path.name
    _reject_runtime_symlink(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _harden_private_file(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _open_private_log(path: Path, *, text: bool = False) -> Any:
    """Open a daemon log only after proving it stays inside a private namespace."""
    parent = _ensure_private_runtime_dir(path.parent)
    path = parent / path.name
    _reject_runtime_symlink(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o600)
    try:
        _harden_private_file(path)
        if text:
            return os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)
        return os.fdopen(descriptor, "ab", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _unlink_state_if_token(runtime_dir: Path, token: str) -> None:
    path = _state_path(runtime_dir)
    current = _read_state(path)
    if current is not None and hmac.compare_digest(current.instance_token, token):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _try_claim_lock(path: Path, owner: dict[str, Any]) -> bool:
    parent = _ensure_private_runtime_dir(path.parent)
    path = parent / path.name
    _reject_runtime_symlink(path)
    try:
        path.mkdir()
    except FileExistsError:
        return False
    try:
        _ensure_private_runtime_dir(path)
        _write_json_atomic(path / LOCK_OWNER_FILE, owner)
    except BaseException:
        try:
            path.rmdir()
        except OSError:
            pass
        raise
    return True


def _remove_lock_if_token(path: Path, token: str) -> bool:
    owner_path = path / LOCK_OWNER_FILE
    owner = _read_state(owner_path)
    if owner is None or not hmac.compare_digest(owner.instance_token, token):
        return False
    try:
        owner_path.unlink()
        path.rmdir()
        return True
    except (FileNotFoundError, OSError):
        return False


def _remove_startup_lock(path: Path, token: str | None = None) -> bool:
    """Remove exactly the startup-lock owner that was observed as stale.

    ``unlink(owner.json); rmdir(startup.lock)`` has an otherwise invisible race:
    a different contender can replace the whole directory between the owner read
    and unlink, causing the stale reclaimer to delete the new contender's lock.
    First move the directory to a private sibling, then re-check the owner token
    in that moved directory.  A replacement is restored intact and never removed.
    """
    if not token:
        return False
    owner_path = path / LOCK_OWNER_FILE
    original_owner = _read_json(owner_path)
    original_token = original_owner.get("lock_token") if isinstance(original_owner, dict) else None
    if not isinstance(original_token, str) or not hmac.compare_digest(original_token, token):
        return False

    quarantined = path.with_name(
        f".{path.name}.reclaim.{os.getpid()}.{secrets.token_hex(6)}"
    )
    try:
        # Directory rename is the one filesystem operation that atomically claims
        # the namespace being reclaimed.  The destination is random and absent.
        os.replace(path, quarantined)
    except (FileNotFoundError, OSError):
        return False

    quarantined_owner_path = quarantined / LOCK_OWNER_FILE
    moved_owner = _read_json(quarantined_owner_path)
    moved_token = moved_owner.get("lock_token") if isinstance(moved_owner, dict) else None
    if not isinstance(moved_token, str) or not hmac.compare_digest(moved_token, token):
        # The directory was replaced after our initial observation.  Restore the
        # newer owner if its canonical name is still free; if another contender
        # already claimed that name, leave both namespaces untouched rather than
        # risking deletion of either lock.
        try:
            if not path.exists():
                os.replace(quarantined, path)
        except OSError:
            pass
        return False

    # Windows can transiently reject rmdir immediately after a contender read the
    # owner file.  At this point only the verified stale directory is quarantined,
    # so retrying cannot remove a newer canonical lock.
    for attempt in range(12):
        try:
            quarantined_owner_path.unlink(missing_ok=True)
            quarantined.rmdir()
            return True
        except FileNotFoundError:
            return not quarantined.exists()
        except OSError:
            if attempt + 1 < 12:
                time.sleep(0.01 * (attempt + 1))
                continue
    return False


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        # os.kill(pid, 0) also succeeds for a zombie: the daemon is a child of the shim
        # that started it, and until that shim reaps it an exited daemon still has a PID.
        import psutil  # lazy: only the control paths that ask about a PID need it

        if not psutil.pid_exists(pid):
            return False
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _process_command_line(pid: int) -> str | None:
    """Read only a same-user command line; failure is deliberately inconclusive."""
    if not _pid_exists(pid):
        return ""
    try:
        if os.name == "nt":
            command = (
                f"$p = Get-CimInstance -ClassName Win32_Process -Filter 'ProcessId = {int(pid)}'; "
                "if ($null -ne $p) { [Console]::Out.Write($p.CommandLine) }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                text=True,
                capture_output=True,
                timeout=4,
                check=False,
            )
            return result.stdout if result.returncode == 0 else None
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline.exists():
            return proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            text=True,
            capture_output=True,
            timeout=4,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _identity_matches(state: DaemonState) -> bool | None:
    """Return True/False, or None if process inspection is unavailable.

    A random token is passed in the daemon command line.  Checking it before a
    direct termination makes a recycled PID harmless: we may clean our stale
    runtime record but never signal the unrelated process.
    """
    command_line = _process_command_line(state.pid)
    if command_line is None:
        return None
    if not command_line:
        return False
    marker = f"--instance-token {state.instance_token}"
    return marker in command_line or state.instance_token in command_line


def _token_headers(state: DaemonState) -> dict[str, str]:
    return {"X-Retrieval-MCP-Instance": state.instance_token}


RAW_PROTOCOL = "retrieval-mcp-raw-v1"
RAW_CONTROL = "retrieval-mcp-control-v1"

# The daemon is an ACCOUNT-LEVEL SINGLETON: one process serves every project's shim,
# and its own working directory is just whichever project happened to start it first.
# Tools that accept a relative scope (`scope_dir="."`, `repo_root="."`) must therefore
# resolve it against the CALLING client's directory, which only the shim knows.
# Resolving it in the daemon pointed one project's journal/index at another project's
# namespace — reads returned a foreign repo and `journal_record` MISFILED entries there.
# The shim sends its directory in the authenticated raw handshake; the daemon binds it
# for the life of that session.  A context variable (not a global) is what keeps two
# concurrent sessions from different projects independent.
CLIENT_CWD: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_mcp_client_cwd", default=None
)


def client_working_directory() -> str:
    """Directory of the shim that owns the in-flight MCP session.

    Falls back to the daemon's own directory only when no session is bound (direct
    ``serve`` use, or a shim too old to send one).  The singleton fence rejects a
    version-mismatched pair, so in production the handshake value is always present.
    """
    return CLIENT_CWD.get() or os.getcwd()


def _write_jsonl(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n")
    stream.flush()


def _read_jsonl(stream: Any) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise OSError("local Retrieval MCP raw socket closed before a response")
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("local Retrieval MCP raw response is not an object")
    return value


def _open_raw_mcp_socket(
    state: DaemonState,
    *,
    timeout: float,
    persistent: bool = True,
) -> tuple[socket.socket, Any, Any]:
    connection = socket.create_connection((LOOPBACK_HOST, state.raw_mcp_port_or_default), timeout=timeout)
    reader = connection.makefile("rb")
    writer = connection.makefile("wb")
    try:
        _write_jsonl(writer, {
            "protocol": RAW_PROTOCOL,
            "instance_token": state.instance_token,
            # This shim's directory IS the project the caller is working in.  The
            # singleton on the other end has no other way to learn it: the bridge
            # below copies MCP JSONL byte-for-byte and never parses a request.
            "client_cwd": os.getcwd(),
        })
        response = _read_jsonl(reader)
        if response.get("ready") is not True:
            raise OSError("local Retrieval MCP raw socket rejected this daemon identity")
        connection.settimeout(None if persistent else timeout)
        return connection, reader, writer
    except BaseException:
        try:
            writer.close()
            reader.close()
            connection.close()
        except OSError:
            pass
        raise


def _raw_control(state: DaemonState, action: str, *, timeout: float) -> dict[str, Any]:
    connection = socket.create_connection((LOOPBACK_HOST, state.raw_mcp_port_or_default), timeout=timeout)
    reader = connection.makefile("rb")
    writer = connection.makefile("wb")
    try:
        _write_jsonl(writer, {
            "protocol": RAW_CONTROL,
            "action": action,
            "instance_token": state.instance_token,
        })
        return _read_jsonl(reader)
    finally:
        try:
            writer.close()
            reader.close()
            connection.close()
        except OSError:
            pass


def _probe_health(
    state: DaemonState,
    *,
    timeout: float = HEALTH_TIMEOUT_SECONDS,
    attempts: int = HEALTH_ATTEMPTS,
) -> bool:
    """Authenticate a daemon health response within the caller's bounded budget."""
    for attempt in range(attempts):
        try:
            body = _raw_control(state, "health", timeout=timeout)
            return (
                body.get("ready") is True
                and int(body.get("pid", -1)) == state.pid
                and isinstance(body.get("instance_token"), str)
                and hmac.compare_digest(body["instance_token"], state.instance_token)
            )
        except (OSError, ValueError):
            if attempt + 1 < attempts:
                time.sleep(0.05 * (attempt + 1))
    return False


def _raw_response_for_id(stream: Any, request_id: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = _read_jsonl(stream)
        if str(response.get("id")) == request_id:
            return response
    raise OSError("local Retrieval MCP raw handshake timed out")


def _probe_mcp_handshake(state: DaemonState) -> bool:
    try:
        connection, reader, writer = _open_raw_mcp_socket(
            state,
            timeout=HEALTH_TIMEOUT_SECONDS,
            persistent=False,
        )
        try:
            initialize_id = f"status-init-{secrets.token_hex(8)}"
            _write_jsonl(writer, {
                "jsonrpc": "2.0",
                "id": initialize_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "retrieval-mcp-status", "version": "1"},
                },
            })
            initialize = _raw_response_for_id(reader, initialize_id, timeout=HEALTH_TIMEOUT_SECONDS)
            if not isinstance(initialize.get("result"), dict):
                return False
            _write_jsonl(writer, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            })
            tools_id = f"status-tools-{secrets.token_hex(8)}"
            _write_jsonl(writer, {
                "jsonrpc": "2.0",
                "id": tools_id,
                "method": "tools/list",
                "params": {},
            })
            tools = _raw_response_for_id(reader, tools_id, timeout=HEALTH_TIMEOUT_SECONDS)
            return isinstance(tools.get("result"), dict) and isinstance(tools["result"].get("tools"), list)
        finally:
            try:
                writer.close()
                reader.close()
                connection.close()
            except OSError:
                pass
    except (OSError, ValueError, KeyError):
        return False


def _status_for_state(
    state: DaemonState,
    backend_profile: BackendProfile | None = None,
) -> dict[str, Any]:
    health = _probe_health(state)
    handshake = _probe_mcp_handshake(state) if health else False
    identity = _identity_matches(state)
    expected_version, expected_protocol, expected_digest = _local_compatibility()
    compatible = not _compatibility_mismatches(state, backend_profile)
    running = bool(health and handshake and identity is True and compatible)
    if not compatible:
        reason = _compatibility_reason(state, backend_profile)
    elif running:
        reason = None
    else:
        reason = "unhealthy_or_unverified"
    return {
        "running": running,
        "pid": state.pid,
        "instance_token": state.instance_token,
        "url": state.url,
        "health": health,
        "mcp_handshake": handshake,
        "identity": identity is True,
        "package_version": state.package_version,
        "protocol_schema": state.protocol_schema,
        "tool_surface_digest": state.tool_surface_digest,
        "backend_url_fingerprint": state.backend_url_fingerprint,
        "api_key_fingerprint": state.api_key_fingerprint,
        "backend_profile_fingerprint": state.backend_profile_fingerprint,
        "expected_package_version": expected_version,
        "expected_protocol_schema": expected_protocol,
        "expected_tool_surface_digest": expected_digest,
        "expected_backend_url_fingerprint": (
            backend_profile.backend_url_fingerprint if backend_profile is not None else None
        ),
        "expected_api_key_fingerprint": (
            backend_profile.api_key_fingerprint if backend_profile is not None else None
        ),
        "expected_backend_profile_fingerprint": (
            backend_profile.fingerprint if backend_profile is not None else None
        ),
        "compatible": compatible,
        "reason": reason,
    }


def _inactive_status(
    port: int,
    *,
    reason: str = "not_running",
    backend_profile: BackendProfile | None = None,
) -> dict[str, Any]:
    expected_version, expected_protocol, expected_digest = _local_compatibility()
    return {
        "running": False,
        "pid": None,
        "instance_token": None,
        "url": f"http://{LOOPBACK_HOST}:{port}{MCP_PATH}",
        "health": False,
        "mcp_handshake": False,
        "identity": False,
        "package_version": None,
        "protocol_schema": None,
        "tool_surface_digest": None,
        "backend_url_fingerprint": None,
        "api_key_fingerprint": None,
        "backend_profile_fingerprint": None,
        "expected_package_version": expected_version,
        "expected_protocol_schema": expected_protocol,
        "expected_tool_surface_digest": expected_digest,
        "expected_backend_url_fingerprint": (
            backend_profile.backend_url_fingerprint if backend_profile is not None else None
        ),
        "expected_api_key_fingerprint": (
            backend_profile.api_key_fingerprint if backend_profile is not None else None
        ),
        "expected_backend_profile_fingerprint": (
            backend_profile.fingerprint if backend_profile is not None else None
        ),
        "compatible": False,
        "reason": reason,
    }


def _recover_stale_daemon_lock(runtime_dir: Path) -> bool:
    """Reclaim only a stale lock inside our runtime directory; never kill a PID."""
    path = _daemon_lock_path(runtime_dir)
    if not path.exists():
        return True
    owner = _read_state(path / LOCK_OWNER_FILE)
    if owner is None:
        # A malformed daemon lock has no token identity to compare atomically.
        # Leave it fail-closed rather than reusing the startup-lock reclaimer on
        # a namespace whose owner cannot be proven.
        return False
    if _probe_health(owner):
        return False
    identity = _identity_matches(owner)
    if identity is not False:
        return False
    _unlink_state_if_token(runtime_dir, owner.instance_token)
    return _remove_lock_if_token(path, owner.instance_token)


class _StartupLock:
    """Short-lived atomic lock, with a read-only healthy-daemon escape hatch.

    A first burst of stdio shims has exactly one mutation owner.  Once that
    owner publishes a token-authenticated healthy state, waiters must reuse it
    without lining up behind the lock one health probe at a time.  The callback
    is deliberately read-only: stale-state cleanup and lock removal remain
    inside the owner fence.
    """

    def __init__(
        self,
        runtime_dir: Path,
        healthy_state_probe: Callable[[], DaemonState | None] | None = None,
    ):
        self.path = _startup_lock_path(runtime_dir)
        self.token = secrets.token_urlsafe(18)
        self._held = False
        self._healthy_state_probe = healthy_state_probe
        self.reused_state: DaemonState | None = None

    def __enter__(self) -> "_StartupLock":
        deadline = time.monotonic() + STARTUP_LOCK_WAIT_SECONDS
        owner = {"pid": os.getpid(), "lock_token": self.token, "started_at": time.time()}
        while time.monotonic() < deadline:
            if self._healthy_state_probe is not None:
                reusable = self._healthy_state_probe()
                if reusable is not None:
                    self.reused_state = reusable
                    return self
            if _try_claim_lock(self.path, owner):
                self._held = True
                return self
            existing = _read_json(self.path / LOCK_OWNER_FILE) or {}
            observed_token = existing.get("lock_token") if isinstance(existing, dict) else None
            stale = False
            try:
                existing_pid = int(existing.get("pid", 0))
            except (TypeError, ValueError):
                existing_pid = 0
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError:
                age = 0.0
            if existing_pid:
                stale = not _pid_exists(existing_pid)
            elif age >= STALE_STARTUP_LOCK_SECONDS and isinstance(observed_token, str):
                # The critical section is bounded below.  Owner-less/corrupt
                # directories are deliberately not reclaimed because there is no
                # token to bind a deletion to the stale observation.
                stale = True
            if stale:
                if isinstance(observed_token, str) and _remove_startup_lock(
                    self.path,
                    observed_token,
                ):
                    continue
            time.sleep(0.05)
        raise DaemonControlError("timed out waiting for another Retrieval MCP start operation")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._held:
            _remove_startup_lock(self.path, self.token)
            self._held = False


def _healthy_state_if_available(
    runtime_dir: Path,
    backend_profile: BackendProfile | None = None,
    *,
    fast_reuse_probe: bool = False,
) -> DaemonState | None:
    """Return only a token-authenticated reusable state without mutating files.

    Startup contenders use one short affirmative probe.  The lock owner later
    performs the normal retrying probe before it declares a persisted state
    stale, so an unresponsive stale record cannot amplify into 24 serialized
    multi-second waits.
    """
    state = _read_state(_state_path(runtime_dir))
    if state is None:
        return None
    if fast_reuse_probe:
        healthy = _probe_health(
            state,
            timeout=STARTUP_REUSE_PROBE_TIMEOUT_SECONDS,
            attempts=1,
        )
    else:
        healthy = _probe_health(state)
    if not healthy:
        return None
    compatibility_reason = _compatibility_reason(state, backend_profile)
    if compatibility_reason:
        # A healthy daemon with a different backend profile/version is not stale.
        # Callers must use the explicit token-authenticated ``stop`` boundary.
        raise DaemonControlError(compatibility_reason)
    return state


def _cleanup_stale_state(
    runtime_dir: Path,
    backend_profile: BackendProfile | None = None,
) -> DaemonState | None:
    reusable = _healthy_state_if_available(runtime_dir, backend_profile)
    if reusable is not None:
        return reusable
    state = _read_state(_state_path(runtime_dir))
    if state is None:
        return None
    # ``start``/stdio proxy calls contend during a first burst.  The private
    # loopback health response proves the persisted PID *and* random instance
    # token, so it is a stronger reuse fence than a Windows PowerShell command
    # line query and does not serialize 24 simultaneous shims behind one-second
    # CIM calls.  Direct process termination below still requires the stricter
    # command-line token check to make PID reuse harmless.
    identity = _identity_matches(state)
    if identity is True or identity is None:
        raise DaemonControlError(
            "existing Retrieval MCP daemon PID cannot be proven stale; refusing to start a second daemon"
        )
    _unlink_state_if_token(runtime_dir, state.instance_token)
    return None


def _windows_process_is_in_job() -> bool | None:
    """Return whether this process is job-managed, or None when Windows cannot tell."""
    if os.name != "nt":
        return False
    try:
        in_job = ctypes.c_int()
        kernel32 = ctypes.windll.kernel32
        if not kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None, ctypes.byref(in_job)):
            return None
        return bool(in_job.value)
    except (AttributeError, OSError):
        return None


def _daemon_environment(runtime_dir: Path, log_path: Path) -> dict[str, str]:
    """Pass the daemon's required user environment without exposing it in argv."""
    baseline = {
        "ALL_PROXY", "APPDATA", "COMSPEC", "HOMEDRIVE", "HOMEPATH", "HTTP_PROXY",
        "HTTPS_PROXY", "LOCALAPPDATA", "NO_PROXY", "OS", "PATH", "PATHEXT",
        "PROCESSOR_ARCHITECTURE", "PROGRAMDATA", "PYTHONIOENCODING", "PYTHONPATH",
        "PYTHONUTF8", "SYSTEMROOT", "TEMP", "TMP", "TZ", "USERPROFILE", "WINDIR",
        "all_proxy", "http_proxy", "https_proxy", "no_proxy",
    }
    selected = {
        key: value
        for key, value in os.environ.items()
        if key in baseline or key.startswith("RETRIEVAL_") or key.startswith("RETRIVAL_") or key.startswith("ACE_")
    }
    selected["RETRIEVAL_MCP_STATE_DIR"] = str(runtime_dir)
    selected["RETRIEVAL_MCP_DAEMON_LOG"] = str(log_path)
    return selected


_WINDOWS_WMI_LAUNCHER = """
$raw = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:RETRIEVAL_MCP_WMI_LAUNCH_PAYLOAD))
$payload = $raw | ConvertFrom-Json
$pairs = [string[]]@($payload.environment | ForEach-Object { [string]$_ })
$startup = New-CimInstance -ClassName Win32_ProcessStartup -Property @{
    ShowWindow = [uint16]0
    EnvironmentVariables = $pairs
} -ClientOnly
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = [string]$payload.command_line
    CurrentDirectory = [string]$payload.current_directory
    ProcessStartupInformation = $startup
}
[Console]::Out.Write(($result | ConvertTo-Json -Compress))
""".strip()


def _launch_windows_wmi_daemon(
    command: Sequence[str],
    *,
    runtime_dir: Path,
    log_path: Path,
) -> _WmiLaunchedDaemon:
    """Create a user-owned detached daemon outside an MCP client Job Object.

    ``Win32_Process.Create`` is a Windows OS launcher, not a service manager or
    persistent registration mechanism.  It is only selected for a job-managed
    stdio shim because a regular child would be destroyed with that client's
    kill-on-close Job.  Arguments and environment are passed as an encoded
    transient payload to avoid putting backend keys in a command line.
    """
    environment = _daemon_environment(runtime_dir, log_path)
    payload = {
        "command_line": subprocess.list2cmdline(list(command)),
        "current_directory": os.getcwd(),
        "environment": [f"{key}={value}" for key, value in environment.items()],
    }
    encoded_payload = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    encoded_script = base64.b64encode(_WINDOWS_WMI_LAUNCHER.encode("utf-16le")).decode("ascii")
    launcher_env = dict(os.environ)
    launcher_env["RETRIEVAL_MCP_WMI_LAUNCH_PAYLOAD"] = encoded_payload
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=launcher_env,
            cwd=os.getcwd(),
            creationflags=flags,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DaemonControlError(
            "Windows WMI could not create an independent Retrieval MCP daemon; "
            "refusing a client-owned fallback"
        ) from exc
    try:
        response = json.loads(result.stdout.strip())
        process_id = int(response["ProcessId"])
        return_value = int(response["ReturnValue"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DaemonControlError(
            "Windows WMI returned no valid daemon process identity; refusing a client-owned fallback"
        ) from exc
    if result.returncode != 0 or return_value != 0 or process_id <= 0:
        raise DaemonControlError(
            f"Windows WMI daemon creation failed (Win32_Process.Create={return_value}); "
            "refusing a client-owned fallback"
        )
    return _WmiLaunchedDaemon(process_id)


def _reap_in_background(process: subprocess.Popen) -> None:
    """Collect the daemon's exit status whenever it comes.

    The daemon outlives the shim that started it by design, so that shim usually keeps
    running long after the daemon is told to stop; without a wait() the exited daemon
    would sit as a zombie for as long as the shim lives."""
    threading.Thread(target=process.wait, name="daemon-reaper", daemon=True).start()


def _launch_daemon(runtime_dir: Path, serve_command: Sequence[str], token: str) -> Any:
    runtime_dir = _ensure_private_runtime_dir(runtime_dir)
    log_path = runtime_dir / "daemon.log"
    command = [
        sys.executable,
        *serve_command,
        "serve",
        "--state-dir",
        str(runtime_dir),
        # urlsafe random tokens may begin with ``-``.  Binding the option to
        # its value keeps argparse from mistaking such a token for a new flag.
        f"--instance-token={token}",
    ]
    flags = 0
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        in_job = _windows_process_is_in_job()
        if in_job is not False:
            return _launch_windows_wmi_daemon(command, runtime_dir=runtime_dir, log_path=log_path)
        # An MCP SDK stdio child can be placed in a Windows Job object.  Without
        # breaking away, closing that one shim may terminate every descendant in
        # the job, including the daemon it just started.  The daemon is an
        # account-level singleton, not a child-session resource.
        flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        )
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    with _open_private_log(log_path) as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            cwd=os.getcwd(),
            **kwargs,
        )
    _reap_in_background(process)
    return process


def _ensure_daemon(
    runtime_dir: Path,
    serve_command: Sequence[str],
    backend_profile: BackendProfile | None = None,
) -> tuple[DaemonState, bool]:
    runtime_dir = _ensure_private_runtime_dir(runtime_dir)
    def reusable_state() -> DaemonState | None:
        return _healthy_state_if_available(
            runtime_dir,
            backend_profile,
            fast_reuse_probe=True,
        )

    # Do not make already healthy shims contend on the mutation lock.  This is
    # also the fast path after another process completes a first singleton boot.
    running = reusable_state()
    if running is not None:
        return running, True

    with _StartupLock(runtime_dir, healthy_state_probe=reusable_state) as startup_lock:
        if startup_lock.reused_state is not None:
            return startup_lock.reused_state, True
        running = (
            _cleanup_stale_state(runtime_dir, backend_profile)
            if backend_profile is not None
            else _cleanup_stale_state(runtime_dir)
        )
        if running is not None:
            return running, True
        if not _recover_stale_daemon_lock(runtime_dir):
            # A lock can belong to a daemon that is in its final startup phase.
            # Re-check state once before reporting a conflict.
            current = (
                _cleanup_stale_state(runtime_dir, backend_profile)
                if backend_profile is not None
                else _cleanup_stale_state(runtime_dir)
            )
            if current is not None:
                return current, True
            raise DaemonControlError("another Retrieval MCP daemon owns the runtime lock")

        port = _daemon_port()
        token = secrets.token_urlsafe(32)
        process = _launch_daemon(runtime_dir, serve_command, token)
        expected = DaemonState(
            process.pid,
            token,
            port,
            time.time(),
            backend_url_fingerprint=(
                backend_profile.backend_url_fingerprint if backend_profile is not None else None
            ),
            api_key_fingerprint=(
                backend_profile.api_key_fingerprint if backend_profile is not None else None
            ),
            backend_profile_fingerprint=(
                backend_profile.fingerprint if backend_profile is not None else None
            ),
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            state = _read_state(_state_path(runtime_dir))
            if state is not None and hmac.compare_digest(state.instance_token, token):
                status = _status_for_state(state, backend_profile)
                if status["running"]:
                    return state, False
            if process.poll() is not None:
                break
            time.sleep(0.1)

        # A token observed in a command line is not an atomic process handle:
        # that PID can exit and be recycled before a later signal/TerminateProcess.
        # The raw control channel is not healthy yet, so there is no authenticated
        # shutdown path either.  Preserve safety over cleanup and make the caller
        # diagnose the unverified launch rather than killing by a stale number.
        raise DaemonControlError(
            "startup timeout left daemon unverified; refusing unsafe PID termination"
        )


def _wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def _request_shutdown(state: DaemonState) -> bool:
    try:
        body = _raw_control(state, "shutdown", timeout=HEALTH_TIMEOUT_SECONDS)
        return (
            body.get("accepted") is True
            and int(body.get("pid", -1)) == state.pid
            and isinstance(body.get("instance_token"), str)
            and hmac.compare_digest(body["instance_token"], state.instance_token)
        )
    except (OSError, ValueError):
        return False


def _stop_daemon(runtime_dir: Path) -> tuple[int, dict[str, Any]]:
    state = _read_state(_state_path(runtime_dir))
    if state is None:
        return 0, {"stopped": False, **_inactive_status(_daemon_port())}

    if _probe_health(state) and _request_shutdown(state):
        if _wait_for_exit(state.pid, STOP_TIMEOUT_SECONDS):
            _unlink_state_if_token(runtime_dir, state.instance_token)
            _remove_lock_if_token(_daemon_lock_path(runtime_dir), state.instance_token)
            return 0, {"stopped": True, "pid": state.pid, "instance_token": state.instance_token}
        return 1, {
            "stopped": False,
            "pid": state.pid,
            "instance_token": state.instance_token,
            "reason": (
                "authenticated shutdown accepted but daemon did not exit; "
                "refusing unsafe PID termination"
            ),
        }

    # Do not turn a best-effort command-line check into a PID kill.  On both
    # Windows and POSIX it leaves a TOCTOU interval in which the number can refer
    # to a different process.  Only the raw token-authenticated shutdown channel
    # may request daemon termination.  A demonstrably stale state record itself
    # is safe to clean, but no ambient process is ever signalled.
    identity = _identity_matches(state)
    if identity is False:
        _unlink_state_if_token(runtime_dir, state.instance_token)
        _remove_lock_if_token(_daemon_lock_path(runtime_dir), state.instance_token)
        reason = (
            "identity mismatch; authenticated shutdown unavailable; "
            "stale state cleared without process termination"
        )
    else:
        reason = "authenticated shutdown unavailable; refusing unsafe PID termination"
    return 3, {
        "stopped": False,
        "pid": state.pid,
        "instance_token": state.instance_token,
        "reason": reason,
    }


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _record_test_shim() -> None:
    """Test-only process evidence; production has no shim bookkeeping overhead."""
    directory = os.environ.get("RETRIEVAL_MCP_TEST_SHIM_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": os.getpid(),
        "full_server_loaded": "retrieval_mcp" in sys.modules or "mcp_server.server" in sys.modules,
        "mcp_sdk_loaded": any(name == "mcp" or name.startswith("mcp.") for name in sys.modules),
        "httpx_loaded": "httpx" in sys.modules or any(name.startswith("httpx.") for name in sys.modules),
        "anyio_loaded": "anyio" in sys.modules or any(name.startswith("anyio.") for name in sys.modules),
    }
    _write_json_atomic(path / f"shim-{os.getpid()}.json", record)


def _bridge_stdio_to_raw(state: DaemonState) -> None:
    """Copy MCP JSONL between stdio and the daemon's private raw socket.

    The command-side process intentionally does not parse MCP messages or import
    the MCP SDK, HTTP client, or async runtime.  It authenticates one local
    socket once, then copies the existing stdio JSONL protocol byte-for-byte.
    Full framing, sessions, and tool registration remain in the singleton.
    """
    connection, reader, writer = _open_raw_mcp_socket(
        state,
        timeout=STARTUP_TIMEOUT_SECONDS,
        persistent=True,
    )
    write_lock = threading.Lock()
    remote_finished = threading.Event()

    def forward_remote_to_stdout() -> None:
        try:
            while True:
                line = reader.readline()
                if not line:
                    return
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        except (OSError, ValueError):
            # A daemon/session close is the normal peer-side EOF boundary.  The
            # caller's input loop will subsequently observe the pipe close.
            return
        finally:
            remote_finished.set()

    # Capture this only after the raw token handshake.  It is test evidence that
    # the ordinary command path stayed independent of FastMCP and its SDK.
    _record_test_shim()
    remote_thread = threading.Thread(
        target=forward_remote_to_stdout,
        name="retrieval-mcp-stdio-raw-output",
        daemon=True,
    )
    remote_thread.start()
    try:
        for line in sys.stdin.buffer:
            if remote_finished.is_set():
                break
            with write_lock:
                writer.write(line)
                writer.flush()
    finally:
        # EOF closes only this raw MCP session.  It never asks the singleton to
        # exit, which keeps a WMI-detached Windows daemon alive after SDK cleanup.
        try:
            connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        remote_thread.join(timeout=2.0)
        try:
            writer.close()
            reader.close()
            connection.close()
        except OSError:
            pass


def run_stdio_proxy(
    runtime_dir: Path,
    serve_command: Sequence[str],
    backend_default_url: str = DEFAULT_PLUGIN_BACKEND_URL,
) -> int:
    """Autostart/reuse the singleton, then expose its MCP endpoint over stdio."""
    try:
        backend_profile = backend_profile_from_environment(backend_default_url)
        state, _ = _ensure_daemon(runtime_dir, serve_command, backend_profile)
        _bridge_stdio_to_raw(state)
        return 0
    except DaemonControlError as exc:
        LOG.error("Retrieval MCP stdio proxy could not ensure local daemon: %s", exc)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        LOG.error("Retrieval MCP stdio proxy failed: %s", exc)
        return 1


def _parse_control_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retrieval-mcp",
        description="Manage the local singleton Retrieval MCP daemon.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "status", "stop"):
        command = subparsers.add_parser(name)
        command.add_argument("--state-dir", help="per-user runtime directory override")
    return parser.parse_args(list(argv))


def run_thin_cli(
    argv: Sequence[str],
    *,
    serve_command: Sequence[str],
    backend_default_url: str = DEFAULT_PLUGIN_BACKEND_URL,
) -> int:
    """Run commands that must never import the full Retrieval tool module."""
    if not argv:
        return run_stdio_proxy(
            _runtime_dir(None),
            serve_command,
            backend_default_url,
        )
    parsed = _parse_control_args(argv)
    runtime_dir = _ensure_private_runtime_dir(_runtime_dir(parsed.state_dir))
    # ``stop`` authenticates the persisted daemon token and must remain usable
    # even if a caller has since supplied an invalid or different backend URL.
    # It is the explicit migration boundary for a profile mismatch.
    backend_profile = (
        backend_profile_from_environment(backend_default_url)
        if parsed.command in {"start", "status"}
        else None
    )
    if parsed.command == "start":
        try:
            state, reused = _ensure_daemon(runtime_dir, serve_command, backend_profile)
            payload = _status_for_state(state, backend_profile)
            payload["action"] = "already_running" if reused else "started"
            _emit(payload)
            return 0
        except DaemonControlError as exc:
            _emit({
                "action": "failed",
                **_inactive_status(
                    _daemon_port(),
                    reason=str(exc),
                    backend_profile=backend_profile,
                ),
            })
            return 1
    if parsed.command == "status":
        state = _read_state(_state_path(runtime_dir))
        _emit(
            _status_for_state(state, backend_profile)
            if state is not None
            else _inactive_status(_daemon_port(), backend_profile=backend_profile)
        )
        return 0
    if parsed.command == "stop":
        code, payload = _stop_daemon(runtime_dir)
        _emit(payload)
        return code
    raise AssertionError(f"unhandled control command {parsed.command!r}")


async def _write_socket_jsonl(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    """Serialize exactly one daemon-side JSONL control or MCP envelope."""
    writer.write(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n")
    await writer.drain()


def _raw_ready_payload(state: DaemonState) -> dict[str, Any]:
    """One token-authenticated health identity used by status and socket clients."""
    return {
        "ready": True,
        "pid": state.pid,
        "instance_token": state.instance_token,
        "url": state.url,
        "raw_mcp_port": state.raw_mcp_port_or_default,
        "package_version": state.package_version,
        "protocol_schema": state.protocol_schema,
        "tool_surface_digest": state.tool_surface_digest,
        "backend_url_fingerprint": state.backend_url_fingerprint,
        "api_key_fingerprint": state.api_key_fingerprint,
        "backend_profile_fingerprint": state.backend_profile_fingerprint,
    }


async def _serve_raw_mcp_session(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    mcp: Any,
) -> None:
    """Adapt one authenticated TCP JSONL session to the SDK's low-level server.

    This is deliberately the same parse/serialize boundary as the official
    ``mcp.server.stdio`` transport: the socket layer knows no MCP method names,
    lifecycle state, or tool semantics.  Those remain in ``Server.run``.
    """
    import anyio
    import mcp.types as mcp_types
    from mcp.shared.message import SessionMessage

    incoming_send, incoming_receive = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    outgoing_send, outgoing_receive = anyio.create_memory_object_stream[SessionMessage](0)

    async def socket_reader() -> None:
        try:
            async with incoming_send:
                while True:
                    line = await reader.readline()
                    if not line:
                        return
                    try:
                        message = mcp_types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:  # SDK turns parse errors into protocol diagnostics.
                        await incoming_send.send(exc)
                    else:
                        await incoming_send.send(SessionMessage(message))
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    async def socket_writer() -> None:
        try:
            async with outgoing_receive:
                async for session_message in outgoing_receive:
                    raw = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    writer.write(raw.encode("utf-8") + b"\n")
                    await writer.drain()
        except (ConnectionError, OSError, anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(socket_reader)
        task_group.start_soon(socket_writer)
        try:
            await mcp._mcp_server.run(  # noqa: SLF001 - FastMCP's documented low-level transport server.
                incoming_receive,
                outgoing_send,
                mcp._mcp_server.create_initialization_options(),  # noqa: SLF001
            )
        finally:
            await outgoing_send.aclose()
            # EOF on the local stdio side closes the corresponding raw socket
            # session, not the daemon.  Cancellation only tears down this pair.
            task_group.cancel_scope.cancel()


async def _handle_raw_mcp_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    mcp: Any,
    state: DaemonState,
    server_holder: dict[str, Any],
) -> None:
    """Authenticate a raw bridge or control frame before touching the SDK."""
    try:
        first_line = await asyncio.wait_for(reader.readline(), timeout=HEALTH_TIMEOUT_SECONDS)
        if not first_line:
            return
        try:
            envelope = json.loads(first_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            await _write_socket_jsonl(writer, {"ready": False})
            return
        if not isinstance(envelope, dict):
            await _write_socket_jsonl(writer, {"ready": False})
            return
        supplied = envelope.get("instance_token")
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, state.instance_token):
            await _write_socket_jsonl(writer, {"ready": False})
            return

        protocol = envelope.get("protocol")
        if protocol == RAW_CONTROL:
            action = envelope.get("action")
            if action == "health":
                if server_holder.get("http_ready") is True:
                    await _write_socket_jsonl(writer, _raw_ready_payload(state))
                else:
                    await _write_socket_jsonl(writer, {"ready": False})
                return
            if action == "shutdown":
                server = server_holder.get("server")
                if server is None:
                    await _write_socket_jsonl(writer, {"accepted": False})
                else:
                    request_raw_shutdown = server_holder.get("request_raw_shutdown")
                    if callable(request_raw_shutdown):
                        request_raw_shutdown()
                    server.should_exit = True
                    await _write_socket_jsonl(writer, {
                        "accepted": True,
                        "pid": state.pid,
                        "instance_token": state.instance_token,
                    })
                return
            await _write_socket_jsonl(writer, {"accepted": False})
            return

        if protocol != RAW_PROTOCOL or server_holder.get("http_ready") is not True:
            await _write_socket_jsonl(writer, {"ready": False})
            return
        await _write_socket_jsonl(writer, {"ready": True})
        # Bind the caller's directory before the SDK runs a single tool.  Every
        # request task the session spawns inherits a copy of this context, so
        # concurrent sessions from different projects each keep their own scope.
        supplied_cwd = envelope.get("client_cwd")
        bound = CLIENT_CWD.set(supplied_cwd if isinstance(supplied_cwd, str) and supplied_cwd else None)
        try:
            await _serve_raw_mcp_session(reader, writer, mcp)
        finally:
            CLIENT_CWD.reset(bound)
    except (ConnectionError, OSError, asyncio.TimeoutError):
        # Peer disconnect and incomplete local setup are expected socket events;
        # do not log untrusted request contents into the daemon diagnostics.
        return
    except Exception:
        LOG.exception("Retrieval MCP raw session failed")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


def serve_daemon(
    mcp: Any,
    *,
    runtime_dir: Path,
    instance_token: str,
    backend_default_url: str = DEFAULT_PLUGIN_BACKEND_URL,
) -> int:
    """Run the full FastMCP tool server as the one authenticated local daemon."""
    if len(instance_token) < 16:
        raise DaemonControlError("serve requires the internal random instance token")
    runtime_dir = _ensure_private_runtime_dir(runtime_dir)
    registered_tools = asyncio.run(mcp.list_tools())
    registered_names = frozenset(tool.name for tool in registered_tools)
    if _tool_name_surface_digest(registered_names) != TOOL_SURFACE_NAMES_DIGEST:
        missing = sorted(TOOL_SURFACE_NAMES - registered_names)
        unexpected = sorted(registered_names - TOOL_SURFACE_NAMES)
        raise DaemonControlError(
            "Retrieval MCP tool manifest drift; refusing to advertise an incompatible daemon "
            f"(missing={missing}, unexpected={unexpected})"
        )
    registered_surface = canonical_tool_surface(registered_tools)
    registered_digest = canonical_tool_surface_digest(registered_surface)
    if registered_digest != TOOL_SURFACE_DIGEST:
        first_diff_path = _first_tool_surface_diff_path(registered_surface)
        raise DaemonControlError(
            "Retrieval MCP full tools/list schema drift; refusing to advertise an incompatible daemon "
            f"(expected_digest={TOOL_SURFACE_DIGEST}, actual_digest={registered_digest}, "
            f"first_diff_path={first_diff_path})"
        )
    port = _daemon_port()
    package_version, protocol_schema, tool_surface_digest = _local_compatibility()
    backend_profile = backend_profile_from_environment(backend_default_url)
    state = DaemonState(
        os.getpid(),
        instance_token,
        port,
        time.time(),
        raw_mcp_port=_raw_mcp_port(),
        package_version=package_version,
        protocol_schema=protocol_schema,
        tool_surface_digest=tool_surface_digest,
        backend_url_fingerprint=backend_profile.backend_url_fingerprint,
        api_key_fingerprint=backend_profile.api_key_fingerprint,
        backend_profile_fingerprint=backend_profile.fingerprint,
    )
    lock_path = _daemon_lock_path(runtime_dir)
    if not _try_claim_lock(lock_path, state.as_dict()):
        if not _recover_stale_daemon_lock(runtime_dir) or not _try_claim_lock(lock_path, state.as_dict()):
            raise DaemonControlError("another Retrieval MCP daemon already owns this runtime directory")
    _write_json_atomic(_state_path(runtime_dir), state.as_dict())

    try:
        from starlette.requests import Request
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        import uvicorn

        # FastMCP v1 creates its streamable HTTP Starlette app lazily.  Register
        # the private loopback routes before doing so; they are authenticated by
        # an unguessable instance token kept in the per-user state file.
        mcp.settings.host = LOOPBACK_HOST
        mcp.settings.port = port
        mcp.settings.streamable_http_path = MCP_PATH
        server_holder: dict[str, Any] = {"http_ready": False}

        def authorization_failure_status(request: Request) -> int | None:
            supplied = request.headers.get("X-Retrieval-MCP-Instance", "")
            if not supplied:
                return 401
            if not hmac.compare_digest(supplied, state.instance_token):
                return 403
            return None

        @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
        async def daemon_health(request: Request) -> JSONResponse:
            if (status_code := authorization_failure_status(request)) is not None:
                return JSONResponse({"ready": False}, status_code=status_code)
            return JSONResponse(_raw_ready_payload(state))

        @mcp.custom_route(SHUTDOWN_PATH, methods=["POST"], include_in_schema=False)
        async def daemon_shutdown(request: Request) -> JSONResponse:
            if (status_code := authorization_failure_status(request)) is not None:
                return JSONResponse({"accepted": False}, status_code=status_code)
            server = server_holder.get("server")
            request_raw_shutdown = server_holder.get("request_raw_shutdown")
            if callable(request_raw_shutdown):
                request_raw_shutdown()
            if server is not None:
                server.should_exit = True
            return JSONResponse({
                "accepted": True,
                "pid": state.pid,
                "instance_token": state.instance_token,
            }, status_code=202)

        app = mcp.streamable_http_app()

        async def require_instance_token_for_mcp(request: Request, call_next: Any) -> Any:
            """Keep the compatibility HTTP endpoint private to this state identity.

            The normal command-mode path uses the separately authenticated raw
            bridge.  HTTP remains for diagnostics and compatibility, but must not
            turn the per-user loopback daemon into an unauthenticated local API.
            """
            if request.url.path == MCP_PATH:
                if (status_code := authorization_failure_status(request)) is not None:
                    return JSONResponse({"error": "instance_token_required"}, status_code=status_code)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=require_instance_token_for_mcp)

        config = uvicorn.Config(app, host=LOOPBACK_HOST, port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        server_holder["server"] = server

        async def serve_http_and_raw_bridge() -> None:
            raw_connection_tasks: set[asyncio.Task[Any]] = set()
            raw_connection_writers: set[asyncio.StreamWriter] = set()
            raw_shutdown_requested = asyncio.Event()

            def request_raw_shutdown() -> None:
                raw_shutdown_requested.set()

            server_holder["request_raw_shutdown"] = request_raw_shutdown

            async def handle_raw_connection(
                reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter,
            ) -> None:
                current = asyncio.current_task()
                if current is not None:
                    raw_connection_tasks.add(current)
                raw_connection_writers.add(writer)
                try:
                    await _handle_raw_mcp_connection(
                        reader,
                        writer,
                        mcp=mcp,
                        state=state,
                        server_holder=server_holder,
                    )
                finally:
                    raw_connection_writers.discard(writer)
                    if current is not None:
                        raw_connection_tasks.discard(current)

            raw_server = await asyncio.start_server(
                handle_raw_connection,
                host=LOOPBACK_HOST,
                port=state.raw_mcp_port_or_default,
                backlog=128,
                # The raw bridge carries the existing stdio JSONL records.  Keep
                # a generous local limit rather than silently truncating a valid
                # tool argument that the SDK stdio transport would accept.
                limit=16 * 1024 * 1024,
            )
            server_task = asyncio.create_task(server.serve(), name="retrieval-mcp-http")

            async def close_raw_sessions_after_shutdown_request() -> None:
                await raw_shutdown_requested.wait()
                # Flush the token-authenticated shutdown acknowledgement before
                # closing its control socket, then cancel every open MCP session.
                # FastMCP's HTTP lifespan waits for low-level server runs, so
                # leaving stdio/raw sessions open here would make ``stop`` hang
                # until every client happens to exit.
                await asyncio.sleep(0.05)
                raw_server.close()
                for writer in list(raw_connection_writers):
                    writer.close()
                pending = [task for task in raw_connection_tasks if not task.done()]
                for task in pending:
                    task.cancel()

            raw_shutdown_task = asyncio.create_task(
                close_raw_sessions_after_shutdown_request(),
                name="retrieval-mcp-close-raw-sessions",
            )
            try:
                # ``start`` succeeds only once the public compatibility endpoint
                # and private bridge are both live.  Raw health returns false
                # during this short interval, so callers naturally retry.
                while not server.started:
                    if server_task.done():
                        await server_task
                        return
                    await asyncio.sleep(0.01)
                server_holder["http_ready"] = True
                await server_task
            finally:
                server.should_exit = True
                server_holder["http_ready"] = False
                raw_shutdown_requested.set()
                raw_server.close()
                await raw_server.wait_closed()
                pending = [task for task in raw_connection_tasks if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if not raw_shutdown_task.done():
                    raw_shutdown_task.cancel()
                await asyncio.gather(raw_shutdown_task, return_exceptions=True)

        asyncio.run(serve_http_and_raw_bridge())
        return 0
    finally:
        _unlink_state_if_token(runtime_dir, state.instance_token)
        _remove_lock_if_token(lock_path, state.instance_token)
