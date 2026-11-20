"""Claude Code / Codex conversation sync: real deja binary, real transcripts, no mocks.

deja runs against temp transcript roots (``DEJA_CLAUDE_ROOT``/``DEJA_CODEX_ROOT``) and a
temp ``DEJA_INDEX_DIR``, so the developer's own ``~/.claude`` and ``~/.codex`` are never
indexed.  Upload behaviour against a real backend lives in
``test_mcp_conversations_real_backend.py`` (opt-in).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from mcp_plugin import retrieval_mcp_conversations as conv
from tests.conversation_transcripts import ClaudeSession, CodexRollout, claude_dir

def _test_deja() -> str | None:
    """An explicit RETRIEVAL_DEJA_BIN, else the plugin's own pinned install (downloaded
    once into the ignored pytest cache, exactly as the plugin installs it)."""
    try:
        return conv.resolve_deja(Path(__file__).resolve().parents[1] / ".pytest_cache" / "deja")
    except conv.DejaNotFound:
        return None


DEJA = _test_deja()
FIXTURES = Path(__file__).parent / "fixtures" / "conversations"
needs_deja = pytest.mark.skipif(DEJA is None, reason="pinned deja-vu could not be installed (offline?)")

SCOPE = "C:/work/demo/proj"  # deja project "demo/proj"
SCOPE_CWD = r"C:\work\demo\proj"


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.claude_root = root / "claude" / "projects"
        self.codex_root = root / "codex"
        self.state_dir = root / "state"
        self.claude_root.mkdir(parents=True)
        self.codex_root.mkdir(parents=True)
        self.state_dir.mkdir()

    def deja(self) -> conv.Deja:
        return conv.Deja(DEJA, conv.deja_environment(self.state_dir, DEJA))

    def state(self, scope: str = SCOPE) -> conv.ConversationState:
        return conv.ConversationState(self.state_dir, "testhost", scope)

    def plan(self, mode: str = "main", scope: str = SCOPE, deja: conv.Deja | None = None):
        errors: list[str] = []
        plans, seen = conv.plan_scope_sync(
            deja or self.deja(), conv.deja_project_name(scope), self.state(scope), mode, errors)
        assert errors == []
        return plans, seen

    def commit(self, plans, scope: str = SCOPE) -> None:
        state = self.state(scope)
        for plan in plans:
            state.commit(plan)
        state.save()


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    box = Sandbox(tmp_path)
    monkeypatch.setenv("DEJA_CLAUDE_ROOT", str(box.claude_root))
    monkeypatch.setenv("DEJA_CODEX_ROOT", str(box.codex_root))
    return box


class CountingDeja(conv.Deja):
    """The real deja, plus a count of ``show`` page reads (instrumentation, not a stub)."""

    shows = 0

    def show_page(self, harness, session_id, offset):
        type(self).shows += 1
        return super().show_page(harness, session_id, offset)


# ------------------------------------------------------------------ pure: config / names

def test_mode_switch():
    assert conv.conversation_mode({}) == "off"
    assert conv.conversation_mode({conv.MODE_ENV: "0"}) == "off"
    assert conv.conversation_mode({conv.MODE_ENV: "1"}) == "main"
    assert conv.conversation_mode({conv.MODE_ENV: "agents"}) == "agents"
    assert conv.conversation_mode({conv.MODE_ENV: "AGENTS"}) == "agents"


def test_deja_binary_explicit_override_must_exist(tmp_path):
    fake = tmp_path / "deja.exe"
    assert conv.deja_binary({conv.DEJA_BIN_ENV: str(fake), "PATH": ""}) is None
    fake.write_bytes(b"")
    assert conv.deja_binary({conv.DEJA_BIN_ENV: str(fake), "PATH": ""}) == str(fake)


@pytest.mark.parametrize("scope,expected", [
    ("C:/Users/Arc/prd/retrival", "prd/retrival"),
    ("c:\\Users\\Arc\\Prd\\retrival", "Prd/retrival"),
    ("C:/prd", "prd"),          # second-last segment is a drive
    ("C:/", "C:"),              # drive root is named after the drive
    ("D:\\x\\y\\Z", "y/Z"),
    ("/home/u/a/b", "a/b"),
    ("/x", "x"),
    ("", None),
    ("/", None),
])
def test_scope_to_deja_project_name(scope, expected):
    assert conv.deja_project_name(scope) == expected


@needs_deja
def test_project_name_rule_matches_what_real_deja_reports(sandbox):
    """The mapping is a mirror of deja's rule; prove it against deja, drive root included."""
    cwds = {r"C:\work\demo\proj": "a", r"C:\prd": "b", "C:\\": "c", "/home/u/a/b": "d",
            r"D:\x\y\Z": "e", r"C:\Users\Arc\a.b\c_d": "f"}
    for cwd, sid in cwds.items():
        ClaudeSession(sandbox.claude_root, cwd, sid).user("hello").assistant("hi")
    for cwd, sid in cwds.items():
        name = conv.deja_project_name(cwd.replace("\\", "/"))
        found = sandbox.deja().list_sessions(name)
        assert [s["id"] for s in found if s["id"] == sid] == [sid], (cwd, name, found)


def test_deja_environment_isolates_index_and_forces_offline(tmp_path):
    env = conv.deja_environment(tmp_path, binary=str(tmp_path / "bin" / "deja.exe"),
                                base={"PATH": "/usr/bin", "DEJA_INCLUDE_SUBAGENTS": "0"})
    assert env["DEJA_OFFLINE"] == "1" and env["DEJA_TUI"] == "0"
    assert env["DEJA_STORES"] == "claude,codex"
    assert env["DEJA_INCLUDE_SUBAGENTS"] == "1"
    assert Path(env["DEJA_INDEX_DIR"]) == tmp_path / "deja" / "index.db"
    # The zstd shim goes LAST: a real zstd on PATH still wins; deja's own dir is not added.
    assert env["PATH"].split(os.pathsep) == ["/usr/bin", str(tmp_path / "deja" / "zstd-shim")]


def _zip_with(name: str, data: bytes) -> bytes:
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, data)
        zf.writestr("README.md", b"readme")
    return buf.getvalue()


def test_install_deja_archive_refuses_a_hash_mismatch(tmp_path):
    archive = _zip_with("deja.exe", b"binary")
    target = tmp_path / "bin" / "deja.exe"
    with pytest.raises(conv.DejaNotFound, match="sha256"):
        conv.install_deja_archive(archive, expected_sha256="0" * 64, system="windows", target=target)
    assert not target.exists()
    conv.install_deja_archive(archive, expected_sha256=hashlib.sha256(archive).hexdigest(),
                              system="windows", target=target)
    assert target.read_bytes() == b"binary"


def test_install_deja_archive_reads_the_tarball_layout(tmp_path):
    import io, tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("LICENSE", b"mit"), ("deja", b"elf")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()
    target = conv.install_deja_archive(archive, expected_sha256=hashlib.sha256(archive).hexdigest(),
                                       system="linux", target=tmp_path / "deja")
    assert target.read_bytes() == b"elf" and os.access(target, os.X_OK)


def test_every_supported_platform_has_a_pinned_hash():
    assert set(conv.DEJA_RELEASE_SHA256) == {(o, a) for o in ("darwin", "linux", "windows")
                                            for a in ("amd64", "arm64")}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in conv.DEJA_RELEASE_SHA256.values())


@needs_deja
def test_pinned_install_is_the_release_version():
    import subprocess
    if os.environ.get(conv.DEJA_BIN_ENV):
        pytest.skip("an explicit RETRIEVAL_DEJA_BIN is under test")
    assert Path(DEJA).parent.name == conv.DEJA_VERSION
    out = subprocess.run([DEJA, "version"], capture_output=True, text=True, timeout=60).stdout
    assert conv.DEJA_VERSION in out


@needs_deja
def test_compressed_codex_rollout_is_read_through_the_zstd_shim(sandbox, tmp_path):
    """No zstd CLI on PATH at all: deja must read .jsonl.zst via the plugin's shim."""
    import zstandard
    rollout = CodexRollout(sandbox.codex_root, SCOPE_CWD, "zst-sess").turns(3)
    raw = rollout.path.read_bytes()
    half = raw.index(b"\n", len(raw) // 2) + 1
    # Two frames, as a rollout compressed while still being appended can have.
    packed = zstandard.ZstdCompressor().compress(raw[:half]) + zstandard.ZstdCompressor().compress(raw[half:])
    rollout.path.with_name(rollout.path.name + ".zst").write_bytes(packed)
    rollout.path.unlink()
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    base = {**os.environ, "PATH": os.pathsep.join([str(Path(system_root) / "System32"), system_root])}
    assert shutil.which("zstd", path=base["PATH"]) is None
    deja = conv.Deja(DEJA, conv.deja_environment(sandbox.state_dir, DEJA, base=base))
    sessions = [s for s in deja.list_sessions("demo/proj") if s["harness"] == "codex"]
    assert [s["id"] for s in sessions] == ["zst-sess"]
    messages, total = deja.show_page("codex", "zst-sess", 0)
    assert total == 6 and [m["role"] for m in messages] == ["user", "assistant"] * 3


# ------------------------------------------------------------------- pure: rendering

# Message shapes copied from `deja show --json` output on real Claude Code sessions.
REAL_COMMAND = {"role": "command", "time": "2026-09-25T17:25:30.659+08:00",
                "text": "$ cd C:/repo && python -m pytest -q tests/test_a.py 2>&1 | tail -5  — exit 0"}
REAL_EDIT = {"role": "edit", "time": "2026-09-26T01:05:46.35+08:00",
             "text": "C:\\repo\\restore.py\nopener = re.search(r'X[^\\n]*\\{\\n', c)"}
REAL_WROTE = {"role": "wrote", "time": "2026-09-25T17:26:36.786+08:00",
              "text": "C:\\repo\\progress.py\n32142feb3d925990 ef01514a9a214dcd"}


def test_render_keeps_prose_and_compacts_actions():
    assert conv.render_message({"role": "user", "text": "line1\nline2", "time": "2026-07-17T17:00:01+08:00"}) \
        == "**user** · 2026-07-17 09:00Z\nline1\nline2\n"
    assert conv.render_message({"role": "assistant", "text": "ok"}) == "**assistant**\nok\n"
    assert conv.render_message(REAL_COMMAND) == (
        "**command** · 2026-09-25 09:25Z\n"
        "$ cd C:/repo && python -m pytest -q tests/test_a.py 2>&1 | tail -5  — exit 0\n")
    assert conv.render_message(REAL_EDIT) == "**edit** · 2026-09-25 17:05Z\nedited: C:\\repo\\restore.py\n"
    assert conv.render_message(REAL_WROTE).splitlines()[1] == "wrote: C:\\repo\\progress.py"


def test_render_clips_command_and_drops_noise_roles():
    long_command = {"role": "command", "text": "$ " + "x" * 500 + "\nsecond line"}
    rendered = conv.render_message(long_command).splitlines()
    assert len(rendered) == 2 and len(rendered[1]) <= conv.COMMAND_LINE_CHARS
    assert conv.render_message({"role": "command", "text": "ls -la"}).splitlines()[1] == "$ ls -la"
    for role in ("files", "tool-output", "summary", "mystery"):
        assert conv.render_message({"role": role, "text": "something"}) is None
    assert conv.render_message({"role": "user", "text": "   \n"}) is None


# Shapes copied from this machine's Codex 0.154 rollouts as deja 0.23.5 shows them.
CODEX_INJECTED_USER = [
    "# AGENTS.md instructions for C:\\Users\\Arc\\prd\\retrival\n\n<INSTRUCTIONS>\n# Rules\n",
    "<turn_aborted>\nThe user interrupted the previous turn on purpose.\n</turn_aborted>",
    '<subagent_notification>\n{"agent_path":"019f2a00","status":"completed"}\n</subagent_notification>',
    '<codex_internal_context source="goal">\nContinue working toward the active goal.\n',
    "<environment_context>\n  <cwd>C:\\repo</cwd>\n</environment_context>",
]


def test_render_drops_harness_written_user_messages():
    for text in CODEX_INJECTED_USER:
        assert conv.render_message({"role": "user", "text": text}) is None, text[:40]
    # The same words typed mid-sentence by the person are kept.
    kept = conv.render_message({"role": "user", "text": "why did <turn_aborted> show up?"})
    assert kept == "**user**\nwhy did <turn_aborted> show up?\n"


def test_render_keeps_only_the_request_from_codex_ide_context():
    text = ("# Context from my IDE setup:\n\n## Active file: PLAN.md\n\n## Open tabs:\n"
            "- PLAN.md: PLAN.md\n\n## My request for Codex:\n融入的效果如何\n")
    assert conv.render_message({"role": "user", "text": text}) == "**user**\n融入的效果如何\n"
    no_request = "# Context from my IDE setup:\n\n## Active file: PLAN.md\n"
    assert conv.render_message({"role": "user", "text": no_request}) is None


def test_time_is_rendered_in_utc_whatever_the_machine_zone():
    for stamp in ("2026-07-17T17:00:01+08:00", "2026-07-17T09:00:01Z", "2026-07-16T22:00:01-11:00"):
        assert conv.render_message({"role": "user", "text": "t", "time": stamp}).startswith(
            "**user** · 2026-07-17 09:00Z\n")
    assert conv.render_message({"role": "user", "text": "t", "time": "garbage"}).startswith("**user**\n")


def test_segment_title_uses_started_utc_date_and_safe_ids():
    assert conv.segment_title("claude", "abc-123", "2026-09-25T09:23:27.14Z", 3) \
        == ".conversations/claude/2026-09-25_abc-123/003.md"
    assert conv.segment_title("codex", "a/b c", "2026-09-25T23:59:59+08:00", 1) \
        == ".conversations/codex/2026-09-25_a_b_c/001.md"


def test_segmentation_boundaries_depend_only_on_the_prefix():
    msgs = []
    for i in range(40):
        msgs.append((2 * i, {"role": "user", "text": f"q{i} " + "w " * 100}))
        msgs.append((2 * i + 1, {"role": "assistant", "text": f"a{i} " + "w " * 400}))
    full = conv.segment_messages("claude", "s", "2026-07-17T09:00:00Z", msgs)
    assert len(full) > 3
    # Every cut is a soft-limit cut: the next segment opens on a user turn.
    for segment in full[1:]:
        assert msgs[segment.first_offset][1]["role"] == "user"
    # Cutting the transcript anywhere changes only segments from the cut point on.
    for cut in (30, 31, 57, 79):
        partial = conv.segment_messages("claude", "s", "2026-07-17T09:00:00Z", msgs[:cut])
        closed = partial[:-1]
        assert [s.sha for s in closed] == [s.sha for s in full[:len(closed)]]
    # The hard limit closes regardless of role: 4 x ~5000 chars pass 16000, so 10
    # assistant-only messages split 4 / 4 / 2.
    huge = [(i, {"role": "assistant", "text": "z" * 5000}) for i in range(10)]
    parts = conv.segment_messages("claude", "s", None, huge)
    assert [(s.part, s.first_offset, s.last_offset) for s in parts] == [(1, 0, 3), (2, 4, 7), (3, 8, 9)]


def test_header_carries_only_prefix_stable_facts():
    segment = conv.segment_messages(
        "codex", "sess-1", "2026-07-17T09:00:00Z",
        [(3, {"role": "user", "text": "hello"}), (4, {"role": "assistant", "text": "hi"})], first_part=2)[0]
    assert segment.body.startswith(
        "# codex session sess-1 · part 2\nstarted: 2026-07-17T09:00:00Z · messages 3–4\n\n**user**")
    assert segment.title == ".conversations/codex/2026-07-17_sess-1/002.md"
    assert segment.sha == hashlib.sha256(segment.body.encode("utf-8")).hexdigest()


def test_pypi_token_is_masked_even_when_deja_misses_it():
    token = "pypi-AgEIcHlwaS5vcmcCJGJhMmQ4ZjQ0LTAwMDAtNDAwMC04MDAw"
    segment = conv.segment_messages("claude", "s", None,
                                    [(0, {"role": "user", "text": f"publish with {token} now"})])[0]
    assert token not in segment.body and "[redacted:pypi-token]" in segment.body
    assert "pypi-short" in conv.segment_messages(
        "claude", "s", None, [(0, {"role": "user", "text": "pypi-short is fine"})])[0].body


# ------------------------------------------------------- state file: atomic, forgiving

def test_state_roundtrip_and_corruption_falls_back_to_empty(tmp_path):
    state = conv.ConversationState(tmp_path, "h", "C:/x")
    state.sessions["claude/s"] = {"updated": "u", "segments": {"t": "sha"}}
    state.save()
    assert conv.ConversationState(tmp_path, "h", "C:/x").sessions == state.sessions
    assert conv.ConversationState(tmp_path, "other-host", "C:/x").sessions == {}
    assert not list(state.path.parent.glob("*.tmp"))
    state.path.write_text("{not json", encoding="utf-8")
    assert conv.ConversationState(tmp_path, "h", "C:/x").sessions == {}
    state.path.write_text(json.dumps({"version": 999, "sessions": {"a": {}}}), encoding="utf-8")
    assert conv.ConversationState(tmp_path, "h", "C:/x").sessions == {}


# ------------------------------------------------------- real deja: planning behaviour

@needs_deja
def test_official_deja_samples_flow_through_to_segments(sandbox):
    """deja's own registry fixtures (MIT, see fixtures/conversations/NOTICE)."""
    claude_target = claude_dir(sandbox.claude_root, SCOPE_CWD) / "registry-claude.jsonl"
    shutil.copy(FIXTURES / "claude-session.jsonl", claude_target)
    codex_target = sandbox.codex_root / "sessions" / "2026" / "07" / "17"
    codex_target.mkdir(parents=True)
    shutil.copy(FIXTURES / "codex-rollout.jsonl",
                codex_target / "rollout-2026-07-17T09-00-00-registry-codex.jsonl")

    plans, seen = sandbox.plan(scope=SCOPE)
    assert seen == 1 and [p.key for p in plans] == ["claude/registry-claude"]
    body = plans[0].to_upload[0].body
    assert "inspect the registry fixture" in body and "the fixture has a valid message pair" in body
    assert plans[0].to_upload[0].title == ".conversations/claude/2026-07-17_registry-claude/001.md"

    plans, seen = sandbox.plan(scope="/workspace/registry-demo")
    assert [p.key for p in plans] == ["codex/registry-codex"]
    body = plans[0].to_upload[0].body
    assert "now check the tail" in body and "inspect the rollout" in body
    assert "CODEX-SUMMARY" not in body  # summary role is dropped


@needs_deja
def test_first_sync_uploads_every_segment_and_appending_touches_only_the_open_one(sandbox):
    session = ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-a").turns(28)  # 7 segments, last ~4 KB
    CodexRollout(sandbox.codex_root, SCOPE_CWD, "sess-b").turns(28)

    plans, seen = sandbox.plan()
    assert seen == 2 and {p.key for p in plans} == {"claude/sess-a", "codex/sess-b"}
    first = {p.key: p for p in plans}
    for plan in plans:
        assert len(plan.to_upload) == 7
        assert [s.part for s in plan.to_upload] == list(range(1, len(plan.to_upload) + 1))
        assert plan.new_state["open_part"] == len(plan.to_upload)
    sandbox.commit(plans)

    # Nothing changed: no session is re-read at all.
    CountingDeja.shows = 0
    plans, seen = sandbox.plan(deja=CountingDeja(DEJA, conv.deja_environment(sandbox.state_dir, DEJA)))
    assert plans == [] and seen == 2 and CountingDeja.shows == 0

    session.user("one more question").assistant("one more answer")
    plans, _ = sandbox.plan()
    assert [p.key for p in plans] == ["claude/sess-a"]
    plan = plans[0]
    old = first["claude/sess-a"]
    assert not plan.rebuilt
    assert [s.title for s in plan.to_upload] == [old.to_upload[-1].title]
    assert plan.to_upload[0].sha != old.to_upload[-1].sha
    # Closed segments keep their recorded hash, and resuming equals a from-scratch build.
    for segment in old.to_upload[:-1]:
        assert plan.new_state["segments"][segment.title] == segment.sha
    sandbox.commit(plans)
    fresh = conv.ConversationState(sandbox.state_dir / "scratch", "h", SCOPE)
    scratch, _ = conv.plan_scope_sync(sandbox.deja(), "demo/proj", fresh, "main")
    rebuilt = {p.key: p for p in scratch}["claude/sess-a"]
    assert {s.title: s.sha for s in rebuilt.to_upload} == plan.new_state["segments"]


@needs_deja
def test_appending_past_the_soft_limit_opens_a_new_segment_and_keeps_the_old_one_byte_identical(sandbox):
    session = ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-a").turns(30)  # last segment 8319 chars
    plans, _ = sandbox.plan()
    old = plans[0]
    sandbox.commit(plans)
    session.user("a new topic").assistant("an answer")
    plans, _ = sandbox.plan()
    assert [s.part for s in plans[0].to_upload] == [len(old.to_upload) + 1]
    assert plans[0].new_state["segments"][old.to_upload[-1].title] == old.to_upload[-1].sha


@needs_deja
def test_segments_are_deterministic_across_runs_and_carry_no_deja_title(sandbox):
    ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-a").turns(12)
    first, _ = sandbox.plan()
    second, _ = sandbox.plan()
    assert [s.sha for s in first[0].to_upload] == [s.sha for s in second[0].to_upload]
    header = first[0].to_upload[0].body.split("\n\n", 1)[0]
    assert "turn 0 alpha0_0" not in header  # deja's own title is the elided first prompt
    assert re.fullmatch(r"# claude session sess-a · part 1\nstarted: \S+ · messages \d+–\d+", header)


@needs_deja
def test_agents_switch_controls_spawned_sessions(sandbox):
    ClaudeSession(sandbox.claude_root, SCOPE_CWD, "main-1").user("do the thing").assistant("done")
    ClaudeSession(sandbox.claude_root, SCOPE_CWD, "helper-1", agent_of="main-1").user("sub task").assistant("sub done")
    raw = sandbox.deja().list_sessions("demo/proj")
    assert {s["id"] for s in raw} == {"main-1", "helper-1"}
    assert [s.get("kind") for s in raw if s["id"] == "helper-1"] == ["sidechain"]

    plans, _ = sandbox.plan(mode="main")
    assert [p.session_id for p in plans] == ["main-1"]
    plans, _ = sandbox.plan(mode="agents")
    assert sorted(p.session_id for p in plans) == ["helper-1", "main-1"]


@needs_deja
def test_real_deja_redaction_and_pypi_gap(sandbox):
    github = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    pypi = "pypi-AgEIcHlwaS5vcmcCJGJhMmQ4ZjQ0LTAwMDAtNDAwMC04MDAw"
    ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-s").user(
        f"my tokens: {github} and {pypi}, use them").assistant("will not echo them")
    plans, _ = sandbox.plan()
    body = plans[0].to_upload[0].body
    assert github not in body, "deja stopped redacting GitHub tokens"
    assert pypi not in body and "[redacted:pypi-token]" in body


@needs_deja
def test_edit_actions_are_kept_as_paths_and_file_lists_dropped(sandbox):
    (ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-e").user("fix it")
     .edit("C:/work/demo/proj/app/mod.py").assistant("fixed"))
    plans, _ = sandbox.plan()
    body = plans[0].to_upload[0].body
    assert "edited: C:/work/demo/proj/app/mod.py" in body
    assert "**files**" not in body and "**tool-output**" not in body


@needs_deja
def test_rewritten_session_is_rebuilt_and_uploads_only_changed_segments(sandbox):
    session = ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-r").turns(30)
    plans, _ = sandbox.plan()
    old = plans[0]
    sandbox.commit(plans)
    assert old.new_state["open_segment_start_offset"] > 0

    # Rewrite the transcript in place: only the final turn changes text, but the open
    # segment's first message no longer matches what was remembered.
    lines = session.path.read_text(encoding="utf-8").splitlines()
    first_open_offset = old.new_state["open_segment_start_offset"]
    record = json.loads(lines[first_open_offset])
    record["message"]["content"] = "completely different opening message"
    lines[first_open_offset] = json.dumps(record)
    lines.append(json.dumps({**record, "uuid": "extra", "timestamp": "2026-07-17T10:00:00Z",
                             "message": {"role": "assistant",
                                         "content": [{"type": "text", "text": "tail"}]}}))
    session.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plans, _ = sandbox.plan()
    assert plans[0].rebuilt
    assert [s.title for s in plans[0].to_upload] == [old.to_upload[-1].title]
    assert plans[0].new_state["segments"].keys() >= old.new_state["segments"].keys()


@needs_deja
def test_unchanged_rendering_does_not_reupload_when_only_dropped_roles_grow(sandbox):
    session = ClaudeSession(sandbox.claude_root, SCOPE_CWD, "sess-d").user("hi").assistant("hello")
    before, _ = sandbox.plan()
    sandbox.commit(before)
    # A Read tool_use shows up as a dropped `files` message but moves `updated`.
    session._record("assistant", [{"type": "tool_use", "id": "toolu_x", "name": "Read",
                                   "input": {"file_path": "/a/b.py"}}])
    plans, _ = sandbox.plan()
    assert len(plans) == 1 and plans[0].to_upload == []
    assert plans[0].new_state["updated"] != before[0].new_state["updated"]


def test_missing_deja_binary_is_a_clear_error(tmp_path):
    with pytest.raises(conv.DejaNotFound):
        conv.run_deja(str(tmp_path / "no-such-deja"), ["last", "--json"], {}, 5)


@needs_deja
def test_nonzero_exit_and_non_json_are_clear_errors(sandbox):
    with pytest.raises(conv.DejaError, match="exited"):
        conv.run_deja(DEJA, ["show", "no-such-session-id", "--json"],
                      conv.deja_environment(sandbox.state_dir, DEJA), 60)
    with pytest.raises(conv.DejaError):
        conv.run_deja(DEJA, ["version"], conv.deja_environment(sandbox.state_dir, DEJA), 60)


# ---------------------------------------------------- the background syncer, no backend

def _no_http(*_a, **_k):
    raise AssertionError("an unchanged or empty sync must make zero HTTP requests")


@needs_deja
def test_syncer_is_single_flight_rate_limited_and_makes_no_http_when_nothing_to_do(sandbox, monkeypatch):
    monkeypatch.setenv(conv.MODE_ENV, "1")

    async def scenario():
        calls = []

        async def post(path, body):
            calls.append(path)
            return _no_http()

        async def get(path):
            calls.append(path)
            return _no_http()

        syncer = conv.ConversationSyncer(post=post, get=get, state_dir=lambda: sandbox.state_dir)
        assert syncer.kick("h", SCOPE) is None  # a healthy sync adds nothing to the output
        task = syncer._tasks[("h", SCOPE)]
        assert syncer.kick("h", SCOPE) is None
        assert syncer._tasks[("h", SCOPE)] is task  # single flight
        await task
        assert syncer.kick("h", SCOPE) is None
        assert syncer._tasks[("h", SCOPE)] is task  # <60 s since the last start: no new run
        assert syncer._last[("h", SCOPE)]["state"] == "done"
        assert calls == []

    asyncio.run(scenario())


def test_syncer_is_silent_when_disabled(monkeypatch):
    monkeypatch.delenv(conv.MODE_ENV, raising=False)

    async def scenario():
        syncer = conv.ConversationSyncer(post=_no_http, get=_no_http, state_dir=lambda: Path("."))
        assert syncer.kick("h", SCOPE) is None
        assert syncer._tasks == {}

    asyncio.run(scenario())


def test_syncer_reports_missing_deja(monkeypatch, tmp_path):
    monkeypatch.setenv(conv.MODE_ENV, "1")
    monkeypatch.setenv(conv.DEJA_BIN_ENV, str(tmp_path / "nope.exe"))

    async def scenario():
        syncer = conv.ConversationSyncer(post=_no_http, get=_no_http, state_dir=lambda: tmp_path)
        syncer.kick("h", SCOPE)
        await syncer._tasks[("h", SCOPE)]
        assert syncer.kick("h", SCOPE) == (
            f"Conversations sync failed: {conv.DEJA_BIN_ENV} does not name a file")

    asyncio.run(scenario())


# -------------------------------------------------- plugin wiring (tool-level helpers)

def _load_plugin():
    path = Path(__file__).resolve().parents[1] / "mcp_plugin" / "retrieval_mcp.py"
    spec = importlib.util.spec_from_file_location("retrieval_mcp_conversations_wiring", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plugin_helper_is_inert_unless_enabled_and_local(monkeypatch, tmp_path):
    plugin = _load_plugin()
    monkeypatch.setenv("RETRIEVAL_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv(conv.MODE_ENV, raising=False)
    assert plugin._conversation_sync_line(str(tmp_path)) is None
    monkeypatch.setenv(conv.MODE_ENV, "0")
    assert plugin._conversation_sync_line(str(tmp_path)) is None
    assert plugin._with_conversation_line("body", None) == "body"
    assert plugin._with_conversation_line("body", "Conversations: x") == "body\n\nConversations: x"

    monkeypatch.setenv(conv.MODE_ENV, "1")
    monkeypatch.setenv(conv.DEJA_BIN_ENV, str(tmp_path / "nope.exe"))

    async def scenario():
        # a scope that is not a local directory has no local transcripts
        assert plugin._conversation_sync_line(str(tmp_path / "missing-dir")) is None
        assert plugin._conversation_sync_line(str(tmp_path)) is None  # started, nothing to say
        await asyncio.gather(*plugin._CONVERSATION_SYNCER._tasks.values())
        assert plugin._conversation_sync_line(str(tmp_path)) == (
            f"Conversations sync failed: {conv.DEJA_BIN_ENV} does not name a file")

    asyncio.run(scenario())
