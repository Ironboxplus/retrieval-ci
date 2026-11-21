"""Write small Claude Code / Codex transcripts in the agents' real on-disk formats.

The record shapes follow deja-vu's own registry samples (MIT, Vladislav Shulcz;
``tests/fixtures/conversations/``) plus the fields real Claude Code adds around tool
calls.  Tests point deja's ``DEJA_CLAUDE_ROOT`` / ``DEJA_CODEX_ROOT`` at the directories
these helpers fill, so the developer's real ``~/.claude`` and ``~/.codex`` are never read.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_TIME = datetime(2026, 7, 17, 9, 0, 0, tzinfo=timezone.utc)


def _stamp(index: int) -> str:
    return (BASE_TIME + timedelta(seconds=index)).strftime("%Y-%m-%dT%H:%M:%SZ")


def filler(index: int, words: int) -> str:
    return f"turn {index} " + " ".join(f"alpha{index}_{n}" for n in range(words))


def claude_dir(root: Path, cwd: str) -> Path:
    """Claude Code names a project directory after the cwd with non-alphanumerics -> '-'."""
    path = root / re.sub(r"[^A-Za-z0-9]", "-", cwd)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ClaudeSession:
    """An append-only Claude Code session file."""

    def __init__(self, root: Path, cwd: str, session_id: str, *, agent_of: str | None = None) -> None:
        directory = claude_dir(root, cwd)
        if agent_of:
            directory = directory / agent_of / "subagents"
            directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (f"agent-{session_id}.jsonl" if agent_of else f"{session_id}.jsonl")
        self.cwd, self.session_id, self.agent_of = cwd, session_id, agent_of
        self.count = 0

    def _record(self, role: str, content, **extra) -> None:
        self.count += 1
        record = {
            "parentUuid": None, "isSidechain": bool(self.agent_of), "type": role,
            "uuid": f"{self.session_id}-{self.count}",
            "sessionId": self.agent_of or self.session_id, "cwd": self.cwd,
            "timestamp": _stamp(self.count),
            "message": {"role": role, "content": content},
        }
        if self.agent_of:
            record["agentId"] = self.session_id
        record.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def user(self, text: str) -> "ClaudeSession":
        self._record("user", text)
        return self

    def assistant(self, text: str) -> "ClaudeSession":
        self._record("assistant", [{"type": "text", "text": text}])
        return self

    def edit(self, file_path: str) -> "ClaudeSession":
        self._record("assistant", [{"type": "tool_use", "id": f"toolu_{self.count}", "name": "Edit",
                                    "input": {"file_path": file_path, "old_string": "a",
                                              "new_string": "b"}}])
        return self

    def turns(self, pairs: int, *, user_words: int = 20, assistant_words: int = 160,
              start: int = 0) -> "ClaudeSession":
        for i in range(start, start + pairs):
            self.user(filler(i, user_words))
            self.assistant(filler(i, assistant_words))
        return self


class CodexRollout:
    """An append-only Codex rollout; the project comes from ``session_meta.cwd``."""

    def __init__(self, root: Path, cwd: str, session_id: str) -> None:
        directory = root / "sessions" / "2026" / "07" / "17"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rollout-2026-07-17T09-00-00-{session_id}.jsonl"
        self.count = 0
        self._line({"type": "session_meta", "payload": {"session_id": session_id, "cwd": cwd}})

    def _line(self, record: dict) -> None:
        self.count += 1
        record = {"timestamp": _stamp(self.count), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def user(self, text: str) -> "CodexRollout":
        self._line({"type": "response_item", "payload": {
            "role": "user", "content": [{"type": "input_text", "text": text}]}})
        return self

    def assistant(self, text: str) -> "CodexRollout":
        self._line({"type": "response_item", "payload": {
            "role": "assistant", "content": [{"type": "output_text", "text": text}]}})
        return self

    def turns(self, pairs: int, *, user_words: int = 20, assistant_words: int = 160,
              start: int = 0) -> "CodexRollout":
        for i in range(start, start + pairs):
            self.user(filler(i, user_words))
            self.assistant(filler(i, assistant_words))
        return self
