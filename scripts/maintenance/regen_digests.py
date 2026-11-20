"""Recompute the pinned tool-surface digests from the plugin's real tools/list.

AGENTS.md: the surface is pinned in four places, and after regenerating you must
compare entry by entry — only the tools this change touched may differ. So this
prints a per-tool verdict rather than a single hash, and refuses to write when a
tool outside the expected set moved.

Usage: python scripts/maintenance/regen_digests.py [--write] [--expect name[,name...]]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp_plugin"))

import retrieval_mcp as plugin  # noqa: E402
import retrieval_mcp_daemon as daemon  # noqa: E402

DAEMON_FILE = ROOT / "mcp_plugin" / "retrieval_mcp_daemon.py"

expected = set()
for i, arg in enumerate(sys.argv):
    if arg == "--expect" and i + 1 < len(sys.argv):
        expected = {n.strip() for n in sys.argv[i + 1].split(",") if n.strip()}

tools = asyncio.run(plugin.mcp.list_tools())
names = {t.name for t in tools}

print(f"tools advertised: {len(tools)}")

# 1. names
pinned_names = set(daemon.TOOL_SURFACE_NAMES)
if names != pinned_names:
    print(f"  NAME SET CHANGED  added={sorted(names - pinned_names)} "
          f"removed={sorted(pinned_names - names)}")
else:
    print("  name set: unchanged")

# 2/3. digests
new_overall = daemon.canonical_tool_surface_digest(tools)
new_fields = daemon.tool_surface_field_digests(tools)
old_fields = daemon.TOOL_SURFACE_FIELD_DIGESTS

moved = []
for name in sorted(set(new_fields) | set(old_fields)):
    old = old_fields.get(name)
    new = new_fields.get(name)
    if old is None:
        moved.append((name, "ADDED")); continue
    if new is None:
        moved.append((name, "REMOVED")); continue
    diff = [f for f in ("description", "inputSchema") if old.get(f) != new.get(f)]
    if diff:
        moved.append((name, "+".join(diff)))

print(f"  overall digest: {daemon.TOOL_SURFACE_DIGEST[:16]} -> {new_overall[:16]}")
print(f"  tools whose field digests moved ({len(moved)}):")
for name, what in moved:
    flag = "ok" if name in expected else "UNEXPECTED"
    print(f"    {name:<28} {what:<24} [{flag}]")

unexpected = [n for n, _ in moved if n not in expected]
if unexpected:
    print(f"\nREFUSING TO WRITE: unexpected surface movement in {unexpected}")
    sys.exit(1)

if "--write" not in sys.argv:
    print("\n(dry run; pass --write to pin)")
    sys.exit(0)

source = DAEMON_FILE.read_text(encoding="utf-8")
source = re.sub(
    r'^TOOL_SURFACE_DIGEST = "[0-9a-f]{64}"$',
    f'TOOL_SURFACE_DIGEST = "{new_overall}"',
    source, count=1, flags=re.MULTILINE,
)
for name, fields in new_fields.items():
    pattern = re.compile(
        r'^(    "' + re.escape(name) + r'": \{"description": ")[0-9a-f]{64}'
        r'(", "inputSchema": ")[0-9a-f]{64}(".*)$',
        re.MULTILINE,
    )
    replacement = rf'\g<1>{fields["description"]}\g<2>{fields["inputSchema"]}\g<3>'
    source, n = pattern.subn(replacement, source, count=1)
    if n != 1 and name in expected:
        raise SystemExit(f"could not pin field digests for {name}")
DAEMON_FILE.write_text(source, encoding="utf-8")
print(f"\npinned {DAEMON_FILE}")
