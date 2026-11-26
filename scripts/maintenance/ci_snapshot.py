#!/usr/bin/env python3
"""Run the test suite on GitHub Actions against the current working tree.

The working tree is committed to a throwaway branch ``ci/wt-<UTC timestamp>`` without
touching the local branch, index or files: a temporary GIT_INDEX_FILE is filled from
HEAD plus ``git add -A`` (so .gitignore is honoured), written as a tree and committed
with ``commit-tree -p HEAD``.  Only ``ci/wt-*`` is ever pushed; pushing ``main`` to a
remote with the deploy workflows would trigger them.

Default target is the PUBLIC CI-only repository (remote ``ci-public``, so Actions minutes
are free).  Nothing private may reach it: the notes, ``docs/``, the plan files and the
deploy workflows are left out of the tree, and the private history is never published.
The pushed chain is two commits: A = HEAD's tree minus those paths with no parent, and
B = the working-tree snapshot minus those paths with parent A (so ``HEAD^`` stays valid
for tests that diff against it).  ``--remote github`` keeps the old private flow (one
commit on top of HEAD, nothing left out).

Before anything leaves the machine the snapshot is scanned with the official gitleaks
release binary (tree and the commits about to be published); any finding aborts.
Download goes through the local proxy and fails closed when the proxy is down.

    python scripts/maintenance/ci_snapshot.py            # snapshot, scan, push (public)
    python scripts/maintenance/ci_snapshot.py --watch    # ... and follow the run
    python scripts/maintenance/ci_snapshot.py --dry-run  # snapshot + scan, list what is
                                                         # excluded and the commit ids
    python scripts/maintenance/ci_snapshot.py --remote github   # the private repository
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# remote name -> (GitHub repository for gh -R, is it public)
REMOTES = {
    "ci-public": ("Ironboxplus/retrieval-ci", True),
    "github": ("Ironboxplus/RetrievalPaper", False),
}
DEFAULT_REMOTE = "ci-public"
# Left out of every PUBLIC snapshot: git pathspecs removed from the private index (the files
# stay on disk).  `:(glob)` keeps `*` from crossing a directory separator.
PUBLIC_EXCLUDE_PATHSPECS = [
    *(f":(literal){name}" for name in (
        "journal.md", "env.md", "plan.md", "AGENTS.md", "INDEX.md", "overview.md",
        "CLAUDE.md")),
    ":(glob)PLAN-*.md",
    ":(glob)RESEARCH-*.md",
    ":(literal)docs",
    ":(glob).github/workflows/deploy-*.yml",
]
# Kept although inside an excluded directory (a test reads it; it holds no host names).
PUBLIC_KEEP = ["docs/openapi.json"]
BRANCH_PREFIX = "ci/wt-"
BRANCH_PATTERN = re.compile(r"^ci/wt-\d{8}T\d{6}Z$")
WORKFLOW = "tests.yml"
PROXY = "http://127.0.0.1:10808"  # AGENTS.md 2.2: every download on this host uses it
# Untracked junk kept out of the snapshot (never deleted locally).  The name carries
# U+F03A, the private-use stand-in Windows gives a ':' in "ip:443:" file names.
DEFAULT_EXCLUDES = ["20.205.243.166" + chr(0xF03A) + "443" + chr(0xF03A)]

GITLEAKS_VERSION = "8.30.1"
GITLEAKS_BASE = f"https://github.com/gitleaks/gitleaks/releases/download/v{GITLEAKS_VERSION}"
GITLEAKS_ARCHIVES = {
    "win32": (f"gitleaks_{GITLEAKS_VERSION}_windows_x64.zip", "gitleaks.exe"),
    "linux": (f"gitleaks_{GITLEAKS_VERSION}_linux_x64.tar.gz", "gitleaks"),
}


def git(*args: str, env: dict | None = None, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, stdin=subprocess.DEVNULL,
        encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def repo_root() -> Path:
    return Path(git("rev-parse", "--show-toplevel").strip())


def local_state(root: Path) -> tuple[str, str]:
    """What the dirty tree looks like to the user; must be identical after the run."""
    return (git("status", "--porcelain", cwd=root), git("diff", "--cached", "--stat", cwd=root))


def is_public_excluded(name: str) -> bool:
    """The PUBLIC_EXCLUDE_PATHSPECS rules, restated on a repo-relative path (an independent
    check of what the pathspec removal left behind)."""
    parts = PurePosixPath(name).parts
    if len(parts) == 1:
        return (name in {"journal.md", "env.md", "plan.md", "AGENTS.md", "INDEX.md",
                         "overview.md", "CLAUDE.md"}
                or (name.startswith(("PLAN-", "RESEARCH-")) and name.endswith(".md")))
    if parts[0] == "docs":
        return name not in PUBLIC_KEEP
    return (len(parts) == 3 and parts[:2] == (".github", "workflows")
            and parts[2].startswith("deploy-") and parts[2].endswith(".yml"))


def _index_names(root: Path, env: dict) -> list[str]:
    return [n for n in git("ls-files", "-z", env=env, cwd=root).split("\0") if n]


def _drop_from_index(root: Path, env: dict, pathspecs: list[str]) -> None:
    for spec in pathspecs:  # drops it from the private index only; the file stays on disk
        git("rm", "--cached", "-r", "-q", "--ignore-unmatch", "--", spec, env=env, cwd=root)


def _drop_public(root: Path, env: dict, extra: list[str]) -> None:
    """Remove the public exclusions (and ``extra``) from the index, then put PUBLIC_KEEP back
    with the entry it had."""
    kept = []
    for path in PUBLIC_KEEP:
        entry = git("ls-files", "-s", "-z", "--", f":(literal){path}", env=env, cwd=root)
        for record in (r for r in entry.split("\0") if r):
            meta, _, name = record.partition("\t")
            mode, sha, _stage = meta.split()
            kept.append((mode, sha, name))
    _drop_from_index(root, env, extra + PUBLIC_EXCLUDE_PATHSPECS)
    for mode, sha, name in kept:
        git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{name}", env=env, cwd=root)


def _apply_overrides(root: Path, env: dict, overrides: dict[str, str],
                     drops: list[str]) -> None:
    """Replace or delete entries of the private index only; the working tree is untouched."""
    for repo_path, local in overrides.items():
        local_path = Path(local)
        if not local_path.is_file():
            raise SystemExit(f"--override {repo_path}: {local} is not a file")
        sha = git("hash-object", "-w", "--path", repo_path, str(local_path),
                  env=env, cwd=root).strip()
        entry = git("ls-files", "-s", "-z", "--", f":(literal){repo_path}", env=env, cwd=root)
        mode = entry.split()[0] if entry.strip() else "100644"
        git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{repo_path}",
            env=env, cwd=root)
    _drop_from_index(root, env, [f":(literal){path}" for path in drops])


def build_snapshot(root: Path, stamp: str, excludes: list[str],
                   public: bool = False, overrides: dict[str, str] | None = None,
                   drops: list[str] | None = None) -> dict:
    """Commit the working tree through a private index.

    Returns commit/tree/parent/dropped/files. Private: one commit on top of HEAD. Public:
    commit A (HEAD's tree minus PUBLIC_EXCLUDE_PATHSPECS, no parent) and commit B (the
    working tree minus the same, parent A); ``dropped`` lists the paths the public
    exclusions removed from B."""
    head = git("rev-parse", "HEAD", cwd=root).strip()
    literal = [f":(literal){path}" for path in excludes]
    parent = head
    dropped: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ci_snapshot_index_") as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        git("read-tree", head, env=env, cwd=root)
        for attempt in (1, 2):  # another editor may delete a file between listing and reading
            added = subprocess.run(["git", "add", "-A"], cwd=root, env=env, capture_output=True,
                                   stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
            if not added.returncode:
                break
            if attempt == 2:
                raise SystemExit(f"git add -A failed: {added.stderr.strip()}")
        _drop_from_index(root, env, literal)
        if public:
            before = set(_index_names(root, env))
            _drop_public(root, env, [])
            kept = _index_names(root, env)
            dropped = sorted(before - set(kept))
            files = len(kept)
            env_a = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index_base")}
            git("read-tree", head, env=env_a, cwd=root)
            _drop_public(root, env_a, literal)
            tree_a = git("write-tree", env=env_a, cwd=root).strip()
            parent = git("commit-tree", tree_a, "-m",
                         f"CI base: tracked files of {head[:12]} without notes or history",
                         cwd=root).strip()
        _apply_overrides(root, env, overrides or {}, drops or [])
        files = len(_index_names(root, env))
        tree = git("write-tree", env=env, cwd=root).strip()
    message = f"CI snapshot of the working tree on {head[:12]} ({stamp})"
    commit = git("commit-tree", tree, "-p", parent, "-m", message, cwd=root).strip()
    return {"commit": commit, "tree": tree, "parent": parent, "dropped": dropped,
            "files": files, "public": public}


def check_tree_contents(root: Path, commit: str, excludes: list[str],
                        public: bool = False) -> None:
    commits = [commit] + ([git("rev-parse", f"{commit}^", cwd=root).strip()] if public else [])
    for index, one in enumerate(commits):
        # The workflow files may be untracked in HEAD (so absent from base commit A); only
        # the snapshot commit B has to carry them.
        _check_one_tree(root, one, excludes, public, require_workflow=index == 0)


def _check_one_tree(root: Path, commit: str, excludes: list[str], public: bool,
                    require_workflow: bool = True) -> None:
    names = [n for n in git("ls-tree", "-r", "--name-only", "-z", commit, cwd=root).split("\0") if n]
    leaked = [n for n in names if n in excludes]
    if leaked:
        raise SystemExit(f"snapshot still contains excluded path(s): {len(leaked)}")
    if public:
        private = [n for n in names if is_public_excluded(n)]
        if private:
            raise SystemExit(f"public snapshot still contains private path(s): {private[:5]}")
        if git("rev-parse", f"{commit}^", cwd=root, check=False).strip() and \
                git("rev-list", "--count", commit, cwd=root).strip() not in {"1", "2"}:
            raise SystemExit("public snapshot has history beyond the A/B chain")
    secret_named = [n for n in names if PurePosixPath(n).name.startswith("pypi-")]
    if secret_named:
        raise SystemExit(f"snapshot contains PyPI token file(s): {len(secret_named)}; add them to .gitignore")
    for required in (f".github/workflows/{WORKFLOW}", ".github/actions/ci-env/action.yml"):
        if require_workflow and required not in names:
            raise SystemExit(f"snapshot lacks {required}")


def _download(url: str) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY, "http": PROXY}))
    try:
        with opener.open(url, timeout=120) as response:
            return response.read()
    except OSError as exc:  # never fall back to a direct connection
        raise SystemExit(f"download via proxy {PROXY} failed for {url}: {exc}") from exc


def ensure_gitleaks() -> Path:
    found = shutil.which("gitleaks")
    if found:
        return Path(found)
    if sys.platform not in GITLEAKS_ARCHIVES:
        raise SystemExit(f"no gitleaks on PATH and no release mapping for {sys.platform}")
    archive_name, binary_name = GITLEAKS_ARCHIVES[sys.platform]
    cache = Path.home() / ".cache" / "ci_snapshot" / f"gitleaks-{GITLEAKS_VERSION}"
    binary = cache / binary_name
    if binary.is_file():
        return binary
    archive = _download(f"{GITLEAKS_BASE}/{archive_name}")
    checksums = _download(f"{GITLEAKS_BASE}/gitleaks_{GITLEAKS_VERSION}_checksums.txt").decode()
    expected = dict(reversed(line.split()) for line in checksums.splitlines() if line.strip())[archive_name]
    if hashlib.sha256(archive).hexdigest() != expected:
        raise SystemExit(f"{archive_name}: sha256 does not match the release checksums")
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / archive_name
    archive_path.write_bytes(archive)
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as bundle:
            bundle.extract(binary_name, cache)
    else:
        with tarfile.open(archive_path) as bundle:
            bundle.extract(binary_name, cache)
        binary.chmod(0o755)
    return binary


def run_gitleaks(binary: Path, args: list[str], report: Path) -> list[dict]:
    result = subprocess.run(
        [str(binary), *args, "--no-banner", "--redact", "--exit-code", "1",
         "--report-format", "json", "--report-path", str(report)],
        capture_output=True, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    if result.returncode not in (0, 1):
        raise SystemExit(f"gitleaks failed ({result.returncode}): {result.stderr.strip()[-500:]}")
    findings = json.loads(report.read_text(encoding="utf-8") or "[]") if report.exists() else []
    if result.returncode == 1 and not findings:
        raise SystemExit("gitleaks exited 1 without a readable report")
    return findings


def scan_snapshot(root: Path, commit: str, base: str | None) -> None:
    binary = ensure_gitleaks()
    with tempfile.TemporaryDirectory(prefix="ci_snapshot_scan_") as scratch:
        scratch_path = Path(scratch)
        tree_dir = scratch_path / "tree"
        tree_dir.mkdir()
        with subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=root,
                              stdout=subprocess.PIPE) as archive:
            with tarfile.open(fileobj=archive.stdout, mode="r|") as bundle:
                bundle.extractall(tree_dir, filter="data")
        # The allowlist that is scanned with is the one inside the snapshot, i.e. the
        # one that gets pushed, never a different working-tree copy.
        config = tree_dir / ".gitleaks.toml"
        if not config.is_file():
            raise SystemExit("snapshot lacks .gitleaks.toml")
        findings = run_gitleaks(
            binary, ["dir", str(tree_dir), "--config", str(config)], scratch_path / "tree.json")
        # History the push would newly publish (the snapshot commit plus unpushed commits).
        log_range = f"{base}..{commit}" if base else commit
        findings += run_gitleaks(
            binary, ["git", str(root), "--config", str(config), "--log-opts", log_range],
            scratch_path / "history.json")
    if findings:
        # Rule and location only: the secret itself must not reach the terminal or any log.
        for finding in findings:
            print(f"gitleaks: {finding.get('RuleID')} {finding.get('File')}:{finding.get('StartLine')} "
                  f"{(finding.get('Commit') or '')[:12]}", file=sys.stderr)
        raise SystemExit(f"gitleaks reported {len(findings)} finding(s); nothing was pushed")
    print(f"gitleaks {GITLEAKS_VERSION}: tree and {log_range} clean")


def remote_branches(root: Path, remote: str) -> list[str]:
    out = git("ls-remote", "--heads", remote, f"refs/heads/{BRANCH_PREFIX}*", cwd=root)
    names = [line.split("\trefs/heads/", 1)[1] for line in out.splitlines() if "\trefs/heads/" in line]
    return sorted(n for n in names if BRANCH_PATTERN.match(n))  # the timestamp sorts lexically


def prune_old(root: Path, remote: str, keep: int) -> None:
    names = remote_branches(root, remote)
    for name in names[:-keep] if keep else names:
        assert BRANCH_PATTERN.match(name), name
        git("push", remote, "--delete", name, cwd=root)
        print(f"deleted old remote branch {name}")


def find_run(root: Path, repo: str, branch: str, commit: str) -> dict | None:
    for _ in range(30):
        out = subprocess.run(
            ["gh", "run", "list", "-R", repo, "--branch", branch, "--workflow", WORKFLOW, "--limit", "5",
             "--json", "databaseId,url,status,headSha"],
            cwd=root, capture_output=True, encoding="utf-8", errors="replace").stdout
        for run in json.loads(out or "[]"):
            if run["headSha"] == commit:
                return run
        time.sleep(4)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--remote", choices=sorted(REMOTES), default=DEFAULT_REMOTE,
                        help="git remote to push to (default: the public CI-only repository)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and scan the snapshot, list the excluded paths and the "
                             "commit ids, do not push")
    parser.add_argument("--watch", action="store_true", help="follow the run until it finishes")
    parser.add_argument("--keep", type=int, default=5, help="remote ci/wt-* branches to keep")
    parser.add_argument("--override", action="append", default=[], metavar="REPO_PATH=LOCAL_FILE",
                        help="put LOCAL_FILE's content at REPO_PATH in the snapshot's private "
                             "index only (repeatable); the working tree is not touched")
    parser.add_argument("--drop", action="append", default=[], metavar="REPO_PATH",
                        help="leave REPO_PATH out of the snapshot's private index only "
                             "(repeatable)")
    parser.add_argument("--exclude", action="append", metavar="PATH",
                        help="repo-relative path to leave out of the snapshot (repeatable; "
                             "replaces the default stray-file list)")
    args = parser.parse_args()
    excludes = args.exclude or DEFAULT_EXCLUDES
    remote = args.remote
    repo, public = REMOTES[remote]

    root = repo_root()
    before = local_state(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch = f"{BRANCH_PREFIX}{stamp}"
    assert BRANCH_PATTERN.match(branch)

    overrides = {}
    for item in args.override:
        repo_path, sep, local = item.partition("=")
        if not sep or not repo_path or not local:
            raise SystemExit(f"--override expects REPO_PATH=LOCAL_FILE, got {item!r}")
        overrides[repo_path] = local
    snap = build_snapshot(root, stamp, excludes, public, overrides, args.drop)
    commit, tree = snap["commit"], snap["tree"]
    print(f"snapshot commit {commit} (tree {tree}) for {branch} -> {remote} ({repo})")
    check_tree_contents(root, commit, excludes, public)
    if public:
        # Nothing of the private history is published: the whole chain (A and B) is new.
        base = None
        print(f"base commit A {snap['parent']} (no parent); snapshot commit B {commit}")
    else:
        base = git("rev-parse", "--verify", "-q", f"{remote}/main", cwd=root,
                   check=False).strip() or None
    for repo_path, local in overrides.items():
        print(f"override {repo_path} <- {local}")
    for repo_path in args.drop:
        print(f"drop {repo_path}")
    if args.dry_run:
        print(f"files in commit B: {snap['files']}")
        if public:
            print(f"excluded from the public snapshot ({len(snap['dropped'])} path(s)):")
            for name in snap["dropped"]:
                print(f"  {name}")
    scan_snapshot(root, commit, base)

    if not args.dry_run:
        git("push", remote, f"{commit}:refs/heads/{branch}", cwd=root)
        print(f"pushed {remote} {branch}")
        run = find_run(root, repo, branch, commit)
        if run is None:
            print("no workflow run appeared for the snapshot; check the Actions tab", file=sys.stderr)
        else:
            print(f"run {run['databaseId']}: {run['url']}")
            if args.watch:
                subprocess.run(["gh", "run", "watch", "-R", repo, str(run["databaseId"]), "--exit-status"], cwd=root)
        prune_old(root, remote, args.keep)

    after = local_state(root)
    if before != after:
        print("local git status / staged diff changed during the run (concurrent edits?):", file=sys.stderr)
        for label, old, new in (("status", before[0], after[0]), ("cached", before[1], after[1])):
            if old != new:
                print(f"  {label}: {len(old.splitlines())} -> {len(new.splitlines())} lines", file=sys.stderr)
        return 2
    print("local git status and index unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
