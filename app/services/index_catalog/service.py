"""Application service for the index-inventory catalog."""
from __future__ import annotations

from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.schemas import (
    IndexOwnerSummary,
    IndexReportRequest,
    IndexReportResponse,
    IndexScopePolicyUpsert,
    InventoryDirNode,
    InventoryFileNode,
    InventoryHostNode,
    InventoryProjectNode,
    InventoryResponse,
    InventoryUserNode,
    ScopeConflict,
    ScopeOverlapResponse,
)
from app.services.scope_paths import (
    canonicalize_scope_path,
    file_abspath,
    is_scope_ancestor,
    is_windows_scope,
    normalize_scope,
    scope_looks_like_path,
)


def _dir_of(rel_path: str) -> str:
    norm = rel_path.replace("\\", "/")
    return norm.rsplit("/", 1)[0] if "/" in norm else "."


class IndexCatalogService:
    def __init__(self, repository: IndexCatalogRepository | None = None):
        self.repository = repository or IndexCatalogRepository()

    def report(self, request: IndexReportRequest, *, owner_user_id: str) -> IndexReportResponse:
        upserted, removed = self.repository.report_files(
            owner_user_id=owner_user_id,
            host=request.host,
            project=request.project,
            root_dir=request.root_dir,
            files=[f.model_dump() for f in request.files],
            replace=request.replace,
            patterns=request.patterns,
            max_depth=request.max_depth,
        )
        return IndexReportResponse(
            host=request.host, project=request.project, owner_user_id=owner_user_id,
            upserted=upserted, removed=removed, total_files=len(request.files),
        )

    @staticmethod
    def _policy_out(row) -> IndexScopePolicyUpsert:
        return IndexScopePolicyUpsert(
            host=row.host, project=row.project, kind=row.kind, root_dir=row.root_dir,
            auto_refresh=bool(row.auto_refresh),
            respect_gitignore=bool(row.respect_gitignore),
            exclude_globs=list(row.exclude_globs_json or []),
            min_check_interval_s=row.min_check_interval_s,
        )

    def upsert_policy(
        self, request: IndexScopePolicyUpsert, *, owner_user_id: str,
    ) -> IndexScopePolicyUpsert:
        if normalize_scope(request.root_dir) != normalize_scope(request.project):
            raise ValueError("index policy root_dir must match project scope")
        row = self.repository.upsert_policy(
            owner_user_id=owner_user_id, **request.model_dump())
        return self._policy_out(row)

    def get_policy(
        self, *, owner_user_id: str, host: str, project: str, kind: str,
    ) -> IndexScopePolicyUpsert | None:
        row = self.repository.get_policy(
            owner_user_id=owner_user_id, host=host, project=project, kind=kind)
        return self._policy_out(row) if row is not None else None

    def inventory(
        self,
        *,
        owner_user_id: str | None = None,
        host: str | None = None,
        project: str | None = None,
        kind: str | None = None,
        detail: bool = True,
        code_scope_states: dict[tuple[str, str, str], dict[str, object]] | None = None,
    ) -> InventoryResponse:
        """Build the user → host → project → dir → file tree from flat rows.

        ``detail=False`` returns a SUMMARY: project-level counts + server-computed freshness
        (last_indexed_at / last_commit), with the per-dir/per-file nodes OMITTED. The full
        per-file tree carries a 64-char content_hash on every file and ballooned to ~10MB for
        a heavy indexer — big enough to truncate over the wire and break the inventory tool,
        which only ever renders project-level counts + freshness. Detail (default) is kept for
        the admin console and any consumer that genuinely wants the file list."""
        runtime_code_scope_states = code_scope_states or {}
        # Queue state disappears on a backend restart, but an in-place Qdrant mutation
        # must remain fail-closed until the job records a complete snapshot. A runtime
        # queued job can refine a durable marker, but must never downgrade a durable
        # recovery_required boundary from an interrupted write.
        durable_code_scope_states: dict[tuple[str, str, str], dict[str, object]] = {}
        # ``host`` filters inventory display rows, not Code KB visibility. A fan-out
        # writer persists its marker under its own host while mutating a target whose
        # catalog row can belong to another host; marker identity is owner + project.
        visibility_markers = self.repository.list_code_scope_visibility(
            owner_user_id=owner_user_id, project=project,
        ) if kind in (None, "code") else []
        for marker in visibility_markers:
            durable_code_scope_states[(
                str(marker["owner_user_id"]), str(marker["host"]), str(marker["project"]),
            )] = {
                "active_index_job_id": marker.get("job_id"),
                "previous_completed_snapshot_available": bool(
                    marker.get("previous_completed_snapshot_available")),
                "recovery_required": marker.get("state") == "recovery_required",
                "visibility_state": marker.get("state"),
            }
        def visibility_strength(state: dict[str, object]) -> int:
            # Lower is stronger: recovery_required > preparing > completed. Runtime
            # states omit ``visibility_state`` by design, which means active preparing.
            if bool(state.get("recovery_required")):
                return 0
            if state.get("visibility_state") != "completed":
                return 1
            return 2

        code_scope_states = dict(durable_code_scope_states)
        for key, runtime_state in runtime_code_scope_states.items():
            durable_state = code_scope_states.get(key)
            if durable_state is None or visibility_strength(runtime_state) < visibility_strength(durable_state):
                code_scope_states[key] = runtime_state
        # A Code KB policy is a durable registration even before its first successful
        # report.  Include policy-only scopes so the inventory does not silently turn
        # "submitted/preparing" into "nothing exists".
        code_policies = self.repository.list_policies(
            owner_user_id=owner_user_id,
            host=host,
            project=project,
            kind="code",
        ) if kind in (None, "code") else []

        def ensure_project(
            owner: str,
            host_name: str,
            project_name: str,
            root_dir: str | None,
        ) -> dict:
            u = users.setdefault(owner, {"hosts": {}})
            h = u["hosts"].setdefault(host_name, {"projects": {}})
            return h["projects"].setdefault(
                project_name,
                {
                    "root_dir": root_dir,
                    "dirs": {},
                    "agg": [],
                    "kinds": set(),
                    "registered_code_policy": False,
                    "latest": None,
                    "commit": None,
                },
            )

        users: dict[str, dict] = {}
        if detail:
            # Per-file tree: needs every row (content_hash included) to build the
            # dir/file breakdown the admin console renders.
            rows = self.repository.list_inventory(
                owner_user_id=owner_user_id, host=host, project=project, kind=kind,
            )
            for row in rows:
                p = ensure_project(row.owner_user_id, row.host, row.project, row.root_dir)
                p["kinds"].add(row.kind)
                d = p["dirs"].setdefault(_dir_of(row.rel_path), [])
                d.append(row)
                # Freshness is derived HERE, while rows are still in list_inventory()'s
                # rel_path order, not in the dir-grouped loop below: sorting by dir name
                # is not the same order as sorting by rel_path ("a-b/c" precedes "a/z"
                # by path but follows it by dir), and on an indexed_at tie the two orders
                # would pick different commits. Strict ">" keeps the smallest rel_path
                # among tied rows — the same rule repository.aggregate_inventory()
                # reproduces in SQL, so SUMMARY and DETAIL cannot disagree.
                if row.indexed_at and (p["latest"] is None or row.indexed_at > p["latest"]):
                    p["latest"] = row.indexed_at
                    # Pair the commit with the FRESHEST file so "last indexed <ts>
                    # @<commit>" is honest (not a stale first-seen commit from an
                    # unrelated older file).
                    p["commit"] = (row.metadata_json or {}).get("git_commit")
        else:
            # SUMMARY: SQL-side per-(project, kind) sums instead of downloading every
            # file row just to re-sum them in Python (A.2/A.3 of the plan doc).
            agg_rows = self.repository.aggregate_inventory(
                owner_user_id=owner_user_id, host=host, project=project, kind=kind,
            )
            for agg in agg_rows:
                p = ensure_project(
                    agg["owner_user_id"], agg["host"], agg["project"], agg["root_dir"])
                p["kinds"].add(agg["kind"])
                p["agg"].append(agg)

        for policy in code_policies:
            p = ensure_project(
                policy["owner_user_id"], policy["host"], policy["project"],
                policy["root_dir"],
            )
            p["registered_code_policy"] = True

        def scope_visibility_state(
            owner: str, host_name: str, project_name: str,
        ) -> dict[str, object] | None:
            exact = code_scope_states.get((owner, host_name, project_name))
            if exact is not None:
                return exact
            # Fan-out's destination host is not part of the vector namespace.  Its
            # durable marker may carry the writer's host while the existing target
            # catalog row displays another host, so match owner/project as a fallback.
            candidates = [
                state for (state_owner, _state_host, state_project), state
                in code_scope_states.items()
                if state_owner == owner and state_project == project_name
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda state: (
                visibility_strength(state),
                str(state.get("active_index_job_id") or ""),
            ))
            return candidates[0]

        # Direct REST callers are allowed to submit an index without first persisting a
        # refresh policy.  Preserve that active scope too, rather than omitting its
        # preparation state merely because it has no catalog rows yet.
        for (state_owner, state_host, state_project) in code_scope_states:
            if owner_user_id is not None and state_owner != owner_user_id:
                continue
            if host is not None and state_host != host:
                continue
            if project is not None and state_project != project:
                continue
            ensure_project(state_owner, state_host, state_project, state_project)

        user_nodes: list[InventoryUserNode] = []
        tot_hosts = tot_projects = tot_files = tot_chunks = 0
        for owner, u in users.items():
            host_nodes: list[InventoryHostNode] = []
            u_files = u_chunks = 0
            for host_name, h in u["hosts"].items():
                tot_hosts += 1
                project_nodes: list[InventoryProjectNode] = []
                h_files = h_chunks = 0
                for project_name, p in h["projects"].items():
                    tot_projects += 1
                    dir_nodes: list[InventoryDirNode] = []
                    p_files = p_chunks = 0
                    # newest indexed_at across the project → freshness, and the
                    # git_commit of that freshest file → indexed revision. DETAIL fills
                    # these while ingesting rel_path-ordered rows above; SUMMARY merges
                    # the per-kind aggregates below using the same tie-break.
                    p_latest = p["latest"]
                    p_commit = p["commit"]
                    if detail:
                        for dir_name, dir_rows in sorted(p["dirs"].items()):
                            d_chunks = 0
                            files = [
                                InventoryFileNode(
                                    rel_path=r.rel_path, kind=r.kind, collection=r.collection,
                                    language=r.language, chunk_count=r.chunk_count or 0,
                                    content_hash=r.content_hash, size_bytes=r.size_bytes,
                                    indexed_at=r.indexed_at, metadata=r.metadata_json or {},
                                )
                                for r in dir_rows
                            ]
                            for r in dir_rows:
                                d_chunks += r.chunk_count or 0
                            dir_nodes.append(InventoryDirNode(
                                dir=dir_name, file_count=len(dir_rows),
                                chunk_count=d_chunks, files=files,
                            ))
                            p_files += len(dir_rows)
                            p_chunks += d_chunks
                    else:
                        # SUMMARY: same counts/freshness, summed from the SQL-side
                        # per-(project, kind) aggregate rows instead of per-file rows.
                        # dir_nodes stays [] — that's the whole point (KB, not MB).
                        p_commit_rel_path = None
                        for agg in p["agg"]:
                            p_files += agg["file_count"]
                            p_chunks += agg["chunk_count"]
                            last_indexed_at = agg["last_indexed_at"]
                            if not last_indexed_at:
                                continue
                            # A project's kinds are separate aggregate rows, so the
                            # cross-kind winner is resolved here. Newest timestamp wins;
                            # an exact tie goes to the SMALLEST rel_path — DETAIL's rule,
                            # since it scans one rel_path-ordered stream over all kinds.
                            # Ordering by kind instead would report a different project's
                            # commit than DETAIL whenever two kinds share a batch stamp.
                            if p_latest is None or last_indexed_at > p_latest or (
                                last_indexed_at == p_latest
                                and p_commit_rel_path is not None
                                and (agg["last_commit_rel_path"] or "") < p_commit_rel_path
                            ):
                                p_latest = last_indexed_at
                                p_commit = agg["last_commit"]
                                p_commit_rel_path = agg["last_commit_rel_path"] or ""
                    has_completed_code_snapshot = "code" in p["kinds"]
                    state = scope_visibility_state(owner, host_name, project_name)
                    if state is not None and state.get("visibility_state") != "completed":
                        code_snapshot_state = "preparing"
                        code_active_index_job_id = str(
                            state.get("active_index_job_id") or "") or None
                        code_previous_completed_snapshot_available = bool(
                            state.get("previous_completed_snapshot_available")
                        ) or has_completed_code_snapshot
                        code_recovery_required = bool(state.get("recovery_required"))
                    elif has_completed_code_snapshot:
                        code_snapshot_state = "completed"
                        code_active_index_job_id = None
                        code_previous_completed_snapshot_available = True
                        code_recovery_required = False
                    elif state is not None and state.get("visibility_state") == "completed":
                        code_snapshot_state = "completed"
                        code_active_index_job_id = None
                        code_previous_completed_snapshot_available = True
                        code_recovery_required = False
                    elif p["registered_code_policy"]:
                        code_snapshot_state = "registered"
                        code_active_index_job_id = None
                        code_previous_completed_snapshot_available = False
                        code_recovery_required = False
                    else:
                        code_snapshot_state = None
                        code_active_index_job_id = None
                        code_previous_completed_snapshot_available = None
                        code_recovery_required = None
                    project_nodes.append(InventoryProjectNode(
                        project=project_name, host=host_name, root_dir=p["root_dir"],
                        file_count=p_files, chunk_count=p_chunks,
                        last_indexed_at=p_latest, last_commit=p_commit,
                        code_snapshot_state=code_snapshot_state,
                        code_active_index_job_id=code_active_index_job_id,
                        code_previous_completed_snapshot_available=(
                            code_previous_completed_snapshot_available),
                        code_recovery_required=code_recovery_required,
                        dirs=dir_nodes,
                    ))
                    h_files += p_files
                    h_chunks += p_chunks
                host_nodes.append(InventoryHostNode(
                    host=host_name, file_count=h_files, chunk_count=h_chunks,
                    projects=project_nodes,
                ))
                u_files += h_files
                u_chunks += h_chunks
            user_nodes.append(InventoryUserNode(
                owner_user_id=owner, file_count=u_files, chunk_count=u_chunks,
                hosts=host_nodes,
            ))
            tot_files += u_files
            tot_chunks += u_chunks

        return InventoryResponse(
            users=user_nodes,
            totals={
                "users": len(user_nodes), "hosts": tot_hosts, "projects": tot_projects,
                "files": tot_files, "chunks": tot_chunks,
            },
        )

    def has_completed_code_scope(self, *, owner_user_id: str, project: str) -> bool:
        """Durable catalog proof that this owner/project has a completed Code KB snapshot."""
        marker = self.repository.code_scope_visibility(
            owner_user_id=owner_user_id, projects=[project])
        if marker is not None:
            return marker["state"] == "completed"
        return self.repository.has_inventory(
            owner_user_id=owner_user_id, project=project, kind="code")

    def mark_code_scope_preparing(
        self, *, owner_user_id: str, host: str, project: str, job_id: str | None,
    ) -> dict:
        return self.repository.set_code_scope_visibility(
            owner_user_id=owner_user_id, host=host, project=project,
            job_id=job_id, state="preparing")

    def mark_code_scope_completed(
        self, *, owner_user_id: str, host: str, project: str, job_id: str | None,
    ) -> dict:
        return self.repository.set_code_scope_visibility(
            owner_user_id=owner_user_id, host=host, project=project,
            job_id=job_id, state="completed")

    def mark_code_scope_recovery_required(
        self, *, owner_user_id: str, host: str, project: str, job_id: str | None,
    ) -> dict:
        return self.repository.set_code_scope_visibility(
            owner_user_id=owner_user_id, host=host, project=project,
            job_id=job_id, state="recovery_required")

    def code_scope_visibility(
        self, *, owner_user_id: str, projects: list[str],
    ) -> dict | None:
        return self.repository.code_scope_visibility(
            owner_user_id=owner_user_id, projects=projects)

    def owner_code_scope_visibility(self, *, owner_user_id: str) -> dict | None:
        """One incomplete marker is enough to make an owner-wide search fail closed.

        Used by the single-project route (`/api/code/search`, `project=None`), which
        deliberately withholds the WHOLE owner-wide search rather than judging scopes
        individually. `list_incomplete_code_scopes` below is the per-scope counterpart
        federated-search's decision-10 exclusion uses instead.
        """
        markers = self.repository.list_code_scope_visibility(
            owner_user_id=owner_user_id, incomplete_only=True)
        return markers[0] if markers else None

    def list_incomplete_code_scopes(self, *, owner_user_id: str) -> list[dict]:
        """Every non-completed code-scope durable marker for one owner.

        Federated-search decision 10: per-scope exclusion needs the FULL incomplete
        candidate set (each judged individually), not just the first one
        `owner_code_scope_visibility` returns for the all-or-nothing legacy path.
        """
        return self.repository.list_code_scope_visibility(
            owner_user_id=owner_user_id, incomplete_only=True)

    def _scope_respects_gitignore(
        self, *, owner_user_id: str, host: str | None, project: str, kind: str | None,
    ) -> bool:
        """Whether an already-indexed scope skipped the files git ignores.

        Read from its stored policy. Policies are keyed by the canonical scope spelling
        while catalog rows can still carry a legacy case-variant, so try the row's own
        spelling first and the canonical one second. A scope indexed before policies were
        recorded has no row at all; the indexer's default is respect_gitignore=True, and
        assuming that is what ran is the reading that matches the files on disk."""
        for candidate in (project, canonicalize_scope_path(project)):
            policy = self.repository.get_policy(
                owner_user_id=owner_user_id, host=host or "", project=candidate,
                kind=kind or "code")
            if policy is not None:
                return bool(policy.respect_gitignore)
        return True

    def scope_overlap(
        self,
        *,
        owner_user_id: str,
        project: str,
        host: str | None = None,
        kind: str | None = None,
        respect_gitignore: bool | None = None,
    ) -> ScopeOverlapResponse:
        """Compute the parent/child overlap for a candidate index ``project`` scope, so the
        MCP nested-scope guard gets a KB answer instead of downloading the whole per-file
        inventory (O(all files ever indexed), which grew to ~9MB and began truncating).

        Mirrors the guard's former client-side logic exactly: among the caller's OTHER
        indexed scopes of this kind, an ancestor scope that actually indexed a file under
        the candidate is a block; a descendant scope is a warning. Only ancestor candidates'
        files are loaded — bounded by the candidate's path-prefix depth (~0-1 in practice),
        never the whole tree; and the response is bounded by project count, not file count.
        ``kind`` is always passed by the MCP guard ("doc"/"code"), so a doc scope never
        blocks a code candidate; kind=None (no filter) is legacy parity with /inventory.

        ``respect_gitignore`` is the filter the CANDIDATE will index under. Turning it off
        under a parent that had it on is the one case where an overlapping child still adds
        coverage — the ignored files were never in the parent — so that pairing reports an
        ``ancestor_widening`` rather than a block. Omitting it keeps the pre-existing
        behaviour exactly, which is what an older client that never sends it must see.

        An ancestor verdict is reported only for a candidate scope that does NOT exist
        yet. Refusing a scope the owner already has cannot un-create it; it just freezes
        it at whatever it held when it was made, and the refusal's advice ("delete the
        parent scope first") would destroy a repository index to unfreeze one directory.
        Descendant warnings are unaffected — re-indexing a parent still reports the child
        scopes living under it."""
        project_norm = normalize_scope(project)
        if not scope_looks_like_path(project_norm):
            return ScopeOverlapResponse()

        projects = self.repository.list_projects(
            owner_user_id=owner_user_id, host=host, kind=kind)
        child_warnings: list[ScopeConflict] = []
        ancestor_candidates: list[dict] = []
        candidate_exists = False
        for p in projects:
            existing = p["project"] or ""
            existing_norm = normalize_scope(existing)
            if not existing or not scope_looks_like_path(existing_norm):
                continue
            if existing_norm == project_norm:
                # Same directory, whatever NTFS spelling the row was written under.
                candidate_exists = True
                continue
            if is_scope_ancestor(existing_norm, project_norm):
                ancestor_candidates.append(p)
            elif is_scope_ancestor(project_norm, existing_norm):
                child_warnings.append(ScopeConflict(project=existing, host=p["host"]))

        if candidate_exists:
            # A refresh, not a new scope. Returning early also skips the per-ancestor
            # inventory loads, which auto-refresh would otherwise pay on every tick.
            return ScopeOverlapResponse(child_warnings=child_warnings)

        ancestor_blocks: list[ScopeConflict] = []
        ancestor_widenings: list[ScopeConflict] = []
        for p in ancestor_candidates:
            anc_host, anc_proj = p["host"], p["project"]
            # Load ONLY this ancestor project's files (small) to confirm it indexed
            # something under the candidate subtree — an ancestor with no file under the
            # child adds no coverage overlap, so it must not block.
            rows = self.repository.list_inventory(
                owner_user_id=owner_user_id, host=anc_host, project=anc_proj, kind=kind)
            root = p["root_dir"] or anc_proj or ""
            if not any(is_scope_ancestor(project_norm,
                                         normalize_scope(file_abspath(root, r.rel_path)))
                       for r in rows):
                continue
            conflict = ScopeConflict(project=anc_proj, host=anc_host)
            if respect_gitignore is False and self._scope_respects_gitignore(
                    owner_user_id=owner_user_id, host=anc_host,
                    project=anc_proj, kind=kind):
                ancestor_widenings.append(conflict)
            else:
                ancestor_blocks.append(conflict)

        return ScopeOverlapResponse(
            ancestor_blocks=ancestor_blocks, child_warnings=child_warnings,
            ancestor_widenings=ancestor_widenings)

    def equivalent_project_scopes(
        self,
        *,
        owner_user_id: str,
        project: str,
        kind: str = "code",
    ) -> list[str]:
        """Return one canonical Windows scope plus legacy case-variant spellings.

        New clients write canonical case-folded Windows scope names. Older inventory rows
        can still contain the same NTFS directory with different component casing; search
        includes them until the owner re-indexes instead of claiming the canonical spelling
        is absent. This reads the compact project list only, never a per-file inventory or
        Qdrant path enumeration. POSIX scopes intentionally remain case-sensitive.
        """
        canonical = canonicalize_scope_path(project)
        if not is_windows_scope(canonical):
            return [canonical]
        variants = {
            row["project"]
            for row in self.repository.list_projects(
                owner_user_id=owner_user_id, kind=kind)
            if canonicalize_scope_path(row["project"] or "") == canonical
        }
        return [canonical, *sorted(variant for variant in variants if variant != canonical)]

    def delete_project(self, *, owner_user_id: str, project: str,
                       host: str | None = None, kind: str | None = None) -> int:
        """Soft-delete all of a project's catalog rows (across hosts by default). Returns
        the number removed and also removes its persisted refresh policy."""
        removed = self.repository.delete_files(
            owner_user_id=owner_user_id, project=project, host=host, kind=kind)
        self.repository.delete_policy(
            owner_user_id=owner_user_id, project=project, host=host, kind=kind)
        if kind in (None, "code"):
            self.repository.clear_code_scope_visibility(
                owner_user_id=owner_user_id, project=project)
        return removed

    def delete_files(self, *, owner_user_id: str, project: str, host: str | None,
                     kind: str | None, rel_paths: list[str]) -> int:
        """Soft-delete selected inventory rows while preserving the scope policy."""
        if not rel_paths:
            return 0
        return self.repository.delete_files(
            owner_user_id=owner_user_id,
            project=project,
            host=host,
            kind=kind,
            rel_paths=rel_paths,
        )

    def owners(self) -> list[IndexOwnerSummary]:
        return [IndexOwnerSummary(**row) for row in self.repository.list_owners()]

    def overview(self) -> dict[str, int]:
        return self.repository.overview()


_SERVICE: IndexCatalogService | None = None


def get_index_catalog_service() -> IndexCatalogService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = IndexCatalogService()
    return _SERVICE
