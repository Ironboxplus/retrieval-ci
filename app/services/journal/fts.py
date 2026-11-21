"""SQLite FTS5 keyword index for ACE journal entries.

Keyword search is deliberately **embedding-free**: it runs entirely in SQLite, so
it keeps working when the shared BGE-M3 embed endpoint (4090) is down (W131). It
complements the dense/sparse Qdrant path — literal terms, file paths, error codes
and W-numbers match here without a model round-trip.

The virtual table lives in the same SQLite DB as the ORM tables but is managed by
raw DDL (SQLAlchemy does not model FTS5 vtables). owner_user_id / project_slug /
source_kind are UNINDEXED columns so we can filter the same match by tenant.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

logger = logging.getLogger("journal.fts")

TABLE = "journal_fts"

_CREATE = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5(
    entry_id UNINDEXED,
    owner_user_id UNINDEXED,
    project_slug UNINDEXED,
    source_kind UNINDEXED,
    title,
    summary,
    body,
    tags,
    tokenize = 'unicode61'
)
"""

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def ensure_fts(session: Session) -> None:
    """Idempotently create the FTS5 virtual table."""
    session.execute(text(_CREATE))


# ASCII tokens shorter than this are matched as whole words, not prefixes.
# journal_fts has no prefix= index, so SQLite FTS5 (fts5SetupPrefixIter in
# ext/fts5/fts5_index.c) visits every term with the prefix and merges their
# doclists in memory. Measured on a copy of the production index (2026-11-21):
# `a*` / `s*` / `1*` used 22-32 MB and 0.5-0.9 s, and a query mixing several
# 1-char prefixes 36.8 MB / 1.7 s; quoted exact terms used 1-5 MB / under 35 ms.
MIN_ASCII_PREFIX_CHARS = 4


def _prefix_term(token: str) -> bool:
    """True when `token` is expanded as a prefix (non-ASCII, or long enough)."""
    return not token.isascii() or len(token) >= MIN_ASCII_PREFIX_CHARS


def _fts_term(token: str) -> str:
    """One quoted FTS5 term, with `*` appended when it is a prefix term."""
    return '"' + token.replace('"', '""') + '"' + ("*" if _prefix_term(token) else "")


def _match_query(raw: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression (OR semantics).

    Extracting `\\w+` tokens strips punctuation operators (quotes, `-`, `:`), but the
    upper-case barewords `AND`, `OR`, `NOT` and `NEAR` survive and FTS5 would parse
    them as operators (`fts5: syntax error near "OR"`). So every term is emitted as a
    double-quoted string, which FTS5 always treats as a phrase, never a keyword.

    A term is a prefix (`"tok"*`) only when it is non-ASCII or at least
    MIN_ASCII_PREFIX_CHARS characters long, so `observ` still matches `observability`
    and CJK runs still match through a prefix (unicode61 makes a CJK run one token).
    Shorter ASCII tokens match whole words only: a 1-3 character prefix covers a large
    share of the vocabulary and costs tens of MB and up to seconds (see the constant).

    Terms are OR-ed, not AND-ed: this is the tool's default "always works" recall
    path, and a natural-language sentence rarely has EVERY token in one entry; AND
    made such queries silently return nothing. bm25 still ranks entries that match
    more (and rarer) terms first, so precision comes from ordering, not exclusion.
    """
    tokens = query_tokens(raw)
    if not tokens:
        return None
    return " OR ".join(_fts_term(t) for t in tokens)


def query_tokens(raw: str) -> list[str]:
    """Return the same sanitized tokens used to build journal FTS MATCH queries."""
    return _TOKEN_RE.findall(raw or "")


def like_token_pattern(token: str) -> str:
    """Return a SQLite LIKE pattern for one sanitized token.

    FTS tokenization treats identifiers such as ``delete_code`` literally enough
    to find the right entry. Chunk hydration must not broaden that to
    ``delete-code`` by letting LIKE interpret ``_`` as a wildcard.
    """
    escaped = (
        token.lower()
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{escaped}%"


def index_entry(
    session: Session,
    *,
    rowid: int,
    entry_id: str,
    owner_user_id: str | None,
    project_slug: str,
    source_kind: str,
    title: str | None,
    summary: str | None,
    body: str,
    tags: list[str] | None,
) -> None:
    """Replace the FTS row for one entry (delete-then-insert = idempotent upsert).

    `rowid` is `journal_entries.id`. rowid is the only indexed key of an FTS5 table;
    `entry_id` is UNINDEXED, so keying on it made every delete a full scan of all bodies.
    """
    delete_entry(session, rowid)
    session.execute(
        text(
            f"INSERT INTO {TABLE} "
            "(rowid, entry_id, owner_user_id, project_slug, source_kind, title, summary, body, tags) "
            "VALUES (:rowid, :entry_id, :owner, :project, :source, :title, :summary, :body, :tags)"
        ),
        {
            "rowid": rowid,
            "entry_id": entry_id,
            "owner": owner_user_id or "",
            "project": project_slug,
            "source": source_kind,
            "title": title or "",
            "summary": summary or "",
            "body": body or "",
            "tags": " ".join(tags or []),
        },
    )


DELETE_BY_ROWID = f"DELETE FROM {TABLE} WHERE rowid = :rowid"


def delete_entry(session: Session, rowid: int) -> None:
    session.execute(text(DELETE_BY_ROWID), {"rowid": rowid})


def delete_entries(session: Session, rowids: Sequence[int]) -> None:
    if rowids:
        session.execute(text(DELETE_BY_ROWID), [{"rowid": rowid} for rowid in rowids])


def is_aligned(session: Session) -> bool:
    """True iff the FTS rows are exactly the live journal entries keyed by rowid = id.

    Tables written before rowid was tied to `journal_entries.id` hold auto-assigned
    rowids, so deleting by rowid would remove another entry's row. They may also hold
    rows for soft-deleted entries. Any such row, or a live entry without a row, means
    the table must be rebuilt. One pass over the FTS table plus primary-key lookups.
    """
    total, matched = session.execute(text(
        f"SELECT count(*), count(e.id) FROM {TABLE} f "
        "LEFT JOIN journal_entries e "
        "ON e.id = f.rowid AND e.entry_id = f.entry_id AND e.deleted_at IS NULL"
    )).one()
    live = session.execute(text(
        "SELECT count(*) FROM journal_entries WHERE deleted_at IS NULL"
    )).scalar_one()
    return total == matched == live


# Max entry_ids expanded into a single `IN (...)` — well under SQLite's default variable
# cap (999 on old builds, 32766 on recent). A path_glob match-set larger than this is
# batched and merged, so a huge subtree can't raise "too many SQL variables".
_IN_BATCH = 500


def search(
    session: Session,
    query: str,
    *,
    owner_user_id: str | None = None,
    project_slug: str | Sequence[str] | None = None,
    source_kind: str | None = None,
    entry_ids: list[str] | None = None,
    include_non_doc: bool = False,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Return `(entry_id, score)` best-first. Score is `-bm25` (higher = better).

    `entry_ids` restricts the match to that id set (pushed into the WHERE, BEFORE LIMIT),
    so a path_glob filter fills top_k only from matching docs without dropping any (the
    id set is resolved once by the service; passing [] matches nothing). A match-set larger
    than `_IN_BATCH` is queried in batches and merged — each batch is a valid top-`limit`
    for its slice, so re-sorting the union by rank and truncating yields the exact global
    top-`limit` (bm25 is comparable across batches: same MATCH, same table).

    `include_non_doc` widens the id set to "these ids, or any entry that is not a doc":
    a subdirectory search narrows docs by path but keeps the scope's memory entries.
    Every batch then also returns the non-doc rows, so the merge de-duplicates them."""
    match = _match_query(query)
    if match is None:
        return []
    if entry_ids is not None and not entry_ids and not include_non_doc:
        return []  # glob matched nothing — never widen to "match everything"

    if entry_ids is None or len(entry_ids) <= _IN_BATCH:
        return _search_batch(session, match, owner_user_id, project_slug,
                             source_kind, entry_ids, limit, include_non_doc)
    merged: list[tuple[str, float]] = []
    for start in range(0, len(entry_ids), _IN_BATCH):
        merged.extend(_search_batch(session, match, owner_user_id, project_slug,
                                    source_kind, entry_ids[start:start + _IN_BATCH], limit,
                                    include_non_doc))
    merged.sort(key=lambda r: r[1], reverse=True)  # score = -bm25, higher = better
    if include_non_doc:
        seen: set[str] = set()
        merged = [row for row in merged if not (row[0] in seen or seen.add(row[0]))]
    return merged[:limit]


def _search_batch(
    session: Session,
    match: str,
    owner_user_id: str | None,
    project_slug: str | Sequence[str] | None,
    source_kind: str | None,
    entry_ids: list[str] | None,
    limit: int,
    include_non_doc: bool = False,
) -> list[tuple[str, float]]:
    # The FTS table is read for rowid and bm25 only. Its owner/project/source/entry_id columns are
    # UNINDEXED, so a filter on them (or selecting entry_id before ORDER BY ... LIMIT) reads the
    # content row of EVERY match: on .88 (2026-11-21, a 47 GiB DB with almost nothing in the page
    # cache) "systemd-oomd killed backend memory pressure" matched 3,802 rows and took 7.4-25 s,
    # against 0.2-0.5 s with the filters below and the same entries in the same order. So every
    # filter runs on journal_entries/journal_projects (indexed; the FTS columns are copies of
    # them, see index_entry and JournalRepository.reindex_fts) and reaches the FTS table as a
    # rowid set: rowid = journal_entries.id, which the repository's alignment guard enforces
    # before any search runs.
    where: list[str] = []
    params: dict[str, Any] = {"match": match, "limit": limit}
    if owner_user_id is not None:
        where.append("COALESCE(p.owner_user_id, '') = :owner")  # index_entry writes None as ''
        params["owner"] = owner_user_id
    if project_slug is not None:
        projects = [project_slug] if isinstance(project_slug, str) else list(project_slug)
        where.append("p.slug IN :projects")
        params["projects"] = projects
    if source_kind is not None:
        where.append("e.source_kind = :source")
        params["source"] = source_kind
    if entry_ids is not None and include_non_doc and not entry_ids:
        where.append("e.source_kind <> 'doc'")
        entry_ids = None
    elif entry_ids is not None and include_non_doc:
        where.append("(e.entry_id IN :entry_ids OR e.source_kind <> 'doc')")
        params["entry_ids"] = entry_ids
    elif entry_ids is not None:
        where.append("e.entry_id IN :entry_ids")
        params["entry_ids"] = entry_ids
    sql = f"SELECT rowid, bm25({TABLE}) AS rank FROM {TABLE} WHERE {TABLE} MATCH :match"
    if where:
        sql += (" AND rowid IN (SELECT e.id FROM journal_entries e JOIN journal_projects p "
                f"ON p.project_id = e.project_id WHERE {' AND '.join(where)})")
    stmt = text(sql + " ORDER BY rank LIMIT :limit")
    if project_slug is not None:
        stmt = stmt.bindparams(bindparam("projects", expanding=True))
    if entry_ids is not None:
        stmt = stmt.bindparams(bindparam("entry_ids", expanding=True))
    ranked = session.execute(stmt, params).fetchall()
    if not ranked:
        return []
    # entry_id of the top rows only: one rowid point lookup each.
    lookup = text(f"SELECT rowid, entry_id FROM {TABLE} WHERE rowid IN :rowids").bindparams(
        bindparam("rowids", expanding=True))
    entry_of = dict(session.execute(lookup, {"rowids": [row[0] for row in ranked]}).fetchall())
    return [(entry_of[row[0]], -float(row[1])) for row in ranked]
