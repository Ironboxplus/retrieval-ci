"""Server-side ACE playbook service.

Centralized counterpart to the client-side ``mcp_server/playbook.py``: bullets are
stored per (owner_user_id, project_slug) so the admin console can view/manage them
and many clients share one playbook. Embeddings come from the warm BGE-M3 singleton
(same model the search pipeline uses) — no HTTP hop, since this runs inside the
backend. Retrieval is brute-force cosine over the (small) per-scope bullet set.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.playbook.schemas import (
    PlaybookBulletResponse,
    PlaybookDeleteResponse,
    PlaybookListResponse,
    PlaybookRecordRequest,
    PlaybookSearchRequest,
)
from core.db.database import get_db_session
from core.db.models import PlaybookBullet

logger = logging.getLogger("playbook.service")

DEFAULT_DEDUP_THRESHOLD = 0.92

# embed_fn(texts) -> one dense vector per text
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    from core.indexing.embedders import embed_texts
    return [v.tolist() for v in embed_texts(list(texts)).dense]


def _bullet_key(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class PlaybookService:
    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        session_factory: Callable[[], Any] = get_db_session,
    ):
        self.embed_fn = embed_fn or _default_embed
        self.dedup_threshold = dedup_threshold
        self.session_factory = session_factory

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session

    def _embed_one(self, text: str) -> list[float]:
        vecs = self.embed_fn([text])
        if not vecs:
            raise RuntimeError("embed_fn returned no vectors")
        return vecs[0]

    @staticmethod
    def _to_response(b: PlaybookBullet, *, score: float | None = None,
                     deduped: bool = False) -> PlaybookBulletResponse:
        return PlaybookBulletResponse(
            id=b.id, bullet_key=b.bullet_key, owner_user_id=b.owner_user_id,
            project_slug=b.project_slug,
            content=b.content, section=b.section, domain=b.domain,
            helpful=b.helpful, harmful=b.harmful, score=score, deduped=deduped,
            created_at=b.created_at, updated_at=b.updated_at,
        )

    def _live(self, session: Session, owner_user_id: str, project_slug: str,
              *, domain: str | None = None) -> list[PlaybookBullet]:
        q = (session.query(PlaybookBullet)
             .filter(PlaybookBullet.owner_user_id == owner_user_id,
                     PlaybookBullet.project_slug == project_slug,
                     PlaybookBullet.deleted_at.is_(None)))
        if domain:
            q = q.filter(PlaybookBullet.domain == domain)
        return q.all()

    def record(self, request: PlaybookRecordRequest, *, owner_user_id: str) -> PlaybookBulletResponse:
        content = (request.content or "").strip()
        if not content:
            raise ValueError("playbook content must be non-empty")
        vec = self._embed_one(content)

        with self.session() as session:
            # grow-and-refine: a near-duplicate reinforces the existing bullet instead
            # of adding a new one, preventing playbook bloat / context collapse.
            best, best_score = None, -1.0
            for row in self._live(session, owner_user_id, request.project_slug, domain=request.domain):
                emb = row.embedding_json
                if not emb:
                    continue
                s = _cosine(vec, emb)
                if s > best_score:
                    best_score, best = s, row
            if best is not None and best_score >= self.dedup_threshold:
                best.helpful += 1
                best.updated_at = _utcnow()
                session.flush()
                logger.info("playbook dedup-merge id=%s sim=%.3f helpful=%d",
                            best.id, best_score, best.helpful)
                resp = self._to_response(best, deduped=True)
                session.commit()
                return resp

            key = _bullet_key(content)
            existing = (session.query(PlaybookBullet)
                        .filter(PlaybookBullet.owner_user_id == owner_user_id,
                                PlaybookBullet.project_slug == request.project_slug,
                                PlaybookBullet.bullet_key == key)
                        .one_or_none())
            if existing is not None:
                # same exact content resurrected (or previously soft-deleted): reinforce
                existing.deleted_at = None
                existing.helpful += 1
                existing.section = request.section
                existing.domain = request.domain
                existing.embedding_json = vec
                existing.updated_at = _utcnow()
                session.flush()
                resp = self._to_response(existing, deduped=True)
                session.commit()
                return resp

            bullet = PlaybookBullet(
                owner_user_id=owner_user_id, project_slug=request.project_slug,
                bullet_key=key, section=request.section, content=content,
                domain=request.domain, helpful=0, harmful=0, embedding_json=vec,
            )
            session.add(bullet)
            session.flush()
            logger.info("playbook add id=%s section=%s domain=%s", bullet.id, request.section, request.domain)
            resp = self._to_response(bullet)
            session.commit()
            return resp

    def search(self, request: PlaybookSearchRequest, *, owner_user_id: str) -> PlaybookListResponse:
        if not request.query.strip():
            return PlaybookListResponse(bullets=[], total=0)
        qv = self._embed_one(request.query)
        with self.session() as session:
            scored: list[tuple[float, PlaybookBullet]] = []
            for row in self._live(session, owner_user_id, request.project_slug, domain=request.domain):
                emb = row.embedding_json
                if not emb:
                    continue
                scored.append((_cosine(qv, emb), row))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:request.top_k]
            return PlaybookListResponse(
                bullets=[self._to_response(r, score=s) for s, r in top],
                total=len(top),
            )

    def list_bullets(self, *, owner_user_id: str, project_slug: str | None = None,
                     domain: str | None = None, limit: int = 200) -> PlaybookListResponse:
        with self.session() as session:
            q = (session.query(PlaybookBullet)
                 .filter(PlaybookBullet.owner_user_id == owner_user_id,
                         PlaybookBullet.deleted_at.is_(None)))
            if project_slug:
                q = q.filter(PlaybookBullet.project_slug == project_slug)
            if domain:
                q = q.filter(PlaybookBullet.domain == domain)
            rows = q.order_by(PlaybookBullet.updated_at.desc()).limit(limit).all()
            return PlaybookListResponse(
                bullets=[self._to_response(r) for r in rows], total=len(rows),
            )

    def admin_list(self, *, owner_user_id: str | None = None, project_slug: str | None = None,
                   limit: int = 500) -> PlaybookListResponse:
        """Cross-user listing for the admin console (owner filter optional)."""
        with self.session() as session:
            q = session.query(PlaybookBullet).filter(PlaybookBullet.deleted_at.is_(None))
            if owner_user_id:
                q = q.filter(PlaybookBullet.owner_user_id == owner_user_id)
            if project_slug:
                q = q.filter(PlaybookBullet.project_slug == project_slug)
            rows = q.order_by(PlaybookBullet.updated_at.desc()).limit(limit).all()
            return PlaybookListResponse(
                bullets=[self._to_response(r) for r in rows], total=len(rows),
            )

    def mark(self, bullet_id: int, *, helpful: bool, owner_user_id: str) -> PlaybookBulletResponse | None:
        with self.session() as session:
            row = (session.query(PlaybookBullet)
                   .filter(PlaybookBullet.id == bullet_id,
                           PlaybookBullet.owner_user_id == owner_user_id,
                           PlaybookBullet.deleted_at.is_(None))
                   .one_or_none())
            if row is None:
                return None
            if helpful:
                row.helpful += 1
            else:
                row.harmful += 1
            row.updated_at = _utcnow()
            session.flush()
            resp = self._to_response(row)
            session.commit()
            return resp

    def delete(self, bullet_id: int, *, owner_user_id: str) -> PlaybookDeleteResponse:
        with self.session() as session:
            row = (session.query(PlaybookBullet)
                   .filter(PlaybookBullet.id == bullet_id,
                           PlaybookBullet.owner_user_id == owner_user_id,
                           PlaybookBullet.deleted_at.is_(None))
                   .one_or_none())
            if row is None:
                return PlaybookDeleteResponse(deleted=False, bullet_id=bullet_id)
            row.deleted_at = _utcnow()
            session.commit()
            return PlaybookDeleteResponse(deleted=True, bullet_id=bullet_id)


_SERVICE: PlaybookService | None = None


def get_playbook_service() -> PlaybookService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PlaybookService()
    return _SERVICE
