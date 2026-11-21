"""Small real FastAPI process used by the isolated Journal bulk black-box test.

It deliberately registers the production routes and starts the production SQLite
schema.  Keeping it separate from ``main`` avoids loading papers search/reranker
components that are irrelevant to this lifecycle, while Journal itself still uses
its normal process-wide shared BGE-M3 embedder and real Qdrant client.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import sys

from fastapi import FastAPI

from app.api.routes import register_routes
from core.db.database import init_db


# ``main.py`` installs the production structured logger before routes are loaded.
# This intentionally narrower test app does not import that heavy entry point, so
# retain the exact Journal INFO event in the child process transcript.
_journal_logger = logging.getLogger("journal.service")
_journal_logger.setLevel(logging.INFO)
_journal_logger.addHandler(logging.StreamHandler(sys.stdout))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="journal-bulk-blackbox", lifespan=lifespan)
register_routes(app)
