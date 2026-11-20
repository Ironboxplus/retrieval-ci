"""HTTP mapping of errors that every route shares."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.indexing.embedders.bge_m3_core import InputTooLong


async def _input_too_long(_: Request, exc: InputTooLong) -> JSONResponse:
    """A query longer than the embedding model's sequence limit is the caller's input, not a
    server fault: say how long it was instead of returning a 500."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "input_tokens": exc.tokens, "token_limit": exc.limit},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(InputTooLong, _input_too_long)
