from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI


def test_mempalace_is_absent_from_registered_compute_routes():
    from app.api.routes import register_routes

    app = FastAPI()
    register_routes(app)

    assert not any(route.path.startswith("/api/mempalace") for route in app.routes)


def test_mempalace_is_not_a_runtime_setting_or_dependency():
    from config import settings

    assert not hasattr(settings, "mempalace")
    assert "mempalace" not in Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert not Path("requirements-mempalace.txt").exists()
