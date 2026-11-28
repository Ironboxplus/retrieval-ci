from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.frontend_assets import (
    FrontendAssetRoots,
    FrontendAssetsUnavailable,
    PublicFrontendUnavailable,
)


def _write_bundle(root: Path, marker: str) -> None:
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(marker, encoding="utf-8")
    (root / "assets" / "app.js").write_text(marker + " asset", encoding="utf-8")


def test_public_listener_selects_its_own_bundle_for_index_and_assets(tmp_path: Path):
    internal = tmp_path / "dist"
    public = tmp_path / "dist-public"
    _write_bundle(internal, "internal bundle")
    _write_bundle(public, "public bundle")
    roots = FrontendAssetRoots(
        internal_dir=internal,
        public_dir=public,
        internal_port=18080,
        public_port=18181,
    )

    assert roots.spa_file_for(port=18080, path="search").read_text(encoding="utf-8") == "internal bundle"
    assert roots.spa_file_for(port=18181, path="search").read_text(encoding="utf-8") == "public bundle"
    assert roots.asset_file_for(port=18181, path="app.js").read_text(encoding="utf-8") == "public bundle asset"


def test_public_listener_never_falls_back_to_internal_bundle_when_public_build_is_missing(tmp_path: Path):
    internal = tmp_path / "dist"
    _write_bundle(internal, "internal bundle")
    roots = FrontendAssetRoots(
        internal_dir=internal,
        public_dir=tmp_path / "missing-public-dist",
        internal_port=18080,
        public_port=18181,
    )

    with pytest.raises(PublicFrontendUnavailable, match="public frontend bundle is unavailable"):
        roots.spa_file_for(port=18181, path="search")


def test_asset_resolution_cannot_escape_the_selected_bundle_root(tmp_path: Path):
    internal = tmp_path / "dist"
    public = tmp_path / "dist-public"
    _write_bundle(internal, "internal bundle")
    _write_bundle(public, "public bundle")
    outside = tmp_path / "private.txt"
    outside.write_text("must not serve", encoding="utf-8")
    roots = FrontendAssetRoots(
        internal_dir=internal,
        public_dir=public,
        internal_port=18080,
        public_port=18181,
    )

    assert roots.asset_file_for(port=18181, path="../private.txt") is None


def test_public_listener_rejects_an_incomplete_public_bundle(tmp_path: Path):
    internal = tmp_path / "dist"
    public = tmp_path / "dist-public"
    _write_bundle(internal, "internal bundle")
    public.mkdir()
    roots = FrontendAssetRoots(
        internal_dir=internal,
        public_dir=public,
        internal_port=18080,
        public_port=18181,
    )

    with pytest.raises(PublicFrontendUnavailable, match="public frontend bundle is unavailable"):
        roots.spa_file_for(port=18181, path="search")


def test_unknown_listener_port_never_selects_the_internal_bundle(tmp_path: Path):
    internal = tmp_path / "dist"
    public = tmp_path / "dist-public"
    _write_bundle(internal, "internal bundle")
    _write_bundle(public, "public bundle")
    roots = FrontendAssetRoots(
        internal_dir=internal,
        public_dir=public,
        internal_port=18080,
        public_port=18181,
    )

    with pytest.raises(FrontendAssetsUnavailable, match="unexpected listener port"):
        roots.spa_file_for(port=9999, path="search")


def test_static_routes_select_port_specific_bundle_and_fail_closed(tmp_path: Path, monkeypatch):
    """The real FastAPI handlers must preserve the helper's port/security rules."""
    import main

    internal = tmp_path / "dist"
    public = tmp_path / "dist-public"
    _write_bundle(internal, "internal bundle")
    _write_bundle(public, "public bundle")
    monkeypatch.setattr(
        main,
        "_frontend_assets",
        FrontendAssetRoots(
            internal_dir=internal,
            public_dir=public,
            internal_port=18080,
            public_port=18181,
        ),
    )

    internal_response = TestClient(main.app, base_url="http://testserver:18080").get("/search")
    public_response = TestClient(main.app, base_url="http://testserver:18181").get("/search")
    unknown_response = TestClient(main.app, base_url="http://testserver:19000").get("/search")
    asset_directory = TestClient(main.app, base_url="http://testserver:18181").get("/assets")

    assert internal_response.status_code == 200
    assert internal_response.text == "internal bundle"
    assert public_response.status_code == 200
    assert public_response.text == "public bundle"
    assert unknown_response.status_code == 503
    assert asset_directory.status_code == 404
    # A deploy must reach browsers: index.html revalidates, fingerprinted assets are immutable.
    assert internal_response.headers["cache-control"] == "no-cache"
    asset = TestClient(main.app, base_url="http://testserver:18080").get("/assets/app.js")
    assert asset.text == "internal bundle asset"
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_public_static_route_returns_503_for_an_incomplete_bundle(tmp_path: Path, monkeypatch):
    import main

    internal = tmp_path / "dist"
    public = tmp_path / "dist-public"
    _write_bundle(internal, "internal bundle")
    public.mkdir()
    monkeypatch.setattr(
        main,
        "_frontend_assets",
        FrontendAssetRoots(
            internal_dir=internal,
            public_dir=public,
            internal_port=18080,
            public_port=18181,
        ),
    )

    response = TestClient(
        main.app,
        base_url="http://testserver:18181",
        raise_server_exceptions=False,
    ).get("/search")

    assert response.status_code == 503
    assert "public frontend bundle is unavailable" in response.json()["detail"]
