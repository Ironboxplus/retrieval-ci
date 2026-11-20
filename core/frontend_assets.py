"""Port-aware selection of the internal and public Vite bundles.

Both HTTP listeners use one FastAPI application.  The static layer must therefore
select a bundle from the socket's local port, not from the request source address
or a caller-controlled header.  In particular, a missing public build must never
fall back to the internal admin build.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class FrontendAssetsUnavailable(RuntimeError):
    """Raised when the listener's required Vite bundle has not been deployed."""


class PublicFrontendUnavailable(FrontendAssetsUnavailable):
    """Raised instead of leaking the internal bundle from the public listener."""


@dataclass(frozen=True, slots=True)
class FrontendAssetRoots:
    """Internal/public Vite roots and their trusted listener ports."""

    internal_dir: Path
    public_dir: Path
    internal_port: int
    public_port: int

    def _root_for_port(self, port: int | None) -> Path:
        if port == self.public_port:
            root = self.public_dir
            unavailable = PublicFrontendUnavailable
            label = "public"
        elif port == self.internal_port:
            root = self.internal_dir
            unavailable = FrontendAssetsUnavailable
            label = "internal"
        else:
            raise FrontendAssetsUnavailable(
                f"frontend bundle is unavailable: unexpected listener port {port!r}"
            )

        # A directory alone is not a deployed SPA: FileResponse would turn a
        # missing fallback file into a late 500.  Both doors instead expose a
        # deterministic 503 until their selected Vite index is complete.
        if root.is_dir() and (root / "index.html").is_file():
            return root.resolve()
        raise unavailable(
            f"{label} frontend bundle is unavailable: expected directory and index.html under {root}"
        )

    @staticmethod
    def _candidate(root: Path, path: str) -> Path | None:
        # Browser URLs use slash separators, but normalize backslashes too so a
        # crafted Windows-looking path cannot escape the static root on Windows.
        parts = [part for part in path.replace("\\", "/").strip("/").split("/")
                 if part and part != "."]
        if any(part == ".." for part in parts):
            return None
        candidate = (root.joinpath(*parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def asset_file_for(self, *, port: int | None, path: str) -> Path | None:
        """Return an existing asset file from the selected bundle, or ``None`` for 404."""
        root = self._root_for_port(port)
        candidate = self._candidate(root / "assets", path)
        return candidate if candidate is not None and candidate.is_file() else None

    def spa_file_for(self, *, port: int | None, path: str) -> Path:
        """Return a real static file or the selected bundle's SPA index fallback."""
        root = self._root_for_port(port)
        candidate = self._candidate(root, path)
        if candidate is not None and candidate.is_file():
            return candidate
        return root / "index.html"
