"""Path-safety rules shared by every module that stores or filters relative paths."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def safe_relative_path(
    path: str,
    *,
    what: str = "path",
    root: str = "scope",
    allow_trailing_slash: bool = False,
    reject_empty_segments: bool = True,
    error_cls: type[Exception] = ValueError,
) -> str:
    """Return ``path`` as a forward-slash relative path that cannot leave its root.

    Backslashes are accepted as Windows input. Absolute, drive-relative (``E:x``) and
    UNC spellings are refused, and so is any ``..`` component: the result is only ever
    joined under ``root``. ``PureWindowsPath`` / ``PurePosixPath`` decide what is a
    drive or root, so a Windows spelling is refused on a POSIX server too.

    ``reject_empty_segments`` refuses ``a//b`` and ``./a`` instead of collapsing them
    (manifest entries must already be normalised). ``allow_trailing_slash`` keeps one
    trailing ``/`` on the result, to mark a directory filter.

    ``what`` and ``root`` only word the ``error_cls`` message.
    """
    raw = (path or "").replace("\\", "/")
    windows = PureWindowsPath(raw)
    if windows.drive or windows.root or PurePosixPath(raw).is_absolute():
        raise error_cls(f"{what} must be {root}-relative, not absolute")
    trailing_slash = allow_trailing_slash and raw.endswith("/")
    body = raw[:-1] if trailing_slash else raw
    if reject_empty_segments:
        parts = body.split("/")
        if body and any(part in {"", "."} for part in parts):
            raise error_cls(f"{what} must be normalized: no empty or '.' components")
        parts = [part for part in parts if part]
    else:
        parts = list(PurePosixPath(body).parts)
    if ".." in parts:
        raise error_cls(f"{what} must not contain '..' components")
    if not parts:
        raise error_cls(f"{what} must name a {root}-relative file or directory")
    normalized = "/".join(parts)
    return normalized + "/" if trailing_slash else normalized
