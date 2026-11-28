"""Validated public-code source catalog and deterministic scope selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Mapping, Protocol
from urllib.parse import urlsplit


class PublicSourceCatalogError(ValueError):
    """Raised when a public-code source catalog is structurally invalid."""


class PublicSourceSelectionError(PublicSourceCatalogError):
    """Raised when a requested source or domain is not present in a catalog."""


_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_DOMAIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")


@dataclass(frozen=True, slots=True)
class PublicCodeSource:
    """One externally maintained repository admitted to the public code corpus."""

    source_id: str
    domain: str
    relative_path: str
    upstream_url: str
    default_ref: str = "HEAD"
    proxy_profile: str | None = None
    credential_profile: str | None = None
    language_overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SOURCE_ID_PATTERN.fullmatch(self.source_id):
            raise PublicSourceCatalogError(
                "source_id must be a repository identifier containing only "
                "letters, digits, dots, underscores, and hyphens"
            )
        if not _DOMAIN_PATTERN.fullmatch(self.domain):
            raise PublicSourceCatalogError(
                f"domain must be a lowercase identifier for {self.source_id}"
            )
        _validate_upstream_url(self.upstream_url, source_id=self.source_id)
        if not _is_safe_git_ref(self.default_ref):
            raise PublicSourceCatalogError(f"default_ref is invalid for {self.source_id}")
        for field_name, profile in (
            ("proxy_profile", self.proxy_profile),
            ("credential_profile", self.credential_profile),
        ):
            if profile is not None and not _PROFILE_PATTERN.fullmatch(profile):
                raise PublicSourceCatalogError(
                    f"{field_name} must be a lowercase profile identifier for {self.source_id}"
                )

        object.__setattr__(self, "relative_path", _normalize_relative_path(self.relative_path))
        normalized_overrides: dict[str, str] = {}
        for extension, language in self.language_overrides.items():
            normalized_extension = extension.strip().lower()
            normalized_language = language.strip().lower()
            if not normalized_extension.startswith(".") or "/" in normalized_extension:
                raise PublicSourceCatalogError(
                    f"language override keys must be file extensions for {self.source_id}"
                )
            if not normalized_language:
                raise PublicSourceCatalogError(
                    f"language override values must not be empty for {self.source_id}"
                )
            normalized_overrides[normalized_extension] = normalized_language
        object.__setattr__(self, "language_overrides", normalized_overrides)


class PublicCodeSourceSelector(Protocol):
    """Minimal source-selection contract shared by bootstrap catalogs and SQL registries."""

    def select(
        self,
        *,
        source_ids: Iterable[str] = (),
        domains: Iterable[str] = (),
    ) -> tuple[PublicCodeSource, ...]: ...


@dataclass(frozen=True, slots=True)
class PublicCodeSourceCatalog:
    """One source catalog with source and domain selectors for every caller."""

    catalog_id: str
    sources: tuple[PublicCodeSource, ...]

    def __post_init__(self) -> None:
        if not self.catalog_id.strip():
            raise PublicSourceCatalogError("catalog_id must not be empty")

        source_ids = [source.source_id for source in self.sources]
        paths = [source.relative_path for source in self.sources]
        duplicate_source_ids = _duplicates(source_ids)
        if duplicate_source_ids:
            raise PublicSourceCatalogError(
                f"duplicate public-code source IDs: {', '.join(duplicate_source_ids)}"
            )
        duplicate_paths = _duplicates(paths)
        if duplicate_paths:
            raise PublicSourceCatalogError(
                f"duplicate public-code relative paths: {', '.join(duplicate_paths)}"
            )

        object.__setattr__(self, "sources", tuple(sorted(self.sources, key=lambda source: source.source_id)))

    @classmethod
    def from_json_file(cls, path: Path) -> "PublicCodeSourceCatalog":
        """Load the checked-in public source declaration without I/O side effects."""

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PublicSourceCatalogError(f"cannot read public-code catalog: {path}") from exc
        except json.JSONDecodeError as exc:
            raise PublicSourceCatalogError(f"invalid public-code catalog JSON: {path}") from exc

        catalog_id = payload.get("catalog_id") if isinstance(payload, Mapping) else None
        raw_sources = payload.get("repositories") if isinstance(payload, Mapping) else None
        if not isinstance(catalog_id, str) or not isinstance(raw_sources, list):
            raise PublicSourceCatalogError("catalog must contain catalog_id and repositories")

        try:
            sources = tuple(PublicCodeSource(**entry) for entry in raw_sources)
        except (TypeError, PublicSourceCatalogError) as exc:
            raise PublicSourceCatalogError("catalog contains an invalid public-code source") from exc
        return cls(catalog_id=catalog_id, sources=sources)

    @property
    def domains(self) -> tuple[str, ...]:
        """All configured domains in deterministic order for UI and API validation."""

        return tuple(sorted({source.domain for source in self.sources}))

    def select(
        self,
        *,
        source_ids: Iterable[str] = (),
        domains: Iterable[str] = (),
    ) -> tuple[PublicCodeSource, ...]:
        """Resolve explicit sources plus all sources from selected domains.

        Source and domain selectors are inclusive so a UI can combine individual
        repositories with one or more broader domain chips in one request.
        """

        requested_source_ids = frozenset(source_ids)
        requested_domains = frozenset(domains)
        known_sources = {source.source_id for source in self.sources}
        known_domains = set(self.domains)

        unknown_source_ids = sorted(requested_source_ids - known_sources)
        if unknown_source_ids:
            raise PublicSourceSelectionError(
                f"Unknown public-code source IDs: {', '.join(unknown_source_ids)}"
            )
        unknown_domains = sorted(requested_domains - known_domains)
        if unknown_domains:
            raise PublicSourceSelectionError(
                f"Unknown public-code domains: {', '.join(unknown_domains)}"
            )

        if not requested_source_ids and not requested_domains:
            return self.sources

        return tuple(
            source
            for source in self.sources
            if source.source_id in requested_source_ids or source.domain in requested_domains
        )


def _normalize_relative_path(path: str) -> str:
    normalized_input = path.replace("\\", "/")
    if not normalized_input or normalized_input.startswith("/") or len(normalized_input) >= 2 and normalized_input[1] == ":":
        raise PublicSourceCatalogError("relative_path must be a non-empty relative path")
    parts = normalized_input.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PublicSourceCatalogError("relative_path must be normalized and remain within the mirror root")
    normalized_path = str(PurePosixPath(normalized_input))
    if normalized_path == ".":
        raise PublicSourceCatalogError("relative_path must be a non-empty relative path")
    if PurePosixPath(normalized_path).name.lower().endswith(".git"):
        raise PublicSourceCatalogError("relative_path must omit the .git storage suffix")
    return normalized_path


def _validate_upstream_url(url: str, *, source_id: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PublicSourceCatalogError(f"upstream_url must use HTTPS for {source_id}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PublicSourceCatalogError(
            f"upstream_url must not contain credentials, query parameters, or fragments for {source_id}"
        )


def _is_safe_git_ref(ref: str) -> bool:
    return bool(
        _GIT_REF_PATTERN.fullmatch(ref)
        and ".." not in ref
        and "//" not in ref
        and not ref.endswith((".", "/"))
    )


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))
