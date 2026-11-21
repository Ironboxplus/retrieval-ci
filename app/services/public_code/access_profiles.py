"""Resolve public Git access references without persisting or returning secrets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit

from app.services.public_code.source_catalog import PublicCodeSource


_PROXY_PREFIX = "PUBLIC_CODE_PROXY_PROFILE_"
_CREDENTIAL_PREFIX = "PUBLIC_CODE_CREDENTIAL_PROFILE_"
_CREDENTIAL_FIELDS = ("allowed_hosts", "authorization_header", "cookie")


@dataclass(frozen=True, slots=True)
class _CredentialProfile:
    allowed_hosts: tuple[str, ...]
    headers: tuple[str, ...]


class GitAccessProfileResolver:
    """Convert source profile names into a subprocess-only Git environment."""

    def __init__(
        self,
        *,
        default_proxy_url: str | None,
        environ: Mapping[str, str] | None = None,
    ):
        self._default_proxy_url = default_proxy_url
        self._environ = environ if environ is not None else os.environ

    def environment_for(
        self,
        source: PublicCodeSource,
        *,
        network: bool = True,
    ) -> dict[str, str]:
        environment = {
            key: value
            for key, value in self._environ.items()
            if not key.startswith((_PROXY_PREFIX, _CREDENTIAL_PREFIX))
        }
        environment["GIT_TERMINAL_PROMPT"] = "0"
        if not network:
            return environment
        proxy_url = self._proxy_url(source.proxy_profile)
        if proxy_url:
            environment["http_proxy"] = proxy_url
            environment["https_proxy"] = proxy_url
        credential_profile = self._credential_profile(source.credential_profile)
        if credential_profile is not None:
            parsed_upstream = urlsplit(source.upstream_url)
            hostname = (parsed_upstream.hostname or "").lower()
            if not any(_host_matches(hostname, allowed) for allowed in credential_profile.allowed_hosts):
                raise ValueError(
                    f"public-code credential_profile {source.credential_profile} "
                    f"does not allow Git host {hostname}"
                )
            scoped_key = f"http.{parsed_upstream.scheme}://{parsed_upstream.netloc}/.extraHeader"
            environment["GIT_CONFIG_COUNT"] = str(len(credential_profile.headers))
            for index, header in enumerate(credential_profile.headers):
                environment[f"GIT_CONFIG_KEY_{index}"] = scoped_key
                environment[f"GIT_CONFIG_VALUE_{index}"] = header
        return environment

    def validate_source(self, source: PublicCodeSource) -> None:
        """Resolve network access references without returning their secret values."""

        self.environment_for(source, network=True)

    def _proxy_url(self, profile: str | None) -> str | None:
        if profile is None:
            return self._default_proxy_url
        environment_key = _PROXY_PREFIX + _environment_suffix(profile)
        proxy_url = self._environ.get(environment_key)
        if not proxy_url:
            raise ValueError(f"unknown public-code proxy_profile: {profile}")
        if not proxy_url.startswith(("http://", "https://")):
            raise ValueError(f"invalid public-code proxy_profile: {profile}")
        return proxy_url

    def _credential_profile(self, profile: str | None) -> _CredentialProfile | None:
        if profile is None:
            return None
        environment_key = _CREDENTIAL_PREFIX + _environment_suffix(profile)
        raw_profile = self._environ.get(environment_key)
        if not raw_profile:
            raise ValueError(f"unknown public-code credential_profile: {profile}")
        try:
            payload: Any = json.loads(raw_profile)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid public-code credential_profile: {profile}") from exc
        if not isinstance(payload, dict) or set(payload) - set(_CREDENTIAL_FIELDS):
            raise ValueError(f"invalid public-code credential_profile: {profile}")
        allowed_hosts = payload.get("allowed_hosts")
        if not isinstance(allowed_hosts, list) or not allowed_hosts:
            raise ValueError(f"invalid public-code credential_profile: {profile}")
        normalized_hosts: list[str] = []
        for host in allowed_hosts:
            if not isinstance(host, str) or not _is_valid_host_pattern(host):
                raise ValueError(f"invalid public-code credential_profile: {profile}")
            normalized_hosts.append(host.lower())
        headers: list[str] = []
        for field_name, prefix in (
            ("authorization_header", "Authorization: "),
            ("cookie", "Cookie: "),
        ):
            value = payload.get(field_name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
                raise ValueError(f"invalid public-code credential_profile: {profile}")
            headers.append(prefix + value.strip())
        if not headers:
            raise ValueError(f"invalid public-code credential_profile: {profile}")
        return _CredentialProfile(
            allowed_hosts=tuple(sorted(set(normalized_hosts))),
            headers=tuple(headers),
        )


def _environment_suffix(profile: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", profile).upper()


def _is_valid_host_pattern(host: str) -> bool:
    candidate = host.strip().lower()
    if candidate.startswith("*."):
        candidate = candidate[2:]
    return bool(candidate and re.fullmatch(r"[a-z0-9.-]+", candidate))


def _host_matches(hostname: str, allowed_host: str) -> bool:
    if allowed_host.startswith("*."):
        suffix = allowed_host[1:]
        return hostname.endswith(suffix) and hostname != allowed_host[2:]
    return hostname == allowed_host
