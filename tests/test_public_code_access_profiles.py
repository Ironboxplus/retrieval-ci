"""TDD for server-side public Git proxy and credential references."""

from __future__ import annotations

import json

import pytest

from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.source_catalog import PublicCodeSource


def source(**overrides) -> PublicCodeSource:
    values = {
        "source_id": "pion-webrtc",
        "domain": "multimedia",
        "relative_path": "multimedia/pion-webrtc",
        "upstream_url": "https://github.com/pion/webrtc.git",
        "proxy_profile": "local-proxy",
        "credential_profile": "github-read",
    }
    values.update(overrides)
    return PublicCodeSource(**values)


def test_profile_references_resolve_to_git_environment_without_changing_source_metadata():
    resolver = GitAccessProfileResolver(
        default_proxy_url="http://127.0.0.1:10808",
        environ={
            "PUBLIC_CODE_PROXY_PROFILE_LOCAL_PROXY": "http://proxy.internal:10808",
            "PUBLIC_CODE_CREDENTIAL_PROFILE_GITHUB_READ": json.dumps({
                "allowed_hosts": ["github.com"],
                "authorization_header": "Bearer server-side-token",
                "cookie": "session=server-side-cookie",
            }),
        },
    )

    environment = resolver.environment_for(source())

    assert environment["http_proxy"] == "http://proxy.internal:10808"
    assert environment["https_proxy"] == "http://proxy.internal:10808"
    assert environment["GIT_CONFIG_COUNT"] == "2"
    assert environment["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
    assert environment["GIT_CONFIG_VALUE_0"] == "Authorization: Bearer server-side-token"
    assert environment["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraHeader"
    assert environment["GIT_CONFIG_VALUE_1"] == "Cookie: session=server-side-cookie"
    assert "PUBLIC_CODE_CREDENTIAL_PROFILE_GITHUB_READ" not in environment
    assert source().credential_profile == "github-read"


def test_credential_profile_is_rejected_for_an_unlisted_git_host():
    resolver = GitAccessProfileResolver(
        default_proxy_url="http://127.0.0.1:10808",
        environ={
            "PUBLIC_CODE_CREDENTIAL_PROFILE_GITHUB_READ": json.dumps({
                "allowed_hosts": ["github.com"],
                "authorization_header": "Bearer server-side-token",
            }),
        },
    )

    with pytest.raises(ValueError, match="does not allow Git host attacker.example"):
        resolver.validate_source(source(
            upstream_url="https://attacker.example/repo.git",
            proxy_profile=None,
        ))


@pytest.mark.parametrize("field_name", ["proxy_profile", "credential_profile"])
def test_unknown_profile_is_rejected_before_git_network_work(field_name):
    resolver = GitAccessProfileResolver(
        default_proxy_url="http://127.0.0.1:10808",
        environ={},
    )
    overrides = {field_name: "missing-profile"}
    overrides["credential_profile" if field_name == "proxy_profile" else "proxy_profile"] = None

    with pytest.raises(ValueError, match=f"unknown public-code {field_name}"):
        resolver.environment_for(source(**overrides))

    local_environment = resolver.environment_for(source(**overrides), network=False)
    assert "http_proxy" not in local_environment
    assert "GIT_CONFIG_COUNT" not in local_environment
