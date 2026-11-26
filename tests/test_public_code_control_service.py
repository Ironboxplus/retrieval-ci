"""Public-code control operations share catalog reconciliation and durable state."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.services.public_code.control_service import PublicCodeControlService
from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceCatalog
from app.services.public_code.update_checks import PublicCodeUpdateCheck


def catalog() -> PublicCodeSourceCatalog:
    return PublicCodeSourceCatalog(
        catalog_id="fixture",
        sources=(
            PublicCodeSource("ffmpeg", "multimedia", "multimedia/ffmpeg", "https://example.invalid/ffmpeg.git"),
        ),
    )


@dataclass
class FakeRepository:
    recorded_checks: list[tuple[PublicCodeUpdateCheck, ...]] = field(default_factory=list)
    policy_calls: list[dict] = field(default_factory=list)
    create_calls: list[dict] = field(default_factory=list)

    def list_sources(self):
        return ("source-state",)

    def record_update_checks(self, checks):
        self.recorded_checks.append(tuple(checks))

    def set_update_policy(self, **kwargs):
        self.policy_calls.append(kwargs)
        return kwargs

    def create_source(self, source, **kwargs):
        call = {"source": source, **kwargs}
        self.create_calls.append(call)
        return call


@dataclass
class FakeUpdateChecks:
    calls: list[dict] = field(default_factory=list)

    def check(self, **kwargs):
        self.calls.append(kwargs)
        return (
            PublicCodeUpdateCheck(
                source_id="ffmpeg",
                domain="multimedia",
                local_revision="old",
                remote_revision="new",
                status="update_available",
            ),
        )


def test_check_updates_persists_each_observed_result():
    repository = FakeRepository()
    checks = FakeUpdateChecks()
    service = PublicCodeControlService(repository=repository, update_checks=checks)

    result = service.check_updates(source_ids=("ffmpeg",))

    assert checks.calls == [{
        "source_ids": ("ffmpeg",),
        "domains": (),
        "progress_cb": None,
    }]
    assert repository.recorded_checks == [result]


def test_update_policy_persists_operator_choice():
    repository = FakeRepository()
    service = PublicCodeControlService(repository=repository, update_checks=FakeUpdateChecks())

    state = service.set_update_policy(
        source_id="ffmpeg",
        update_policy="manual",
        update_interval_minutes=None,
    )

    assert state["update_policy"] == "manual"
    assert repository.policy_calls == [{
        "source_id": "ffmpeg",
        "update_policy": "manual",
        "update_interval_minutes": None,
    }]


def test_register_source_reuses_catalog_validation_and_repository_persistence():
    repository = FakeRepository()
    service = PublicCodeControlService(
        repository=repository,
        update_checks=FakeUpdateChecks(),
    )
    source = PublicCodeSource(
        source_id="pion-webrtc",
        domain="multimedia",
        relative_path="multimedia/pion-webrtc",
        upstream_url="https://github.com/pion/webrtc.git",
        proxy_profile="local-proxy",
    )

    state = service.register_source(
        source,
        update_policy="manual",
        update_interval_minutes=None,
        created_by_user_id="gateway-admin",
    )

    assert state["source"] == source
    assert repository.create_calls == [{
        "source": source,
        "update_policy": "manual",
        "update_interval_minutes": None,
        "created_by_user_id": "gateway-admin",
    }]


def test_register_source_validates_server_side_profiles_before_persisting():
    repository = FakeRepository()
    validated: list[PublicCodeSource] = []

    def reject_unknown_profile(source):
        validated.append(source)
        raise ValueError("unknown public-code proxy_profile: missing")

    service = PublicCodeControlService(
        repository=repository,
        update_checks=FakeUpdateChecks(),
        source_validator=reject_unknown_profile,
    )
    source = PublicCodeSource(
        source_id="pion-webrtc",
        domain="multimedia",
        relative_path="multimedia/pion-webrtc",
        upstream_url="https://github.com/pion/webrtc.git",
        proxy_profile="missing",
    )

    with pytest.raises(ValueError, match="unknown public-code proxy_profile"):
        service.register_source(
            source,
            update_policy="manual",
            update_interval_minutes=None,
        )

    assert validated == [source]
    assert repository.create_calls == []
