"""Composable Qdrant filters for versioned scopes in shared collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


def build_scope_pair_filter(
    active_scopes: Sequence[Mapping[str, str]],
    *,
    payload_filters: Mapping[str, Any] | None = None,
) -> Filter:
    """Build ``common filters AND (scope A OR scope B ...)`` without post-filtering.

    Versioned public collections need a source and revision to remain paired.
    Two independent ``MatchAny`` conditions would allow invalid cross-pairs, so
    every active scope is emitted as its own nested ``must`` filter under
    Qdrant's ``should`` clause.
    """

    scopes = tuple(dict(scope) for scope in active_scopes)
    if not scopes:
        raise ValueError("at least one active scope is required")

    expected_fields = frozenset(scopes[0])
    if not expected_fields or any(frozenset(scope) != expected_fields for scope in scopes):
        raise ValueError("all active scopes must contain the same fields")
    if any(not value for scope in scopes for value in scope.values()):
        raise ValueError("active scope fields must not be empty")

    common_conditions = _payload_conditions(payload_filters or {})
    scope_conditions = [
        Filter(must=_payload_conditions(scope))
        for scope in scopes
    ]
    return Filter(must=common_conditions or None, should=scope_conditions)


def _payload_conditions(filters: Mapping[str, Any]) -> list[FieldCondition]:
    conditions: list[FieldCondition] = []
    for key in sorted(filters):
        value = filters[key]
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return conditions
