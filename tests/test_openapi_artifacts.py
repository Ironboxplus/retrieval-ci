from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_import_private_scope_contract_matches_current_models():
    from fastapi import FastAPI
    from app.api.routes.public_code import router

    app = FastAPI()
    app.include_router(router, prefix="/api/public-code")
    fresh = app.openapi()
    document = json.loads((ROOT / "docs" / "openapi.json").read_text(encoding="utf-8"))
    for name in ("PublicCodeImportCreateRequest", "PublicCodeImportResponse"):
        assert document["components"]["schemas"][name] == fresh["components"]["schemas"][name]


def _public_compute_contract() -> dict[str, list[str]]:
    tree = ast.parse(
        (ROOT / "scripts" / "gen_public_openapi.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "PUBLIC_COMPUTE"
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("PUBLIC_COMPUTE was not found in gen_public_openapi.py")


def test_generated_openapi_contains_every_public_compute_route():
    source = json.loads((ROOT / "docs" / "openapi.json").read_text(encoding="utf-8"))
    public = json.loads(
        (ROOT / "frontend" / "public" / "openapi.json").read_text(encoding="utf-8")
    )

    for path, methods in _public_compute_contract().items():
        assert path in source["paths"], f"source OpenAPI is missing {path}"
        assert path in public["paths"], f"public OpenAPI is missing {path}"
        for method in methods:
            assert method in source["paths"][path], f"source OpenAPI is missing {method} {path}"
            assert method in public["paths"][path], f"public OpenAPI is missing {method} {path}"


def test_paper_artifact_openapi_declares_binary_responses_and_limits():
    for path in (ROOT / "docs" / "openapi.json",
                 ROOT / "frontend" / "public" / "openapi.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        markdown = document["paths"]["/api/papers/{paper_id}/markdown"]["get"]
        batch = document["paths"]["/api/papers/download-batch"]["post"]

        assert "text/markdown" in markdown["responses"]["200"]["content"]
        assert "application/zip" in batch["responses"]["200"]["content"]
        assert {"413", "416", "429"} <= set(batch["responses"])
        assert "Accept-Ranges" in batch["responses"]["200"]["headers"]
        assert "Accept-Ranges" in batch["responses"]["416"]["headers"]
        assert "Retry-After" in batch["responses"]["429"]["headers"]


def test_public_openapi_declares_paper_billing_and_member_gates():
    public = json.loads(
        (ROOT / "frontend" / "public" / "openapi.json").read_text(encoding="utf-8")
    )
    paths = public["paths"]

    assert "402" in paths["/api/papers/{paper_id}/fulltext"]["get"]["responses"]
    assert "402" in paths["/api/papers/{paper_id}/markdown"]["get"]["responses"]
    assert "403" in paths["/api/papers/{paper_id}/pdf"]["get"]["responses"]
    batch_responses = paths["/api/papers/download-batch"]["post"]["responses"]
    assert {"402", "403"} <= set(batch_responses)


def test_public_openapi_declares_retry_safe_billing_contract():
    public = json.loads(
        (ROOT / "frontend" / "public" / "openapi.json").read_text(encoding="utf-8")
    )
    metered = (
        ("/api/search", "post"),
        ("/api/search/stream", "post"),
        ("/api/match-doc", "post"),
        ("/api/papers/{paper_id}", "get"),
        ("/api/papers/{paper_id}/fulltext", "get"),
        ("/api/papers/{paper_id}/markdown", "get"),
        ("/api/papers/download-batch", "post"),
    )

    for path, method in metered:
        operation = public["paths"][path][method]
        parameters = {
            parameter["name"]: parameter for parameter in operation.get("parameters", [])
        }
        assert parameters["X-Idempotency-Key"]["in"] == "header"
        assert {"409", "425"} <= set(operation["responses"])


def test_public_openapi_exposes_admin_only_public_code_contract():
    public = json.loads(
        (ROOT / "frontend" / "public" / "openapi.json").read_text(encoding="utf-8")
    )

    for path, method in (
        ("/api/public-code/sources", "get"),
        ("/api/public-code/sources", "post"),
        ("/api/public-code/sources/check-updates", "post"),
        ("/api/public-code/sources/index", "post"),
        ("/api/public-code/jobs", "get"),
        ("/api/public-code/jobs/{job_id}", "get"),
        ("/api/public-code/sources/{source_id}/update-policy", "put"),
        ("/api/public-code/search", "post"),
    ):
        operation = public["paths"][path][method]
        assert "403" in operation["responses"]
        assert operation["security"] == [{"ApiKeyAuth": []}, {"SessionCookie": []}]
        parameter_names = {
            parameter.get("name") for parameter in operation.get("parameters", [])
        }
        assert not parameter_names.intersection({"X-Retrival-User-Id", "X-Retrival-User-Tier"})
        if method == "post" and path == "/api/public-code/sources":
            success_status = "201"
        elif method == "post" and path.endswith(("check-updates", "index")):
            success_status = "202"
        else:
            success_status = "200"
        schema = operation["responses"][success_status]["content"]["application/json"]["schema"]
        assert schema not in ({}, None)


def test_public_code_search_artifacts_preserve_retrieval_and_reranker_scores():
    for path in (ROOT / "docs" / "openapi.json",
                 ROOT / "frontend" / "public" / "openapi.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        properties = document["components"]["schemas"]["PublicCodeSearchHitResponse"]["properties"]
        assert {"score", "retrieval_score", "reranker_score"} <= set(properties)
        response_properties = document["components"]["schemas"]["PublicCodeSearchResponseBody"]["properties"]
        assert {
            "requested_mode",
            "effective_mode",
            "notice",
            "rerank_applied",
            "candidate_count",
            "rerank_candidate_count",
        } <= set(response_properties)


def test_code_search_openapi_exposes_recovery_and_catalog_visibility_states():
    source = json.loads((ROOT / "docs" / "openapi.json").read_text(encoding="utf-8"))
    properties = source["components"]["schemas"]["CodeSearchResponse"]["properties"]

    assert "recovery_required" in properties
    assert {"completed", "preparing", "recovery_required", "unavailable"} <= set(
        properties["scope_snapshot_state"]["enum"]
    )


def test_public_openapi_defines_every_referenced_security_scheme():
    document = json.loads(
        (ROOT / "frontend" / "public" / "openapi.json").read_text(encoding="utf-8")
    )
    defined = set(document.get("components", {}).get("securitySchemes", {}))
    referenced = {
        scheme_name
        for path_item in document["paths"].values()
        for operation in path_item.values()
        for requirement in operation.get("security", [])
        for scheme_name in requirement
    }

    assert referenced
    assert referenced <= defined
