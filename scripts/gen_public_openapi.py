"""Generate the PUBLIC OpenAPI spec served at https://retrieval.rnarket.com/openapi.json.

The 4090 backend's full `docs/openapi.json` documents every route incl. internal
admin/pipeline/crawl/document management — those are NOT reachable on the public
edge (traefik only whitelists the paths below) and must not be advertised. This
filters to the public surface, adds the gateway's auth/billing endpoints (which
live on the gateway, not in the 4090 spec), sets the public server URL, and keeps
the security schemes. Output → frontend/public/openapi.json (ships with the SPA build).

    python scripts/gen_public_openapi.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "openapi.json"
OUT = ROOT / "frontend" / "public" / "openapi.json"

# Public COMPUTE paths (proxied through the gateway to the 4090). Methods listed are
# the ones actually reachable publicly; paper artifacts also expose one authenticated
# batch-download POST that returns a streamed ZIP.
PUBLIC_COMPUTE = {
    "/api/search": ["post"],
    "/api/search/stream": ["post"],
    "/api/match-doc": ["post"],
    "/api/conferences": ["get"],
    "/api/stats": ["get"],
    "/api/dashboard/distribution": ["get"],
    "/api/papers/": ["get"],
    "/api/papers/browse": ["get"],
    "/api/papers/tags": ["get"],
    "/api/papers/stats": ["get"],
    "/api/papers/search": ["get"],
    "/api/papers/{paper_id}": ["get"],
    "/api/papers/{paper_id}/fulltext": ["get"],
    "/api/papers/{paper_id}/markdown": ["get"],
    "/api/papers/{paper_id}/pdf": ["get"],
    "/api/papers/{paper_id}/bibtex": ["get"],
    "/api/papers/download-batch": ["post"],
    "/api/public-code/sources": ["get", "post"],
    "/api/public-code/sources/check-updates": ["post"],
    "/api/public-code/sources/index": ["post"],
    "/api/public-code/jobs": ["get"],
    "/api/public-code/jobs/{job_id}": ["get"],
    "/api/public-code/sources/{source_id}/update-policy": ["put"],
    "/api/public-code/search": ["post"],
    "/api/code/federated-search": ["post"],
}

d = json.load(open(SRC, encoding="utf-8"))

# 1) filter paths to the public compute surface
new_paths = {}
for path, methods in PUBLIC_COMPUTE.items():
    if path not in d["paths"]:
        print("  WARN missing in source spec:", path)
        continue
    kept = {m: op for m, op in d["paths"][path].items() if m in methods}
    if kept:
        new_paths[path] = kept

# Gateway-only access policy is not visible in the compute tier's generated spec.
# Annotate the public contract after filtering so SDKs can handle credit and member
# gates without pretending the 4090 itself owns user identity.
for path in ("/api/papers/{paper_id}/fulltext", "/api/papers/{paper_id}/markdown"):
    new_paths[path]["get"]["responses"]["402"] = {
        "description": "Insufficient credits for a paying user's Markdown view.",
    }
new_paths["/api/papers/{paper_id}/pdf"]["get"]["responses"]["403"] = {
    "description": "Original PDF access requires member/internal/admin tier.",
}
batch_responses = new_paths["/api/papers/download-batch"]["post"]["responses"]
batch_responses["402"] = {
    "description": "Insufficient credits for a paying user's Markdown batch.",
}
batch_responses["403"] = {
    "description": "PDF or mixed batch access requires member/internal/admin tier.",
}
for path, methods in new_paths.items():
    if not path.startswith("/api/public-code"):
        continue
    for operation in methods.values():
        operation["security"] = [{"ApiKeyAuth": []}, {"SessionCookie": []}]
        operation["parameters"] = [
            parameter
            for parameter in operation.get("parameters", [])
            if parameter.get("name") not in {
                "X-Retrival-User-Id",
                "X-Retrival-User-Tier",
            }
        ]
        operation["description"] = (
            "Public gateway access is restricted to the admin tier. "
            + (operation.get("description") or "")
        ).strip()
        operation["responses"].setdefault(
            "401", {"description": "Authentication required."}
        )
        operation["responses"]["403"] = {
            "description": "Public Code requires admin tier on the public gateway.",
        }

# Metered calls accept a caller-generated logical request key. The gateway scopes it
# to the authenticated user and request semantics; retries of the same request are
# charged once. A concurrent duplicate receives 425 until the owner finishes;
# reusing the key for a different request is rejected with 409.
idempotency_parameter = {
    "name": "X-Idempotency-Key",
    "in": "header",
    "required": False,
    "schema": {"type": "string"},
    "description": (
        "Stable logical request key for retry-safe billing. Reuse only for the exact "
        "same request. A concurrent duplicate returns 425 until the first request "
        "finishes; a different request with the same key returns 409."
    ),
}
metered_operations = (
    ("/api/search", "post"),
    ("/api/search/stream", "post"),
    ("/api/match-doc", "post"),
    ("/api/papers/{paper_id}", "get"),
    ("/api/papers/{paper_id}/fulltext", "get"),
    ("/api/papers/{paper_id}/markdown", "get"),
    ("/api/papers/download-batch", "post"),
)
for path, method in metered_operations:
    operation = new_paths[path][method]
    operation.setdefault("parameters", []).append(idempotency_parameter)
    operation["responses"]["409"] = {
        "description": "Idempotency key conflicts with a different request.",
    }
    operation["responses"]["425"] = {
        "description": "Another request with this idempotency key is still running.",
    }

# 2) add the gateway auth / billing endpoints (not in the 4090 spec)
def op(tag, summary, description, *, body=None, params=None, sec=True):
    o = {"tags": [tag], "summary": summary, "description": description,
         "responses": {"200": {"description": "OK"}}}
    if params:
        o["parameters"] = params
    if body:
        o["requestBody"] = {"required": True, "content": {"application/json": {"schema": body}}}
    if sec:
        o["security"] = [{"ApiKeyAuth": []}, {"SessionCookie": []}]
    return o

new_paths["/auth/providers"] = {"get": op(
    "账户", "可用登录方式", "返回启用的登录方式 `{cdk,linuxdo,google}`，前端据此渲染登录选择器。匿名可调。", sec=False)}
new_paths["/auth/me"] = {"get": op(
    "账户", "当前用户", "返回 `{username,tier,balance,identities}`。需登录（会话 Cookie 或 sk-key）。")}
new_paths["/auth/keys"] = {
    "get": op("账户", "列出 sk-key", "列出本人的 API 密钥（仅 prefix/last4，不返回明文）。"),
    "post": op("账户", "创建 sk-key", "签发一个新的 sk-key（明文**仅返回一次**），用于 MCP/脚本 `Authorization: Bearer sk-…`。"),
}
new_paths["/auth/keys/{key_id}"] = {"delete": op(
    "账户", "吊销 sk-key", "吊销本人某个 sk-key。", params=[
        {"name": "key_id", "in": "path", "required": True, "schema": {"type": "integer"}}])}
new_paths["/api/topup"] = {"post": op(
    "充值", "创建充值订单", "创建 linux.do Credit 充值订单，返回 `pay_url` 跳转支付。需登录。",
    body={"type": "object", "properties": {"credits": {"type": "integer"}}})}

d["paths"] = new_paths

# 3) public server + trim openapi-doc metadata
d["servers"] = [{"url": "https://retrieval.rnarket.com", "description": "公网生产（BWH gateway → 4090 算力）"}]
d["info"]["title"] = "Retrieval 公开检索 API"
d.setdefault("components", {})["securitySchemes"] = {
    "ApiKeyAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "sk-...",
        "description": "Retrieval API key passed as Authorization: Bearer sk-...",
    },
    "SessionCookie": {
        "type": "apiKey",
        "in": "cookie",
        "name": "session",
        "description": "Signed browser session cookie issued by the public gateway.",
    },
}

# 4) prune components.schemas to the $ref-closure of the kept paths — otherwise the
#    Swagger "Schemas" section advertises internal model names that public callers
#    cannot use. securitySchemes are always kept.
schemas = d.get("components", {}).get("schemas", {})

def refs_in(obj):
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref" and isinstance(v, str) and v.startswith("#/components/schemas/"):
                found.add(v.rsplit("/", 1)[1])
            else:
                found |= refs_in(v)
    elif isinstance(obj, list):
        for x in obj:
            found |= refs_in(x)
    return found

needed, frontier = set(), refs_in(new_paths)
while frontier:
    name = frontier.pop()
    if name in needed or name not in schemas:
        continue
    needed.add(name)
    frontier |= refs_in(schemas[name])

dropped = sorted(set(schemas) - needed)
d["components"]["schemas"] = {k: v for k, v in schemas.items() if k in needed}
print("schemas kept:", len(needed), "| dropped internal:", len(dropped))

# newline="\n": the artifact is committed LF-only; Windows text mode would otherwise
# rewrite every line as CRLF and turn a one-route change into a whole-file diff.
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(d, fh, ensure_ascii=False, indent=2)
print("public paths:", len(d["paths"]))
print("wrote:", OUT.relative_to(ROOT))
