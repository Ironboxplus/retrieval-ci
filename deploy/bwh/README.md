# BWH Edge Deployment

> Canonical release procedure: [../../docs/DEPLOYMENT_CN.md](../../docs/DEPLOYMENT_CN.md). Runtime topology: [../../overview.md](../../overview.md).

BWH owns the public edge only:

```text
Traefik :443
  -> retrieval-frontend :80
  -> retrieval-gateway :8001
       -> retrieval-xray :18000
            -> 4090 compute public door
```

Sensitive users, identities, API-key hashes, balances, transactions, payment configuration and idempotency state live in the gateway SQLite volume. They never move to the 4090 compute database.

## Tracked Files

| File | Purpose |
|---|---|
| `retrieval.yml` | full expected Traefik routers/services for Retrieval |
| `traefik-retrieval.patch.yml` | historical/merge-oriented patch; do not apply blindly |
| `compose.gateway.snippet.yml` | gateway service fragment for the production compose |
| `gateway.env.example` | environment variable names only, no secrets |
| `deploy-gateway.sh` | bounded pull/recreate helper |
| `../../gateway/Dockerfile` | gateway image |

## Safety Rules

1. Read and back up the production dynamic file before changing it.
2. Preserve all unrelated 443 routers, TLS resolver, entrypoints and SNI behavior.
3. Compare gateway handlers with Traefik path rules. The gateway currently has documents/journal/index/code/admin handlers that may not all be represented by `retrieval.yml`.
4. Build/push and pull/recreate are separate states; record the running image digest.
5. Preserve the gateway database volume during image replacement.
6. Verify anonymous, paying and member/internal behavior with real public requests.

## Image

Primary BWH pull target:

```text
docker.io/arccot/retrival-gateway:latest
```

The repository also contains CI for GHCR/Docker Hub publication. Production still requires an explicit or automated pull and container recreation.
