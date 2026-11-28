# Gateway Deployment Entry

The canonical cross-machine deployment runbook is [../docs/DEPLOYMENT_CN.md](../docs/DEPLOYMENT_CN.md). BWH-specific files and safety notes are indexed in [../deploy/bwh/README.md](../deploy/bwh/README.md).

Gateway responsibilities:

- public OIDC/session/API-key authentication;
- tier authorization;
- credit ledger, metering, refund and idempotency leases;
- payment integration;
- caller-header scrubbing and internal-token proxying;
- structured access logs;
- persistent user/billing state in the BWH SQLite volume.

The 4090 is a compute tier and must not receive OAuth/payment secrets or balances.

For every gateway release verify database migration idempotency, old-image rollback, actual Traefik reachability, and the public permission matrix. A running Docker container is not sufficient evidence.
