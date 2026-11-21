#!/usr/bin/env bash
# Manual pull+recreate convenience for the gateway on BWH (run it ON the box).
#   ssh bwh 'bash ~/docker/retrieval/deploy-gateway.sh'
# CI does NOT touch BWH — it only builds+pushes the GHCR image. Deploy is pull-based:
# watchtower auto-updates the container when the image changes, or run this for instant
# pull+recreate. No CI ssh key needed.
set -euo pipefail

COMPOSE=/root/docker/retrieval/compose.yml

# pull the freshly-pushed image and recreate just the gateway container
docker compose -f "$COMPOSE" pull gateway
docker compose -f "$COMPOSE" up -d gateway

# prune the now-dangling old image to spare BWH's tiny disk (~4.6G free)
docker image prune -f >/dev/null 2>&1 || true
echo "gateway redeploy ok: $(docker inspect -f '{{.Config.Image}} {{.State.Status}}' retrieval-gateway 2>/dev/null)"
