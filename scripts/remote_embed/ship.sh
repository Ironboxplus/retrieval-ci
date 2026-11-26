#!/usr/bin/env bash
# Ship chunks up to kss and vectors back, via rsync over ssh.
#
# Usage (run from project root):
#   bash scripts/remote_embed/ship.sh code      # sync code repo (tar over ssh, one-shot)
#   bash scripts/remote_embed/ship.sh up        # chunks  LOCAL -> KSS
#   bash scripts/remote_embed/ship.sh down      # vectors KSS   -> LOCAL
#   bash scripts/remote_embed/ship.sh status    # tmux + disk on kss
#
# Env:
#   KSS_USER        kss
#   KSS_HOST        10.100.100.2
#   KSS_KEY         /c/Users/Arc/.ssh/kss_key
#   KSS_REMOTE_ROOT /mnt/lsb/retrival
#   KSS_CODE_ROOT   /home/kss/arc/retrival
set -euo pipefail

KSS_USER="${KSS_USER:-kss}"
KSS_HOST="${KSS_HOST:-10.100.100.2}"
KSS_KEY="${KSS_KEY:-/c/Users/Arc/.ssh/kss_key}"
KSS_REMOTE_ROOT="${KSS_REMOTE_ROOT:-/mnt/lsb/retrival}"
KSS_CODE_ROOT="${KSS_CODE_ROOT:-/home/kss/arc/retrival}"

LOCAL_CHUNKS="data/remote_embed/chunks"
LOCAL_VECTORS="data/remote_embed/vectors"
REMOTE_CHUNKS="${KSS_REMOTE_ROOT}/chunks"
REMOTE_VECTORS="${KSS_REMOTE_ROOT}/vectors"

# Ensure cwrsync ssh is found before MSYS ssh (cygwin path/pipe compatibility).
export PATH="$HOME/scoop/apps/cwrsync/current/bin:$HOME/scoop/shims:$PATH"
export MSYS_NO_PATHCONV=1

# Translate /c/... into cygwin form (for cwrsync's ssh) and Windows form (for OpenSSH).
# OpenSSH on Windows reads /c/foo as literal "/c/foo" and errors out — use C:/foo instead.
case "$KSS_KEY" in
    /c/*)        KSS_KEY_CYG="/cygdrive/c/${KSS_KEY#/c/}";  KSS_KEY_WIN="C:/${KSS_KEY#/c/}";;
    /[a-zA-Z]/*) KSS_KEY_CYG="/cygdrive/${KSS_KEY:1:1}/${KSS_KEY:3}"; KSS_KEY_WIN="${KSS_KEY:1:1}:/${KSS_KEY:3}";;
    *)           KSS_KEY_CYG="$KSS_KEY"; KSS_KEY_WIN="$KSS_KEY";;
esac

SSH_OPT="-i $KSS_KEY_CYG -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
REMOTE="$KSS_USER@$KSS_HOST"

case "${1:-}" in
    code)
        echo "[code] tar code -> $REMOTE:$KSS_CODE_ROOT"
        ssh -i "$KSS_KEY_WIN" "$REMOTE" "mkdir -p $KSS_CODE_ROOT"
        tar czf - \
            --exclude='.venv' --exclude='.git' --exclude='data' \
            --exclude='frontend/node_modules' --exclude='frontend/dist' --exclude='frontend/.vite' \
            --exclude='tools/qdrant/qdrant.exe' \
            --exclude='*.pyc' --exclude='__pycache__' --exclude='.pytest_cache' \
            --exclude='.mypy_cache' --exclude='.ropeproject' --exclude='*.log' --exclude='*.bak' \
            . | ssh -i "$KSS_KEY_WIN" "$REMOTE" "tar xzf - -C $KSS_CODE_ROOT && du -sh $KSS_CODE_ROOT"
        ;;
    up)
        # Optional incremental: MIN_SHARD=67 ship.sh up  uploads only chunks_0067+
        MIN_SHARD="${MIN_SHARD:-0}"
        files=()
        for f in "$LOCAL_CHUNKS"/chunks_*.jsonl.gz; do
            [ -f "$f" ] || continue
            n=$(basename "$f" | sed -n 's/chunks_\([0-9]\+\)\.jsonl\.gz/\1/p')
            n=$((10#$n))
            if [ "$n" -ge "$MIN_SHARD" ]; then
                files+=("$f")
            fi
        done
        if [ ${#files[@]} -eq 0 ]; then
            echo "[up] no shards >= chunks_$(printf '%04d' "$MIN_SHARD").jsonl.gz under $LOCAL_CHUNKS"
            exit 0
        fi
        echo "[up] uploading ${#files[@]} shards (>= chunks_$(printf '%04d' "$MIN_SHARD")) -> $REMOTE:$REMOTE_CHUNKS"
        ssh -i "$KSS_KEY_WIN" "$REMOTE" "mkdir -p $REMOTE_CHUNKS"
        # /mnt/lsb is SMB — rsync mkstemp fails. Use scp (direct write).
        scp -i "$KSS_KEY_WIN" "${files[@]}" "$REMOTE:$REMOTE_CHUNKS/"
        ;;
    down)
        echo "[down] $REMOTE:$REMOTE_VECTORS -> $LOCAL_VECTORS"
        mkdir -p "$LOCAL_VECTORS"
        # cwrsync's bundled cygwin tools need /cygdrive/X/... not /c/... or msys paths.
        ABS_LOCAL="$(cd "$LOCAL_VECTORS" && pwd)"
        case "$ABS_LOCAL" in
            /c/*)            CYG_DEST="/cygdrive/c/${ABS_LOCAL#/c/}";;
            /[a-zA-Z]/*)     CYG_DEST="/cygdrive/${ABS_LOCAL:1:1}/${ABS_LOCAL:3}";;
            /cygdrive/*)     CYG_DEST="$ABS_LOCAL";;
            *)               CYG_DEST="$ABS_LOCAL";;
        esac
        echo "    cygwin dest = $CYG_DEST"
        rsync -avz --partial --info=progress2 -e "ssh $SSH_OPT" \
            "$REMOTE:$REMOTE_VECTORS/" \
            "$CYG_DEST/"
        ;;
    status)
        ssh -i "$KSS_KEY_WIN" "$REMOTE" "echo ---tmux---; tmux ls 2>&1; echo ---chunks---; du -sh $REMOTE_CHUNKS 2>/dev/null; ls $REMOTE_CHUNKS 2>/dev/null | wc -l; ls $REMOTE_CHUNKS 2>/dev/null | tail -10; echo ---vectors---; du -sh $REMOTE_VECTORS 2>/dev/null; ls $REMOTE_VECTORS 2>/dev/null | head -10; echo ---model---; du -sh /mnt/nvme-disk/.cache/modelscope/BAAI/bge-m3 2>/dev/null"
        ;;
    *)
        echo "usage: $0 {code|up|down|status}"
        echo "  up:  optional MIN_SHARD=N env to upload only chunks_NNNN+ (incremental)"
        exit 2
        ;;
esac
