#!/bin/bash
# Installs the vLLM environment of the model services (arc-vllm-embed.service, arc-vllm-rerank.service) on .88.
#
#   bash deploy/compute88/install_vllm_venv.sh --proxy http://HOST:PORT        install (or repair) the venv
#   bash deploy/compute88/install_vllm_venv.sh --check                         verify only; no network, no proxy
#
# Options (all optional except the proxy of an install):
#   --proxy URL          http(s) proxy for every download, or env VLLM_VENV_PROXY. Required: there is no direct
#                        connection, ever. env.md only records the Arc Windows proxy (http://127.0.0.1:10808), which is
#                        that machine's loopback; on .88 it exists only through a tunnel (ssh -R 10808:127.0.0.1:10808)
#                        or the proxy of whoever runs this. It is probed before anything is created: if it cannot
#                        reach PyPI and the wheel host, the run stops.
#   --venv DIR           default /home/arc/.venvs/retrival-vllm (the path both units name)
#   --requirements FILE  default deploy/compute88/vllm_venv_requirements.txt: every package of the reference venv
#                        (/root/vllm-bench/venv: vllm 0.33.0, torch 2.14.0 on CUDA 13.0, transformers 5.20.2) pinned
#                        with ==, installed with --no-deps so nothing resolves differently here
#   --skip-driver-check  do not require an NVIDIA driver >= 580 (the CUDA 13.0 wheels need it)
#
# Installs: the pinned set, then bge-m3-sparse-plugin from scripts/remote_embed/bge_m3_sparse_plugin (a copy is
# built, the repository is not written to). Needs python3.12 with venv and pip (uv is used instead when present).
# Re-running is safe: a venv that already matches every pin and has the plugin is left alone ("already installed");
# otherwise pip converges it in place (satisfied pins are skipped, nothing is deleted).
# Never touches /home/arc/.venvs/retrival-pvl-server (PaddleOCR-VL's server): any --venv that is, contains or lies
# inside it is refused before any other check.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
VENV=/home/arc/.venvs/retrival-vllm
REQ=$HERE/vllm_venv_requirements.txt
PLUGIN=$REPO/scripts/remote_embed/bge_m3_sparse_plugin
PROTECTED=/home/arc/.venvs/retrival-pvl-server
PROXY=${VLLM_VENV_PROXY:-}
CHECK=0
SKIP_DRIVER=0

die() { echo "install_vllm_venv: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --proxy) PROXY=${2:?--proxy needs a URL}; shift 2 ;;
    --venv) VENV=${2:?--venv needs a directory}; shift 2 ;;
    --requirements) REQ=${2:?--requirements needs a file}; shift 2 ;;
    --check) CHECK=1; shift ;;
    --skip-driver-check) SKIP_DRIVER=1; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# 1. The Paddle server's venv is never this script's business: first check, whatever else is missing.
resolved=$(realpath -m -- "$VENV")
case "$resolved/" in "$PROTECTED/"*) die "refusing --venv $VENV: it is inside $PROTECTED (PaddleOCR-VL's server environment)" ;; esac
case "$PROTECTED/" in "$resolved/"*) die "refusing --venv $VENV: it contains $PROTECTED (PaddleOCR-VL's server environment)" ;; esac
VENV=$resolved
[ -f "$REQ" ] || die "requirements file missing: $REQ"

# 2. Verification: the pins, the plugin's entry point, torch's fp16-accumulation switch. No network.
verify() {
  [ -x "$VENV/bin/python" ] || { echo "install_vllm_venv: venv python missing: $VENV/bin/python" >&2; return 1; }
  "$VENV/bin/python" - "$REQ" <<'PY'
import sys
from importlib import metadata

pins = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#"):
        name, version = line.split("==")
        pins[name] = version
bad = []
for name, wanted in pins.items():
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        bad.append(f"{name}: not installed, the lock has {wanted}")
        continue
    if have != wanted:
        bad.append(f"{name}: installed {have}, the lock has {wanted}")
plugins = {e.name for e in metadata.entry_points(group="vllm.io_processor_plugins")}
if "bge_m3_sparse_plugin" not in plugins:
    bad.append(f"plugin entry point bge_m3_sparse_plugin is not registered (found {sorted(plugins)})")
for line in bad:
    print("MISMATCH " + line, file=sys.stderr)
if bad:
    sys.exit(1)
import torch

print(f"vllm {metadata.version('vllm')}, torch {torch.__version__}, plugin bge_m3_sparse_plugin registered")
print(f"fp16 accumulation switch present: {hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation')}; "
      f"cuda available: {torch.cuda.is_available()}")
print(f"all pins match ({len(pins)})")
PY
}

if [ "$CHECK" = 1 ]; then
  verify || die "the environment at $VENV does not match $REQ"
  exit 0
fi

# 3. Proxy: required, http(s), and it has to reach PyPI and the wheel host. Nothing is created before this passes.
[ -n "$PROXY" ] || die "a proxy is required: pass --proxy http://HOST:PORT or set VLLM_VENV_PROXY (downloads never go direct)"
[[ "$PROXY" =~ ^https?://[^/]+/?$ ]] || die "the proxy must be an http(s) URL like http://HOST:PORT, not '$PROXY'"
status=$(curl -sS --proxy "$PROXY" --noproxy '' -m 30 -o /dev/null -w '%{http_code}' https://pypi.org/simple/pip/ 2>&1) \
  || die "proxy $PROXY is not usable for https://pypi.org: $status"
[ "$status" = 200 ] || die "proxy $PROXY is not usable: https://pypi.org/simple/pip/ answered $status"
status=$(curl -sS --proxy "$PROXY" --noproxy '' -m 30 -o /dev/null -w '%{http_code}' https://files.pythonhosted.org/ 2>&1) \
  || die "proxy $PROXY is not usable for https://files.pythonhosted.org (the wheel host): $status"
echo "proxy $PROXY reaches PyPI and the wheel host"

# 4. Host prerequisites.
if [ "$SKIP_DRIVER" = 0 ]; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 | tr -d ' ') \
    || die "nvidia-smi failed: no NVIDIA driver (use --skip-driver-check to install without a GPU)"
  [ -n "${driver:-}" ] && [ "$driver" -ge 580 ] \
    || die "NVIDIA driver ${driver:-unknown} is older than 580: the CUDA 13.0 wheels of the lock need 580 or newer"
  echo "NVIDIA driver $driver"
fi

exec 9>/tmp/install_vllm_venv.lock
flock -n 9 || die "another install_vllm_venv.sh is running"

if verify >/dev/null 2>&1; then
  echo "already installed: $VENV matches $REQ"
  verify
  exit 0
fi

# 5. Install, every download through the proxy.
export HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY ALL_PROXY=$PROXY http_proxy=$PROXY https_proxy=$PROXY all_proxy=$PROXY
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$(dirname "$VENV")"
  if command -v uv >/dev/null 2>&1; then uv venv --python 3.12 "$VENV"
  else
    command -v python3.12 >/dev/null 2>&1 || die "python3.12 not found: the pins are cp312 wheels"
    python3.12 -m venv "$VENV" || { rm -rf "$VENV"; die "python3.12 -m venv failed (python3.12-venv missing?); $VENV removed"; }
  fi
fi
"$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || die "$VENV is not a Python 3.12 environment: the pins are cp312 wheels; it is left as it is"

if command -v uv >/dev/null 2>&1; then
  INSTALL=(uv pip install --python "$VENV/bin/python")
else
  INSTALL=("$VENV/bin/python" -m pip install --proxy "$PROXY")
fi
echo "installing $(grep -vc '^#' "$REQ") pinned packages into $VENV (several GB; satisfied pins are skipped)"
"${INSTALL[@]}" --no-deps -r "$REQ"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$PLUGIN/." "$work/"
"${INSTALL[@]}" --no-deps --no-build-isolation --force-reinstall "$work"

verify || die "the install finished but $VENV still does not match $REQ"
echo "done: $VENV"
