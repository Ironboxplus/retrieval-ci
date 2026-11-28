#!/usr/bin/env bash
# PDF transport between Arc (Windows + Git Bash) and the kss SMB share.
#
# Modes:
#   pdf-up   --conf CONF --year YEAR        Arc local pdf  → SMB
#   md-down  --conf CONF --year YEAR        SMB markdown   → Arc local
#   ls-smb   --conf CONF --year YEAR        list remote dir for sanity check
#
# Uses robocopy over LAN SMB (W21 path; ~10 MB/s, 4-5× faster than ssh
# tunnel). MSYS2_ARG_CONV_EXCL=* prevents Git Bash from collapsing UNC
# paths like //10.100.100.9/lsb into C:\10.100.100.9\lsb.
set -euo pipefail

SMB_HOST="${SMB_HOST:-10.100.100.9}"
SMB_SHARE="${SMB_SHARE:-lsb}"
REMOTE_BASE="${REMOTE_BASE:-//${SMB_HOST}/${SMB_SHARE}/retrival/papers}"
LOCAL_BASE="${LOCAL_BASE:-data/papers/ccf_a_main_conference_corpus}"

mode="${1:-}"; shift || true
conf=""; year=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conf)  conf="$2"; shift 2 ;;
    --year)  year="$2"; shift 2 ;;
    -h|--help) mode="help"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${conf}" || -z "${year}" ]]; then
  echo "usage:" >&2
  echo "  $0 pdf-up   --conf <CONF> --year <YEAR>" >&2
  echo "  $0 md-down  --conf <CONF> --year <YEAR>" >&2
  echo "  $0 ls-smb   --conf <CONF> --year <YEAR>" >&2
  exit 2
fi

local_pdf="${LOCAL_BASE}/pdf/${conf}/${year}"
local_md="${LOCAL_BASE}/markdown/${conf}/${year}"
remote_pdf="${REMOTE_BASE}/pdf/${conf}/${year}"
remote_md="${REMOTE_BASE}/markdown/${conf}/${year}"

case "${mode}" in
  pdf-up)
    [[ -d "${local_pdf}" ]] || { echo "missing local: ${local_pdf}" >&2; exit 1; }
    echo "[ship] PDF ${local_pdf} -> ${remote_pdf}"
    MSYS2_ARG_CONV_EXCL='*' robocopy "${local_pdf}" "${remote_pdf}" "*.pdf" \
      /MT:32 /R:3 /W:5 /NDL /NJH /NJS /NP
    rc=$?
    # robocopy exit codes: 0 = no work, 1-7 success-like, 8+ = errors
    [[ $rc -lt 8 ]]
    ;;
  md-down)
    mkdir -p "${local_md}"
    echo "[ship] MD ${remote_md} -> ${local_md}"
    MSYS2_ARG_CONV_EXCL='*' robocopy "${remote_md}" "${local_md}" "*.md" \
      /MT:16 /R:3 /W:5 /S /NDL /NJH /NJS /NP
    rc=$?
    [[ $rc -lt 8 ]]
    ;;
  ls-smb)
    echo "[ls] ${remote_pdf}"
    MSYS2_ARG_CONV_EXCL='*' ls -la "${remote_pdf}" 2>/dev/null | head -20 || echo "  (empty or missing)"
    echo
    echo "[ls] ${remote_md}"
    MSYS2_ARG_CONV_EXCL='*' ls -la "${remote_md}" 2>/dev/null | head -20 || echo "  (empty or missing)"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
