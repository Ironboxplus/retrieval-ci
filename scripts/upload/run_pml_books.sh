#!/bin/bash
# Extract both ENGLISH PML volumes to Markdown, strictly serial (one 12GB GPU).
#
# Language matters here. The volume already indexed in the library
# (data/uploads/c7b15aac-*.pdf, 274MB, 1,308 pages, doc_id 102394) is a CHINESE
# TRANSLATION -- sampled at six points across the book it carries CJK on every
# page (3,548 CJK vs 12,277 latin chars). It is deliberately NOT processed here.
# Both files below sampled 0 CJK across the whole book:
#     PML_Introduction.pdf         860 pages,  92,323,940 B
#     PML_Advanced_Topics_EN.pdf  1,369 pages, 151,629,852 B
#
# Each book is driven by marker_extract_book.py, which spawns ONE SUBPROCESS PER
# PAGE BATCH. That is not an optimisation: MarkerExtractor caches its PdfConverter
# at class scope, so batching inside a single process silently reuses the first
# batch's page_range and produces N byte-identical copies of the first pages. An
# overnight run did exactly that. The script's post-merge `distinct` check fails
# the run rather than writing such a file.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID TQDM_DISABLE=1

echo "=== BEGIN $(date -Is) ==="

# Introduction first: 860 pages at ~3.2 s/page validates the per-batch fix in
# about an hour, before committing ~3h of GPU to the 1,369-page volume.
$PY scripts/upload/marker_extract_book.py --batch 100 \
  --pdf data/upload_staging/PML_Introduction.pdf \
  --out data/upload_staging/PML_Introduction.md
INTRO_RC=$?
echo "=== INTRO rc=$INTRO_RC $(date -Is) ==="

if [ "$INTRO_RC" -ne 0 ]; then
  echo "=== ABORT: introduction failed, not starting the advanced volume ==="
  exit "$INTRO_RC"
fi

$PY scripts/upload/marker_extract_book.py --batch 100 \
  --pdf data/upload_staging/PML_Advanced_Topics_EN.pdf \
  --out data/upload_staging/PML_Advanced_Topics_EN.md
ADV_RC=$?
echo "=== ADVANCED rc=$ADV_RC $(date -Is) ==="

ls -la data/upload_staging/*.md 2>/dev/null
echo "=== ALL DONE $(date -Is) ==="
exit "$ADV_RC"
