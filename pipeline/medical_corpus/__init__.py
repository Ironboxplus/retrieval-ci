"""Medical literature corpus preparation (EuropePMC-first, proxy-mandatory).

Default workflows are dry-run / catalog-only. Real PDF downloads require an
explicit ``--execute-download`` flag and always use the local proxy (10808).

Cloudflare edges are handled via vendored TurnstileBypass (minimal headless
Chrome) — see :mod:`pipeline.medical_corpus.cf_bypass`.
"""

from pipeline.medical_corpus.cf_bypass import (
    CfBypassResult,
    solve_cloudflare_for_url,
    solve_cloudflare_for_url_sync,
    turnstile_available,
)
from pipeline.medical_corpus.models import MedicalPaperRecord, MedicalSourceKind
from pipeline.medical_corpus.shield import ShieldDecision, classify_http_shield

__all__ = [
    "CfBypassResult",
    "MedicalPaperRecord",
    "MedicalSourceKind",
    "ShieldDecision",
    "classify_http_shield",
    "solve_cloudflare_for_url",
    "solve_cloudflare_for_url_sync",
    "turnstile_available",
]
