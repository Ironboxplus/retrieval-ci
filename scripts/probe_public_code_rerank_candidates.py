"""Measure full-pool Public Code rerank orchestration without loading a model.

This is deliberately a local, deterministic probe: it exercises the service's
retrieval cap, exact deduplication, candidate telemetry, and reranker call shape with
synthetic code chunks. It does not initialise the shared cross-encoder, touch Qdrant,
or make a deployment claim. Use a production latency benchmark separately when a
model-loaded environment is explicitly in scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from app.services.public_code.repository import PublicCodeSearchScope
from app.services.public_code.search_service import (
    PublicCodeSearchRequest,
    PublicCodeSearchService,
)


_PROBE_SOURCE_ID = "public-code-rerank-probe"
_PROBE_REVISION = "synthetic-revision"


class _ProbeRepository:
    def resolve_search_scope(
        self,
        *,
        source_ids: Sequence[str],
        domains: Sequence[str],
    ) -> PublicCodeSearchScope:
        del domains
        selected = tuple(source_ids) or (_PROBE_SOURCE_ID,)
        return PublicCodeSearchScope(
            source_ids=selected,
            active_revisions={source_id: _PROBE_REVISION for source_id in selected},
        )


class _ProbeVectorStore:
    supports_sparse = True

    def __init__(self, hits: list[dict[str, object]]):
        self._hits = hits

    def search(self, *, top_k: int, **_kwargs) -> list[dict[str, object]]:
        return self._hits[:top_k]


class _CountingReranker:
    def __init__(self):
        self.contents: list[str] = []

    def score_pairs(self, _query: str, contents: Sequence[str]) -> list[float]:
        self.contents = list(contents)
        return [float(index) for index in range(len(self.contents))]


def _synthetic_hit(index: int, *, content_characters: int) -> dict[str, object]:
    code = (f"probe_candidate_{index} = 'rerank-full-pool'\n" * (content_characters // 32 + 1))[
        :content_characters
    ]
    path = f"src/rerank_probe/candidate_{index}.py"
    return {
        "score": 1.0 - index / 1000,
        "source_id": _PROBE_SOURCE_ID,
        "revision": _PROBE_REVISION,
        "domain": "probe",
        "path": path,
        "language": "python",
        "start_line": 1,
        "end_line": 1,
        "label": f"candidate_{index}",
        "content": f"# path: {path}\n{code}",
        "code": code,
    }


def run_probe(
    *,
    candidate_count: int = 20,
    content_characters: int = 2_048,
) -> dict[str, int | float | str]:
    """Run a bounded, model-free rehearsal of the full rerank candidate pool."""

    if not 1 <= candidate_count <= 50:
        raise ValueError("candidate_count must be between 1 and the 50-candidate retrieval cap")
    if content_characters < 1:
        raise ValueError("content_characters must be positive")

    # Four times top_k yields the requested pool after the service's minimum 20 and
    # maximum 50 retrieval bounds. The default is the real top_k=5 -> fetch=20 case.
    top_k = max(5, (candidate_count + 3) // 4)
    reranker = _CountingReranker()
    service = PublicCodeSearchService(
        repository=_ProbeRepository(),
        vector_store=_ProbeVectorStore([
            _synthetic_hit(index, content_characters=content_characters)
            for index in range(candidate_count)
        ]),
        embed_fn=lambda _texts: [[1.0]],
        reranker=reranker,
    )

    started = perf_counter()
    response = service.search(PublicCodeSearchRequest(
        query="full pool rerank probe",
        source_ids=(_PROBE_SOURCE_ID,),
        top_k=top_k,
        mode="dense",
        rerank=True,
    ))
    elapsed_ms = (perf_counter() - started) * 1_000
    return {
        "measurement": "service_orchestration_with_deterministic_no_model_reranker",
        "requested_top_k": top_k,
        "retrieval_candidate_count": response.candidate_count,
        "rerank_candidate_count": response.rerank_candidate_count,
        "scored_candidate_count": len(reranker.contents),
        "returned_hit_count": len(response.hits),
        "scored_content_characters": sum(len(content) for content in reranker.contents),
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--content-chars", type=int, default=2_048)
    args = parser.parse_args()
    print(json.dumps(
        run_probe(candidate_count=args.candidates, content_characters=args.content_chars),
        ensure_ascii=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
