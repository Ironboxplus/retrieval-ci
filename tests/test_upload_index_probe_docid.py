"""The upload index probe must count Qdrant points by the doc_id type that
index_document actually writes.

W: index_document stores the payload doc_id as a STRING (index_service.py:219
`str_doc_id = str(doc_id)`; payload `"doc_id": str_doc_id`). The probe
`_qdrant_point_count` queried `match.value = int(doc_id)` → the keyword(str)
index never matched → probe returned 0 → index_gate reported "0/N upserted" →
the index subtask failed even though all N points were in Qdrant (task #95
showed Index `-`). Fix: query the string form.
"""
import json
import urllib.request

import app.services.pipeline.upload_stages as us


class _FakeResp:
    def __init__(self, body: bytes):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_qdrant_point_count_matches_string_doc_id(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = req.data
        return _FakeResp(json.dumps({"result": {"count": 3783}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    n = us._qdrant_point_count(102394)

    assert n == 3783
    body = json.loads(captured["body"])
    match_val = body["filter"]["must"][0]["match"]["value"]
    # index_document writes doc_id as str → probe must match the str, not int
    assert match_val == "102394"
    assert not isinstance(match_val, int)
