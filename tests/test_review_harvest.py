import json

import pytest
import requests

from scripts.reviews import harvest_openreview as harvest


def response(status, body, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r.headers.update(headers or {})
    return r


def test_browser_challenge_stops_without_retry(monkeypatch):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    calls = []
    monkeypatch.setattr(h.session, "get", lambda *a, **k: calls.append(k) or response(
        403, {"name": "ChallengeRequiredError"}))
    with pytest.raises(harvest.AccessChallengeRequired):
        h._get({"limit": 1})
    assert len(calls) == 1
    assert h.session.trust_env is False


def test_unknown_decision_is_not_accepted():
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    rec = h.parse_submission({"id": "x", "content": {}}, "ICLR.cc/2026/Conference",
                             2026, "ICLR", replies=[])
    assert rec["accepted"] is False


def test_cursor_pagination_continues_after_thousand(monkeypatch):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    calls = []
    def get(params):
        assert "offset" not in params
        calls.append(params)
        start = int(params.get("after", "0"))
        return {"notes": [{"id": str(i), "content": {}, "details": {"replies": []}}
                          for i in range(start + 1, min(start + 51, 1102))]}
    monkeypatch.setattr(h, "_get", get)
    rows = list(h.iter_harvest("ICML.cc/2025/Conference", 2025, "ICML"))
    assert len(rows) == 1101
    assert calls[1]["after"] == "50"
    assert all("offset" not in p for p in calls)
    assert all(p["sort"] == "id" for p in calls)
    assert len({r[1]["openreview_id"] for r in rows}) == 1101


def test_retry_after_is_respected(monkeypatch):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    replies = iter([response(429, {}, {"Retry-After": "120"}), response(200, {"notes": []})])
    sleeps = []
    monkeypatch.setattr(h.session, "get", lambda *a, **k: next(replies))
    monkeypatch.setattr(harvest.time, "sleep", sleeps.append)
    assert h._get({}) == {"notes": []}
    assert sleeps[0] >= 120


def test_permanent_api_error_is_not_rate_limit(monkeypatch):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    monkeypatch.setattr(harvest.time, "sleep", lambda _: None)
    monkeypatch.setattr(h.session, "get", lambda *a, **k: response(400, {}))
    with pytest.raises(requests.HTTPError):
        h._get({})


def test_request_budget_includes_retries(monkeypatch):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0,
                                   max_requests=1)
    calls = []
    monkeypatch.setattr(h.session, "get", lambda *a, **k: calls.append(k) or response(200, {}))
    h._get({})
    with pytest.raises(harvest.RequestBudgetReached):
        h._get({})
    assert len(calls) == 1


def test_retries_cannot_exceed_request_budget(monkeypatch):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0,
                                   max_requests=1)
    calls = []
    monkeypatch.setattr(harvest.time, "sleep", lambda _: None)
    monkeypatch.setattr(h.session, "get", lambda *a, **k: calls.append(k) or response(429, {}))
    with pytest.raises(harvest.RequestBudgetReached):
        h._get({})
    assert len(calls) == 1


@pytest.mark.parametrize("content,accepted", [
    ({"venueid": {"value": "ICLR.cc/2026/Conference"}}, True),
    ({"venue": {"value": "ICLR 2026 Poster"}}, True),
    ({"venue": {"value": "ICLR 2026 Withdrawn Submission"}}, False),
    ({"venue": {"value": "Submitted to ICLR 2026"}}, False),
])
def test_explicit_acceptance_and_negative_statuses(content, accepted):
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    assert h.parse_submission({"id": "x", "content": content},
        "ICLR.cc/2026/Conference", 2026, "ICLR", replies=[])["accepted"] is accepted


def test_private_reply_is_not_exported():
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    rec = h.parse_submission({"id": "x", "content": {}}, "ICLR.cc/2026/Conference",
        2026, "ICLR", replies=[{"id": "private", "readers": ["Reviewers"],
        "invitations": ["ICLR.cc/2026/Conference/Submission1/-/Official_Review"],
        "content": {"weaknesses": {"value": "private text"}}}])
    assert rec["n_reviews"] == 0


def test_resume_uses_only_committed_cursor(tmp_path):
    from scripts.reviews.harvest_all import harvest_venue
    out = tmp_path / "ICML_2025.jsonl"
    rec = {"openreview_id": "a", "n_reviews": 1, "accepted": True}
    class Interrupted:
        def iter_harvest(self, *args, **kwargs):
            yield 1, rec
            raise harvest.AccessChallengeRequired("403")
    with pytest.raises(harvest.AccessChallengeRequired):
        harvest_venue(Interrupted(), "ICML.cc/2025/Conference", 2025, "ICML", out)
    assert not out.with_suffix(".done").exists()
    class Resumed:
        def iter_harvest(self, *args, **kwargs):
            assert kwargs == {"start_offset": 1, "after": "a"}
            yield 2, {**rec, "openreview_id": "b"}
    harvest_venue(Resumed(), "ICML.cc/2025/Conference", 2025, "ICML", out)
    assert [json.loads(l)["openreview_id"] for l in out.open()] == ["a", "b"]
    assert out.with_suffix(".done").exists()


def test_legacy_output_is_never_overwritten(tmp_path):
    from scripts.reviews.harvest_all import harvest_venue
    out = tmp_path / "old.jsonl"
    out.write_text('{"openreview_id":"a"}\n')
    with pytest.raises(RuntimeError, match="legacy"):
        harvest_venue(None, "ICML.cc/2025/Conference", 2025, "ICML", out)
    assert out.read_text() == '{"openreview_id":"a"}\n'


def test_merged_critique_and_provenance_are_preserved():
    h = harvest.OpenReviewHarvester(proxy="http://localhost:10808", pace=0)
    review = {"id": "rv", "readers": ["everyone"], "license": "CC BY 4.0",
              "invitations": ["NeurIPS.cc/2025/Conference/Submission1/-/Official_Review"],
              "content": {"strengths_and_weaknesses": {"value": "Good idea; weak evidence."},
                          "new_form_field": {"value": "Keep this too."}}}
    rec = h.parse_submission({"id": "x", "content": {}}, "NeurIPS.cc/2025/Conference",
                             2025, "NeurIPS", replies=[review])
    assert rec["reviews"][0]["weaknesses"] == "Good idea; weak evidence."
    assert rec["reviews"][0]["license"] == "CC BY 4.0"
    assert rec["reviews"][0]["content"] == review["content"]


def test_uncommitted_tail_is_replayed_once(tmp_path):
    from scripts.reviews.harvest_all import harvest_venue
    out = tmp_path / "ICML_2025.jsonl"
    class Interrupted:
        def iter_harvest(self, *args, **kwargs):
            yield 1, {"openreview_id": "a"}
            raise harvest.RequestBudgetReached()
    with pytest.raises(harvest.RequestBudgetReached):
        harvest_venue(Interrupted(), "v", 2025, "ICML", out)
    with out.open("ab") as f:
        f.write(b'{"openreview_id":"broken')
    class Resumed:
        def iter_harvest(self, *args, **kwargs):
            yield 2, {"openreview_id": "b"}
    harvest_venue(Resumed(), "v", 2025, "ICML", out)
    assert [json.loads(l)["openreview_id"] for l in out.open()] == ["a", "b"]
