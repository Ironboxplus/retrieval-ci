"""/api/schedules end to end: real router over httpx.ASGITransport, real SQLite file,
real AsyncIOScheduler behind it. Admin gating uses the production identity rules
(internal door + private LAN peer, or gateway-injected admin tier)."""
from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from app.api.routes import schedules as schedules_routes
from app.services.pipeline.store import TaskStore
from app.services.scheduler.kinds import FireResult, JobKind, Metric, RetryPolicy, ScheduleSeed
from app.services.scheduler.service import SchedulerService
from tests.sqlite_task_db import make_session_factory

INTERNAL = "http://10.100.100.1:8000"   # internal door (INTERNAL_API_PORT)
PUBLIC = "http://127.0.0.1:8001"        # public tunnel door (PUBLIC_API_PORT)
LAN_PEER = ("10.100.100.9", 51000)


class Params(BaseModel):
    limit: int = 0


def make_kind(store: TaskStore) -> JobKind:
    def fire(params, origin):
        task_id = store.create_task(task_type="unit_api_task", params={"origin": origin, "options": params.model_dump()})
        return FireResult("enqueued", task_id=task_id, reason=f"limit={params.limit}")

    return JobKind(
        name="unit_kind", title="Unit kind", params_model=Params, fire=fire,
        retry=RetryPolicy(max_retries=2, base_delay_s=60),
        status=lambda: [Metric("watermark", "2026-11-02", "date"), Metric("pending", 12, "papers")],
        defaults=(ScheduleSeed(key="builtin", title="Built-in", trigger={"type": "interval", "seconds": 3600},
                               catchup="skip"),),
    )


@pytest_asyncio.fixture
async def env(tmp_path):
    db = make_session_factory(tmp_path / "api.db")
    store = TaskStore(db)
    kind = make_kind(store)
    service = SchedulerService(session_factory=db, store=store, kinds=lambda: {kind.name: kind},
                               lock_path=str(tmp_path / "scheduler.lock"))
    assert service.start()
    app = FastAPI()
    app.include_router(schedules_routes.router, prefix="/api/schedules")
    app.dependency_overrides[schedules_routes.get_service] = lambda: service
    yield app, service, store
    service.shutdown()


def admin(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=LAN_PEER), base_url=INTERNAL)


DAILY = {"type": "daily", "time": "03:00", "timezone": "Asia/Shanghai"}


@pytest.mark.asyncio
async def test_kinds_expose_the_params_schema_defaults_and_retry_policy(env):
    app, _, _ = env
    async with admin(app) as http:
        body = (await http.get("/api/schedules/kinds")).json()
    [kind] = body
    assert kind["name"] == "unit_kind"
    assert kind["params_schema"]["properties"]["limit"]["type"] == "integer"
    assert kind["defaults"][0]["key"] == "builtin"
    assert kind["retry"] == {"max_retries": 2, "base_delay_s": 60, "backoff": 2.0, "max_delay_s": 21600}


@pytest.mark.asyncio
async def test_create_list_patch_delete_round_trip(env):
    app, _, _ = env
    async with admin(app) as http:
        created = await http.post("/api/schedules", json={
            "key": "nightly", "kind": "unit_kind", "title": "Nightly", "trigger": DAILY, "params": {"limit": 7}})
        assert created.status_code == 201
        row = created.json()
        assert row["system"] is False and row["params_json"] == {"limit": 7}
        assert len(row["next_fire_times"]) == 5 and row["next_run_time"] is not None
        assert row["next_run_time"].endswith("+00:00") or "+" in row["next_run_time"]

        listing = (await http.get("/api/schedules")).json()
        assert [r["key"] for r in listing] == ["builtin", "nightly"]
        builtin = listing[0]
        assert builtin["system"] is True
        assert builtin["metrics"] == [
            {"name": "watermark", "value": "2026-11-02", "unit": "date"},
            {"name": "pending", "value": 12, "unit": "papers"},
        ]
        assert builtin["last_tick"] is None and builtin["running_task"] is None

        paused = (await http.patch(f"/api/schedules/{row['id']}", json={"enabled": False})).json()
        assert paused["enabled"] is False and paused["next_run_time"] is None and paused["next_fire_times"] == []

        edited = (await http.patch(f"/api/schedules/{row['id']}", json={
            "enabled": True, "title": "Late night", "params": {"limit": 9},
            "trigger": {"type": "weekly", "days": ["mon", "fri"], "time": "04:15", "timezone": "Asia/Shanghai"},
            "misfire_grace_s": 120,
        })).json()
        assert (edited["title"], edited["params_json"], edited["misfire_grace_s"]) == ("Late night", {"limit": 9}, 120)
        assert {int(t[:10].replace("-", "")) for t in edited["next_fire_times"]}  # parsed dates
        assert edited["next_run_time"] is not None

        # Explicit null resets the grace override to "kind default".
        reset = (await http.patch(f"/api/schedules/{row['id']}", json={"misfire_grace_s": None})).json()
        assert reset["misfire_grace_s"] is None

        gone = await http.delete(f"/api/schedules/{row['id']}")
        assert gone.status_code == 200 and gone.json() == {"id": row["id"], "deleted": True}
        assert [r["key"] for r in (await http.get("/api/schedules")).json()] == ["builtin"]


@pytest.mark.asyncio
async def test_validation_and_not_found_statuses(env):
    app, _, _ = env
    async with admin(app) as http:
        ok = {"key": "k", "kind": "unit_kind", "title": "t", "trigger": DAILY}
        assert (await http.post("/api/schedules", json=ok)).status_code == 201
        assert (await http.post("/api/schedules", json=ok)).status_code == 409
        assert (await http.post("/api/schedules", json={**ok, "key": "k2", "kind": "nope"})).status_code == 422
        bad_trigger = await http.post("/api/schedules", json={
            **ok, "key": "k3", "trigger": {"type": "daily", "time": "99:99"}})
        assert bad_trigger.status_code == 422 and "HH:MM" in bad_trigger.text
        assert (await http.post("/api/schedules", json={
            **ok, "key": "k4", "params": {"limit": "lots"}})).status_code == 422
        assert (await http.post("/api/schedules", json={**ok, "key": "k5", "catchup": "always"})).status_code == 422
        assert (await http.patch("/api/schedules/9999", json={"enabled": False})).status_code == 404
        assert (await http.delete("/api/schedules/9999")).status_code == 404
        assert (await http.post("/api/schedules/9999/run")).status_code == 404
        assert (await http.get("/api/schedules/9999/ticks")).status_code == 404
        # The built-in row is protected.
        builtin = next(r for r in (await http.get("/api/schedules")).json() if r["system"])
        deleted = await http.delete(f"/api/schedules/{builtin['id']}")
        assert deleted.status_code == 400 and "built-in" in deleted.text


@pytest.mark.asyncio
async def test_run_now_records_a_manual_tick_with_its_task_and_lists_it(env):
    app, _, store = env
    async with admin(app) as http:
        row = (await http.post("/api/schedules", json={
            "key": "run", "kind": "unit_kind", "title": "Run", "trigger": DAILY, "params": {"limit": 3}})).json()

        fired = await http.post(f"/api/schedules/{row['id']}/run")
        assert fired.status_code == 200
        body = fired.json()
        assert body["status"] == "enqueued" and body["reason"] == "limit=3" and body["task_id"]
        task = store.get_task(body["task_id"])
        assert task["params"]["origin"]["trigger"] == "manual"
        assert task["params"]["options"] == {"limit": 3}

        listing = {r["key"]: r for r in (await http.get("/api/schedules")).json()}["run"]
        assert listing["running_task"]["id"] == body["task_id"]            # pending task of this schedule
        assert listing["last_tick"]["trigger"] == "manual"

        store.set_task_status(body["task_id"], "running")
        store.update_result(body["task_id"], {"new": 4, "withdrawn": 1})
        store.set_task_status(body["task_id"], "completed")

        ticks = (await http.get(f"/api/schedules/{row['id']}/ticks")).json()
        [tick] = ticks
        assert (tick["trigger"], tick["status"], tick["task_id"]) == ("manual", "enqueued", body["task_id"])
        assert tick["task"]["status"] == "completed"
        assert tick["task"]["result_json"] == {"new": 4, "withdrawn": 1}
        assert tick["task"]["started_at"] and tick["task"]["completed_at"]
        assert (await http.get(f"/api/schedules/{row['id']}")).status_code == 405  # no per-id GET


@pytest.mark.asyncio
async def test_ticks_page_newest_first_with_before_id(env):
    app, service, _ = env
    from datetime import datetime, timedelta, timezone

    async with admin(app) as http:
        row = (await http.post("/api/schedules", json={
            "key": "paged", "kind": "unit_kind", "title": "P", "trigger": DAILY})).json()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for n in range(5):
            service._write_tick(row["id"], base + timedelta(days=n), "cron", "skipped", f"n{n}", None, 0)

        first = (await http.get(f"/api/schedules/{row['id']}/ticks?limit=2")).json()
        assert [t["reason"] for t in first] == ["n4", "n3"]
        second = (await http.get(f"/api/schedules/{row['id']}/ticks?limit=2&before_id={first[-1]['id']}")).json()
        assert [t["reason"] for t in second] == ["n2", "n1"]
        assert (await http.get(f"/api/schedules/{row['id']}/ticks?limit=0")).status_code == 422


@pytest.mark.asyncio
async def test_preview_returns_the_next_times_with_the_trigger_offset(env):
    app, _, _ = env
    async with admin(app) as http:
        ok = (await http.post("/api/schedules/preview", json={"trigger": DAILY, "count": 5})).json()
        assert len(ok["next_fire_times"]) == 5
        assert all(t.endswith("T03:00:00+08:00") for t in ok["next_fire_times"])
        weekly = (await http.post("/api/schedules/preview", json={
            "trigger": {"type": "weekly", "days": ["sun"], "time": "10:00", "timezone": "UTC"}, "count": 2})).json()
        from datetime import datetime
        assert [datetime.fromisoformat(t).weekday() for t in weekly["next_fire_times"]] == [6, 6]
        assert (await http.post("/api/schedules/preview", json={"trigger": {"type": "daily", "time": "nope"}})
                ).status_code == 422
        assert (await http.post("/api/schedules/preview", json={"trigger": DAILY, "count": 99})).status_code == 422


@pytest.mark.asyncio
async def test_only_admins_reach_the_endpoints(env):
    app, _, _ = env
    paths = [("get", "/api/schedules"), ("get", "/api/schedules/kinds"),
             ("post", "/api/schedules/preview"), ("post", "/api/schedules/1/run")]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 4000)),
                                 base_url=PUBLIC) as public:
        for method, path in paths:
            # Public door, no identity: refused (no internal-admin shortcut there).
            assert (await getattr(public, method)(path)).status_code == 401
            # Gateway-vouched but not admin.
            paying = await getattr(public, method)(
                path, headers={"X-Retrival-User-Id": "u1", "X-Retrival-User-Tier": "paying"})
            assert paying.status_code == 403
        # Gateway-vouched admin passes.
        ok = await public.get("/api/schedules", headers={
            "X-Retrival-User-Id": "root", "X-Retrival-User-Tier": "admin"})
        assert ok.status_code == 200
    # A non-private peer on the internal door is not an admin either.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("8.8.8.8", 4000)),
                                 base_url=INTERNAL) as outsider:
        assert (await outsider.get("/api/schedules")).status_code == 401


def _assert_offset_near_now(text: str, within_s: float = 120.0):
    import re
    from datetime import datetime, timezone

    assert re.search(r"(Z|[+-]\d{2}:\d{2})$", text), f"no explicit offset: {text!r}"
    parsed = datetime.fromisoformat(text)
    # A naive-local value read as UTC would be hours off.
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < within_s, text


@pytest.mark.asyncio
async def test_listing_carries_running_task_times_and_last_tick_task_with_offsets(env):
    from datetime import datetime, timedelta, timezone

    app, service, store = env
    async with admin(app) as http:
        row = (await http.post("/api/schedules", json={
            "key": "stamps", "kind": "unit_kind", "title": "S", "trigger": DAILY})).json()
        fired = (await http.post(f"/api/schedules/{row['id']}/run")).json()
        store.set_task_status(fired["task_id"], "running")             # started_at = datetime.now()

        listed = {r["key"]: r for r in (await http.get("/api/schedules")).json()}["stamps"]
        _assert_offset_near_now(listed["running_task"]["started_at"])
        assert isinstance(listed["running_task"]["started_at"], str)
        for field in ("scheduled_for", "fired_at"):
            _assert_offset_near_now(listed["last_tick"][field])
        assert listed["last_tick"]["task"]["id"] == fired["task_id"]
        assert listed["last_tick"]["task"]["status"] == "running"
        _assert_offset_near_now(listed["last_tick"]["task"]["started_at"])
        assert listed["last_tick"]["task"]["completed_at"] is None

        store.update_result(fired["task_id"], {"new": 2})
        store.set_task_status(fired["task_id"], "completed")
        done = {r["key"]: r for r in (await http.get("/api/schedules")).json()}["stamps"]
        assert done["running_task"] is None
        assert done["last_tick"]["task"]["status"] == "completed"
        assert done["last_tick"]["task"]["result_json"] == {"new": 2}
        _assert_offset_near_now(done["last_tick"]["task"]["completed_at"])
        # the same shape /ticks returns
        [tick] = (await http.get(f"/api/schedules/{row['id']}/ticks")).json()
        assert tick["task"] == done["last_tick"]["task"]

        # A tick stored for a known slot comes back as exactly that instant.
        slot = datetime(2026, 11, 3, 3, 0, tzinfo=timezone.utc)
        service._write_tick(row["id"], slot, "cron", "skipped", "x", None, 0)
        ticks = (await http.get(f"/api/schedules/{row['id']}/ticks")).json()
        stored = next(t for t in ticks if t["reason"] == "x")
        assert datetime.fromisoformat(stored["scheduled_for"]) == slot
        assert row["created_at"].endswith("Z") or "+" in row["created_at"]
        _assert_offset_near_now(row["created_at"])
