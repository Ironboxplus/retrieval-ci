"""Scheduler engine against real storage and a real AsyncIOScheduler.

Schedules and ticks live in a real SQLite file; timing is APScheduler's own
(second-level interval triggers, or a daily slot a couple of minutes in the past so
catch-up is exercised without waiting). The job kinds and the task runner below are
small real classes that write real rows, not stand-ins for the unit under test.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

from app.services.pipeline import runner_registry
from app.services.pipeline.runner_registry import register_runner
from app.services.pipeline.store import TaskStore
from app.services.scheduler.kinds import (
    FireResult,
    JobKind,
    RetryPolicy,
    ScheduleSeed,
    all_kinds,
    get_kind,
)
from app.services.scheduler.service import SchedulerService
from app.services.scheduler.triggers import build_trigger, next_fire_times, parse_trigger
from core.db.models import ScheduleTick
from tests.sqlite_task_db import make_session_factory

SHANGHAI = ZoneInfo("Asia/Shanghai")


class Params(BaseModel):
    limit: int = 0


class Recorder:
    """The fire() side of a job kind: remembers every call, returns a canned result."""

    def __init__(self, result=None, error: Exception | None = None):
        self.calls: list[tuple[Params, dict]] = []
        self.result = result or FireResult("enqueued", task_id=None, reason="ok")
        self.error = error

    def __call__(self, params, origin):
        self.calls.append((params, origin))
        if self.error is not None:
            raise self.error
        return self.result


def make_kind(recorder: Recorder, **overrides) -> JobKind:
    fields = dict(name="unit_kind", title="Unit kind", params_model=Params, fire=recorder,
                  default_misfire_grace_s=86400)
    fields.update(overrides)
    return JobKind(**fields)


@pytest.fixture
def db(tmp_path):
    return make_session_factory(tmp_path / "sched.db")


def service_for(db, tmp_path, *kinds: JobKind, lock=True) -> SchedulerService:
    registry = {k.name: k for k in kinds}
    return SchedulerService(
        session_factory=db, kinds=lambda: registry,
        lock_path=str(tmp_path / "scheduler.lock") if lock else None,
    )


def ticks(db, schedule_id=None) -> list[dict]:
    with db() as s:
        q = s.query(ScheduleTick)
        if schedule_id is not None:
            q = q.filter(ScheduleTick.schedule_id == schedule_id)
        return [
            {"scheduled_for": t.scheduled_for, "trigger": t.trigger, "status": t.status,
             "reason": t.reason, "task_id": t.task_id, "attempt": t.attempt}
            for t in q.order_by(ScheduleTick.id).all()
        ]


def add_tick(db, schedule_id, slot_utc_aware: datetime, trigger="cron"):
    with db() as s:
        s.add(ScheduleTick(
            schedule_id=schedule_id, scheduled_for=slot_utc_aware.astimezone(timezone.utc).replace(tzinfo=None),
            trigger=trigger, status="enqueued",
        ))


async def until(predicate, timeout=4.0, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return predicate()


def minutes_ago_slot(minutes=2) -> datetime:
    """A wall-clock minute in Asia/Shanghai that is `minutes`..`minutes+1` in the past."""
    return (datetime.now(SHANGHAI) - timedelta(minutes=minutes)).replace(second=0, microsecond=0)


DAILY_0300 = {"type": "daily", "time": "03:00", "timezone": "Asia/Shanghai"}


def daily_at(slot: datetime) -> dict:
    return {"type": "daily", "time": slot.strftime("%H:%M"), "timezone": "Asia/Shanghai"}


# ------------------------------------------------------------- catch-up (R1/R3)
@pytest.mark.asyncio
async def test_downtime_missed_slot_runs_exactly_once_with_the_slot_recorded(db, tmp_path):
    recorder = Recorder()
    service = service_for(db, tmp_path, make_kind(recorder))
    slot = minutes_ago_slot()
    row = service.create(key="daily", kind="unit_kind", title="d", trigger=daily_at(slot),
                         misfire_grace_s=86400)
    # The process was down for two days: last recorded slot is two days before `slot`.
    add_tick(db, row["id"], slot - timedelta(days=2))

    assert service.start()
    try:
        assert await until(lambda: len(ticks(db, row["id"])) == 2)
        await asyncio.sleep(0.3)  # a second fire would show up here
        assert len(recorder.calls) == 1
        written = ticks(db, row["id"])[-1]
        assert written["status"] == "enqueued" and written["trigger"] == "cron"
        assert written["scheduled_for"] == slot.astimezone(timezone.utc).replace(tzinfo=None)
        assert recorder.calls[0][1]["schedule_id"] == row["id"]
        assert recorder.calls[0][1]["trigger"] == "cron"
    finally:
        service.shutdown()


@pytest.mark.asyncio
async def test_late_beyond_grace_is_recorded_as_missed_and_not_fired(db, tmp_path):
    recorder = Recorder()
    service = service_for(db, tmp_path, make_kind(recorder))
    slot = minutes_ago_slot()
    row = service.create(key="daily", kind="unit_kind", title="d", trigger=daily_at(slot),
                         misfire_grace_s=60)  # the slot is at least 2 min old
    add_tick(db, row["id"], slot - timedelta(days=2))

    assert service.start()
    try:
        assert await until(lambda: len(ticks(db, row["id"])) == 2)
        assert ticks(db, row["id"])[-1]["status"] == "missed"
        assert recorder.calls == []
    finally:
        service.shutdown()


@pytest.mark.asyncio
async def test_first_start_and_skip_mode_do_not_replay_the_past(db, tmp_path):
    recorder = Recorder()
    service = service_for(db, tmp_path, make_kind(recorder))
    slot = minutes_ago_slot()
    fresh = service.create(key="fresh", kind="unit_kind", title="never ran", trigger=daily_at(slot))
    skipping = service.create(key="skipper", kind="unit_kind", title="skip", trigger=daily_at(slot),
                              catchup="skip")
    add_tick(db, skipping["id"], slot - timedelta(days=2))

    assert service.start()
    try:
        await asyncio.sleep(0.6)
        assert recorder.calls == []
        by_key = {s["key"]: s for s in service.list_schedules()}
        for key in ("fresh", "skipper"):
            assert by_key[key]["next_run_time"] is not None
            assert datetime.fromisoformat(by_key[key]["next_run_time"]) > datetime.now(timezone.utc)
    finally:
        service.shutdown()


# --------------------------------------------------------------- ticks (R2)
@pytest.mark.asyncio
async def test_interval_schedule_fires_and_idle_ticks_follow_the_kind(db, tmp_path):
    chatty = Recorder(FireResult("noop"))
    quiet = Recorder(FireResult("noop"))
    service = service_for(
        db, tmp_path,
        make_kind(chatty, name="chatty"),
        make_kind(quiet, name="quiet", record_idle_ticks=False),
    )
    trigger = {"type": "interval", "seconds": 1}
    a = service.create(key="a", kind="chatty", title="a", trigger=trigger, catchup="skip")
    b = service.create(key="b", kind="quiet", title="b", trigger=trigger, catchup="skip")

    assert service.start()
    try:
        assert await until(lambda: len(quiet.calls) >= 1 and len(ticks(db, a["id"])) >= 1, timeout=4.5)
        assert ticks(db, a["id"])[0]["status"] == "skipped"
        assert ticks(db, a["id"])[0]["reason"] == "nothing to do"
        assert ticks(db, b["id"]) == []
    finally:
        service.shutdown()


@pytest.mark.asyncio
async def test_a_fire_that_raises_is_recorded_as_an_error_tick(db, tmp_path):
    recorder = Recorder(error=RuntimeError("boom"))
    service = service_for(db, tmp_path, make_kind(recorder))
    row = service.create(key="bad", kind="unit_kind", title="b",
                         trigger={"type": "interval", "seconds": 1}, catchup="skip")
    assert service.start()
    try:
        assert await until(lambda: len(ticks(db, row["id"])) >= 1, timeout=4.5)
        first = ticks(db, row["id"])[0]
        assert first["status"] == "error" and "RuntimeError: boom" in first["reason"]
    finally:
        service.shutdown()


def test_a_slot_is_recorded_once_per_trigger_kind(db, tmp_path):
    service = service_for(db, tmp_path, make_kind(Recorder()))
    row = service.create(key="k", kind="unit_kind", title="t", trigger={"type": "interval", "seconds": 5})
    slot = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for _ in range(3):
        service._write_tick(row["id"], slot, "cron", "enqueued", None, None, 0)
    service._write_tick(row["id"], slot, "retry", "enqueued", None, None, 1)
    assert [t["trigger"] for t in ticks(db, row["id"])] == ["cron", "retry"]


# ------------------------------------------------------------- log volume
@pytest.mark.asyncio
async def test_per_run_apscheduler_info_logging_is_silenced(db, tmp_path):
    import logging

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture(level=logging.DEBUG)
    logging.getLogger("apscheduler").addHandler(handler)
    recorder = Recorder(FireResult("noop"))
    service = service_for(db, tmp_path, make_kind(recorder))
    service.create(key="poll", kind="unit_kind", title="p", trigger={"type": "interval", "seconds": 1},
                   catchup="skip")
    assert service.start()
    try:
        assert await until(lambda: len(recorder.calls) >= 1, timeout=4.5)
        await asyncio.sleep(0.2)
    finally:
        service.shutdown()
        logging.getLogger("apscheduler").removeHandler(handler)
    noisy = [r for r in records if r.name in ("apscheduler.executors.default", "apscheduler.scheduler")
             and r.levelno < logging.WARNING]
    assert noisy == [], [r.getMessage() for r in noisy]


# --------------------------------------------------- preview follows the live job
@pytest.mark.asyncio
async def test_preview_starts_at_the_live_jobs_next_run_and_keeps_its_spacing(db, tmp_path):
    service = service_for(db, tmp_path, make_kind(Recorder()))
    interval = service.create(key="iv", kind="unit_kind", title="i", catchup="skip",
                              trigger={"type": "interval", "seconds": 45})
    daily = service.create(key="dl", kind="unit_kind", title="d", trigger=DAILY_0300)
    paused = service.create(key="off", kind="unit_kind", title="o", trigger=DAILY_0300, enabled=False)
    assert service.start()
    try:
        await asyncio.sleep(0.4)  # a trigger rebuilt now would be anchored 0.4 s later than the job's
        rows = {r["key"]: r for r in service.list_schedules()}

        iv = rows["iv"]
        times = [datetime.fromisoformat(t) for t in iv["next_fire_times"]]
        assert len(times) == 5
        assert times[0] == datetime.fromisoformat(iv["next_run_time"])
        assert all(b - a == timedelta(seconds=45) for a, b in zip(times, times[1:]))

        dl = rows["dl"]
        assert dl["next_fire_times"][0] == dl["next_run_time"]
        dtimes = [datetime.fromisoformat(t) for t in dl["next_fire_times"]]
        assert all(b - a == timedelta(days=1) for a, b in zip(dtimes, dtimes[1:]))

        # No live job (paused): falls back to the computed preview or nothing.
        assert rows["off"]["next_run_time"] is None and rows["off"]["next_fire_times"] == []
    finally:
        service.shutdown()


# ------------------------------------------------------- pause / edit / seeds
@pytest.mark.asyncio
async def test_pause_resume_and_edit_recompute_the_next_run(db, tmp_path):
    service = service_for(db, tmp_path, make_kind(Recorder()))
    row = service.create(key="s", kind="unit_kind", title="s",
                         trigger={"type": "daily", "time": "03:00", "timezone": "Asia/Shanghai"})
    assert service.start()
    try:
        def next_run():
            return next(s for s in service.list_schedules() if s["id"] == row["id"])["next_run_time"]

        before = next_run()
        assert before is not None
        assert datetime.fromisoformat(before).astimezone(SHANGHAI).strftime("%H:%M") == "03:00"

        service.update(row["id"], {"enabled": False})
        assert next_run() is None and service._scheduler.get_job(f"schedule:{row['id']}") is None

        service.update(row["id"], {"enabled": True})
        assert next_run() == before  # nothing was lost while paused

        service.update(row["id"], {"trigger": {"type": "daily", "time": "04:30", "timezone": "Asia/Shanghai"}})
        edited = next_run()
        assert edited != before
        assert datetime.fromisoformat(edited).astimezone(SHANGHAI).strftime("%H:%M") == "04:30"
    finally:
        service.shutdown()


@pytest.mark.asyncio
async def test_seeds_are_inserted_once_and_never_overwrite_user_edits(db, tmp_path):
    seed = ScheduleSeed(key="seeded", title="Original", trigger={"type": "interval", "seconds": 3600},
                        params={"limit": 5}, catchup="skip")
    kind = make_kind(Recorder(), defaults=(seed,))

    first = service_for(db, tmp_path, kind)
    assert first.start()
    row = next(r for r in first.list_schedules() if r["key"] == "seeded")
    assert row["system"] is True and row["params_json"] == {"limit": 5}
    first.update(row["id"], {"title": "Mine", "params": {"limit": 9},
                             "trigger": {"type": "interval", "seconds": 60}, "enabled": False})
    with pytest.raises(PermissionError):
        first.delete(row["id"])
    first.shutdown()

    second = service_for(db, tmp_path, kind)
    assert second.start()
    try:
        again = next(r for r in second.list_schedules() if r["key"] == "seeded")
        assert (again["id"], again["title"], again["params_json"], again["enabled"]) == (
            row["id"], "Mine", {"limit": 9}, False)
        assert again["trigger_json"]["seconds"] == 60
        assert len([r for r in second.list_schedules() if r["key"] == "seeded"]) == 1
    finally:
        second.shutdown()


def test_create_validates_trigger_params_and_key(db, tmp_path):
    service = service_for(db, tmp_path, make_kind(Recorder()))
    good = {"type": "interval", "seconds": 10}
    with pytest.raises(ValidationError):
        service.create(key="a", kind="unit_kind", title="t", trigger={"type": "daily", "time": "25:00"})
    with pytest.raises(ValidationError):
        service.create(key="a", kind="unit_kind", title="t", trigger=good, params={"limit": "many"})
    with pytest.raises(ValueError):
        service.create(key="a", kind="nope", title="t", trigger=good)
    with pytest.raises(ValueError):
        service.create(key="a", kind="unit_kind", title="t", trigger=good, catchup="always")
    service.create(key="a", kind="unit_kind", title="t", trigger=good)
    with pytest.raises(KeyError):
        service.create(key="a", kind="unit_kind", title="t", trigger=good)


# ----------------------------------------------------------- manual run / retry
def test_run_now_fires_regardless_of_enabled_and_records_a_manual_tick(db, tmp_path):
    recorder = Recorder(FireResult("enqueued", task_id=41, reason="started"))
    service = service_for(db, tmp_path, make_kind(recorder))
    row = service.create(key="m", kind="unit_kind", title="m", trigger={"type": "interval", "seconds": 60},
                         params={"limit": 3}, enabled=False)

    result = service.run_now(row["id"])

    assert result.status == "enqueued" and result.task_id == 41
    params, origin = recorder.calls[0]
    assert params.limit == 3
    assert origin["trigger"] == "manual" and origin["schedule_id"] == row["id"] and origin["attempt"] == 0
    [tick] = ticks(db, row["id"])
    assert (tick["trigger"], tick["status"], tick["task_id"]) == ("manual", "enqueued", 41)

    failing = Recorder(error=ValueError("bad params"))
    service2 = service_for(db, tmp_path, make_kind(failing))
    with pytest.raises(ValueError):
        service2.run_now(row["id"])
    assert ticks(db, row["id"])[-1]["status"] == "error"


class RetryRunner:
    """A TaskRunner whose tasks are real rows; retry() re-drives them like PipelineRunner does."""

    task_type = "unit_retry_runner"

    def __init__(self, store: TaskStore):
        self.store = store
        self.retries: list[tuple[int, float]] = []

    def start(self, params):
        task_id = self.store.create_task(task_type=self.task_type, params=params)
        self.store.set_task_status(task_id, "running")
        return {"status": "started", "task_id": task_id}

    def cancel(self, task_id):
        return False

    def retry(self, task_id):
        self.retries.append((task_id, time.monotonic()))
        self.store.reset_pipeline_for_retry(task_id)
        self.store.set_task_status(task_id, "running")
        return {"status": "started", "task_id": task_id}

    def resume_running(self):
        return []


@pytest.mark.asyncio
async def test_failed_scheduled_task_is_retried_with_backoff_up_to_the_policy(db, tmp_path):
    store = TaskStore(db)
    runner = RetryRunner(store)
    register_runner(runner.task_type, lambda: runner)

    started: list[int] = []

    def fire(params, origin):  # the kind starts a task carrying its origin
        task_id = runner.start({"origin": origin})["task_id"]
        started.append(task_id)
        return FireResult("enqueued", task_id=task_id)

    kind = make_kind(Recorder(), fire=fire, retry=RetryPolicy(max_retries=2, base_delay_s=1, backoff=1.0))
    service = service_for(db, tmp_path, kind)
    row = service.create(key="r", kind="unit_kind", title="r", trigger={"type": "interval", "seconds": 3600},
                         catchup="skip")
    assert service.start()
    try:
        await asyncio.to_thread(service.run_now, row["id"])
        task_id = started[0]
        assert store.get_task(task_id)["params"]["origin"]["schedule_id"] == row["id"]

        # Unrelated failures never trigger a retry.
        stranger = store.create_task(task_type=runner.task_type, params={})
        store.set_task_status(stranger, "failed")

        failed_at = time.monotonic()
        store.set_task_status(task_id, "failed", error="stage x failed")        # failure 1
        assert await until(lambda: len(runner.retries) == 1, timeout=4.0)
        assert 0.8 <= runner.retries[0][1] - failed_at < 3.0                       # ~base_delay_s

        store.set_task_status(task_id, "failed", error="stage x failed again")  # failure 2
        assert await until(lambda: len(runner.retries) == 2, timeout=4.0)

        store.set_task_status(task_id, "failed", error="and again")             # failure 3: budget spent
        await asyncio.sleep(1.6)
        assert [t for t, _ in runner.retries] == [task_id, task_id]
        assert stranger not in [t for t, _ in runner.retries]

        retry_ticks = [t for t in ticks(db, row["id"]) if t["trigger"] == "retry"]
        assert [(t["status"], t["task_id"], t["attempt"]) for t in retry_ticks] == [
            ("enqueued", task_id, 1), ("enqueued", task_id, 2)]
    finally:
        service.shutdown()
        runner_registry._getters.pop(runner.task_type, None)


# ------------------------------------------------------------ single instance (R4)
@pytest.mark.asyncio
async def test_only_one_service_per_lock_file_runs_the_engine(db, tmp_path):
    kind = make_kind(Recorder())
    first = service_for(db, tmp_path, kind)
    second = service_for(db, tmp_path, kind)
    assert first.start() is True
    try:
        assert second.start() is False and second.running is False
        # The idle process still serves the tables.
        assert second.list_schedules() == []
    finally:
        first.shutdown()
    assert second.start() is True  # lock released by the first
    second.shutdown()


# ------------------------------------------------------------------ triggers (R5)
def test_trigger_validation_rejects_bad_shapes():
    for bad in (
        {"type": "daily", "time": "24:00"},
        {"type": "daily", "time": "3:00"},
        {"type": "daily", "time": "03:00", "timezone": "Mars/Olympus"},
        {"type": "weekly", "days": [], "time": "03:00"},
        {"type": "weekly", "days": ["mon", "mon"], "time": "03:00"},
        {"type": "weekly", "days": ["monday"], "time": "03:00"},
        {"type": "interval", "seconds": 0},
        {"type": "cron", "expr": "* * * * *"},
    ):
        with pytest.raises(ValidationError):
            parse_trigger(bad)


def test_weekly_uses_weekday_names_not_cron_numbers():
    spec = parse_trigger({"type": "weekly", "days": ["mon", "wed"], "time": "03:00",
                          "timezone": "Asia/Shanghai"})
    times = next_fire_times(spec, 6, now=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))
    assert [t.weekday() for t in times] == [2, 0, 2, 0, 2, 0]  # Wed, Mon, Wed, ... (0 = Monday)
    assert all(t.hour == 3 and t.minute == 0 and t.utcoffset() == timedelta(hours=8) for t in times)


def test_daily_preview_chains_consecutive_days_in_the_stored_zone():
    spec = parse_trigger({"type": "daily", "time": "03:00", "timezone": "Asia/Shanghai"})
    times = next_fire_times(spec, 5, now=datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc))
    assert len(times) == 5
    assert all(b - a == timedelta(days=1) for a, b in zip(times, times[1:]))
    assert times[0] == datetime(2026, 11, 4, 3, 0, tzinfo=SHANGHAI)  # 03:00 on the 3rd had passed (08:00 there)
    assert build_trigger(spec).get_next_fire_time(None, datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)) == times[0]


# ------------------------------------------------------------------- kinds
def test_builtin_kinds_include_public_code_updates_with_its_polling_seed():
    kind = get_kind("public_code_updates")
    assert "public_code_updates" in all_kinds()
    assert kind.record_idle_ticks is False
    [seed] = kind.defaults
    assert seed.catchup == "skip" and parse_trigger(seed.trigger).seconds == 30
    assert kind.params_model.model_json_schema()["properties"] == {}


def test_retry_policy_backoff_is_capped():
    policy = RetryPolicy(max_retries=5, base_delay_s=1800, backoff=2.0, max_delay_s=6 * 3600)
    assert [policy.delay_s(n) for n in range(5)] == [1800, 3600, 7200, 14400, 21600]
