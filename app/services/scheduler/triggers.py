"""The trigger shapes a schedule may use, and how they become APScheduler triggers.

Only three are offered: `daily` at a wall-clock time, `weekly` on named weekdays,
and `interval` in seconds. There is deliberately no raw cron string: APScheduler
3.x `CronTrigger.from_crontab` numbers weekdays from Monday = 0, the opposite of
cron/cronstrue (Sunday = 0), so a pasted expression would silently fire a day off.
Weekdays are therefore names (`mon`..`sun`), which CronTrigger takes verbatim
(apscheduler/triggers/cron/fields.py WEEKDAYS).

The timezone is stored with every trigger, so "03:00 Asia/Shanghai" keeps meaning
that across DST rules and a server timezone change. CronTrigger computes the slots
in that zone (apscheduler/triggers/cron/__init__.py get_next_fire_time).

Reference: APScheduler 3.11.3 triggers/{cron,interval}.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, Field, TypeAdapter, field_validator

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _check_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(f"unknown timezone {name!r}") from exc
    return name


def _check_clock(value: str) -> str:
    try:
        hour, minute = value.split(":")
        if len(hour) != 2 or len(minute) != 2 or not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"time must be HH:MM (00:00-23:59), got {value!r}") from exc
    return value


class _Zoned(BaseModel):
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def _tz(cls, value: str) -> str:
        return _check_timezone(value)


class DailySpec(_Zoned):
    type: Literal["daily"]
    time: str

    @field_validator("time")
    @classmethod
    def _time(cls, value: str) -> str:
        return _check_clock(value)


class WeeklySpec(_Zoned):
    type: Literal["weekly"]
    days: list[Weekday] = Field(min_length=1)
    time: str

    @field_validator("time")
    @classmethod
    def _time(cls, value: str) -> str:
        return _check_clock(value)

    @field_validator("days")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("days must not repeat")
        return value


class IntervalSpec(_Zoned):
    type: Literal["interval"]
    seconds: int = Field(ge=1)


TriggerSpec = Annotated[Union[DailySpec, WeeklySpec, IntervalSpec], Field(discriminator="type")]
_adapter: TypeAdapter = TypeAdapter(TriggerSpec)


def parse_trigger(trigger_json: dict) -> DailySpec | WeeklySpec | IntervalSpec:
    """Validate a stored/submitted trigger dict; raises pydantic.ValidationError."""
    return _adapter.validate_python(trigger_json)


def build_trigger(spec: DailySpec | WeeklySpec | IntervalSpec) -> BaseTrigger:
    if isinstance(spec, IntervalSpec):
        return IntervalTrigger(seconds=spec.seconds, timezone=spec.timezone)
    hour, minute = (int(part) for part in spec.time.split(":"))
    if isinstance(spec, WeeklySpec):
        return CronTrigger(
            day_of_week=",".join(spec.days), hour=hour, minute=minute, timezone=spec.timezone,
        )
    return CronTrigger(hour=hour, minute=minute, timezone=spec.timezone)


def next_fire_times(
    spec: DailySpec | WeeklySpec | IntervalSpec,
    count: int = 5,
    *,
    after: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> list[datetime]:
    """The next `count` fire times (timezone-aware, in the trigger's own zone).

    Chains get_next_fire_time(previous, now) like Job._get_run_times does; `after`
    is the previous fire time to continue from (None = the first slot after `now`).
    CronTrigger searches from min(now, previous + 1us), and the scheduler only ever
    calls it with now >= previous, so a preview into the future must advance `now`
    with `previous` or every call would return the same slot.
    """
    return _chain(build_trigger(spec), after, count, now or datetime.now(timezone.utc))


def fire_times_from(trigger: BaseTrigger, first: datetime, count: int = 5) -> list[datetime]:
    """`first` followed by the slots after it, using a live job's own trigger.

    An interval trigger is anchored at the moment it was built, so a freshly built
    one disagrees with the running job; the job's own trigger object does not.
    """
    if count < 1:
        return []
    return [first, *_chain(trigger, first, count - 1, first)]


def _chain(trigger: BaseTrigger, previous: Optional[datetime], count: int, now: datetime) -> list[datetime]:
    times: list[datetime] = []
    for _ in range(count):
        nxt = trigger.get_next_fire_time(previous, max(now, previous) if previous else now)
        if nxt is None:
            break
        times.append(nxt)
        previous = nxt
    return times
