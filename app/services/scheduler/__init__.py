"""Recurring jobs: schedule tables + APScheduler as the timing engine.

`kinds` is the lightweight registry (importable by any module that defines a kind);
`service` is the engine and is imported on demand.
"""
