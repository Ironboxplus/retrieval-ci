"""Observable, callback-driven ingest pipeline.

A pipeline run is a parent Task with one TaskSubtask per stage. A generic,
stage-agnostic orchestrator drives an ordered list of `StageSpec` callbacks
(run / probe / gate / preflight). Progress is measured by counting output
artifacts, so it is decoupled from whether a stage runs locally or on KSS and
survives a backend restart.

Public surface:
    StageSpec, Progress, GateResult  — the stage contract (spec.py)
    TaskStore                        — generic Task/Subtask/Event DB I/O (store.py)
    PipelineOrchestrator             — the engine (orchestrator.py)
"""
from __future__ import annotations

from app.services.pipeline.spec import StageSpec, Progress, GateResult
from app.services.pipeline.store import TaskStore
from app.services.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "StageSpec",
    "Progress",
    "GateResult",
    "TaskStore",
    "PipelineOrchestrator",
]
