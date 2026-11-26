"""Robotics conference corpus preparation: ICRA, CoRL, RSS.

Official open-access proceedings where available (RSS, CoRL/PMLR); ICRA via
DBLP catalog + IEEE/CF-aware download. Network I/O always uses the local
proxy ``http://127.0.0.1:10808``.
"""

from pipeline.robotics_corpus.models import RoboticsPaperRecord, RoboticsVenue
from pipeline.robotics_corpus.prepare import PrepareConfig, collect_manifest, prepare, run_download_pass
from pipeline.robotics_corpus.sources import DEFAULT_TARGET_YEARS, DEFAULT_VENUES

__all__ = [
    "DEFAULT_TARGET_YEARS",
    "DEFAULT_VENUES",
    "PrepareConfig",
    "RoboticsPaperRecord",
    "RoboticsVenue",
    "collect_manifest",
    "prepare",
    "run_download_pass",
]
