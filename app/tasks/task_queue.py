"""
任务队列配置 - 使用Huey (SQLite模式)
"""
from huey import SqliteHuey
from pathlib import Path
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

HUEY_DB_PATH = Path(settings.data_dir) / "tasks" / "huey.db"
HUEY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

huey = SqliteHuey(
    filename=str(HUEY_DB_PATH),
    immediate=os.environ.get('HUEY_IMMEDIATE', 'false').lower() == 'true',
)

def get_huey():
    return huey
