"""
数据库迁移脚本 - 添加任务心跳和超时相关字段

用于将现有数据库更新到新版本，添加以下列:
- worker_id: 处理Worker的ID
- last_heartbeat: 最后心跳时间
- timeout_seconds: 任务超时时间(秒)
"""
import sqlite3
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


def get_db_path():
    """获取数据库路径"""
    return Path(settings.data_dir) / "db" / "retrieval.db"


def get_existing_columns(cursor, table_name):
    """获取表的现有列"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def add_column_if_not_exists(cursor, table_name, column_name, column_type, default=None):
    """如果列不存在则添加"""
    existing_columns = get_existing_columns(cursor, table_name)

    if column_name in existing_columns:
        print(f"  [SKIP] Column '{column_name}' already exists in '{table_name}'")
        return False

    if default is not None:
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} DEFAULT {default}"
    else:
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"

    cursor.execute(sql)
    print(f"  [ADD] Column '{column_name}' added to '{table_name}'")
    return True


def migrate():
    """执行迁移"""
    db_path = get_db_path()
    print(f"Database path: {db_path}")

    if not db_path.exists():
        print("Error: Database file not found!")
        return False

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        print("\nMigrating 'tasks' table...")

        # 添加 worker_id 列
        add_column_if_not_exists(
            cursor, "tasks", "worker_id",
            "VARCHAR(100)", None
        )

        # 添加 last_heartbeat 列
        add_column_if_not_exists(
            cursor, "tasks", "last_heartbeat",
            "DATETIME", None
        )

        # 添加 timeout_seconds 列
        add_column_if_not_exists(
            cursor, "tasks", "timeout_seconds",
            "INTEGER", "3600"
        )

        conn.commit()
        print("\nMigration completed successfully!")
        return True

    except Exception as e:
        print(f"\nMigration failed: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def verify():
    """验证迁移结果"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print("\nVerifying 'tasks' table structure:")
    cursor.execute("PRAGMA table_info(tasks)")
    for row in cursor.fetchall():
        print(f"  {row[1]}: {row[2]}")

    conn.close()


if __name__ == "__main__":
    migrate()
    verify()
