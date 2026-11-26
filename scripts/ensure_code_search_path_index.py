"""Install the reversible covering index used by Code KB path selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time


INDEX_NAME = 'ix_index_catalog_code_path_lookup'
INDEX_COLUMNS = ('owner_user_id', 'kind', 'deleted_at', 'rel_path', 'project', 'chunk_count')


def ensure_code_search_path_index(connection: sqlite3.Connection) -> None:
    connection.execute(
        'CREATE INDEX IF NOT EXISTS ' + INDEX_NAME + ' ON index_catalog (' +
        ', '.join(INDEX_COLUMNS) + ')')
    actual = tuple(row[2] for row in connection.execute('PRAGMA index_info(' + INDEX_NAME + ')'))
    if actual != INDEX_COLUMNS:
        raise RuntimeError('Existing Code KB path index has an unexpected definition')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    args = parser.parse_args()
    database = args.database.resolve(strict=True)
    started = time.monotonic()
    with sqlite3.connect(database.as_uri() + '?mode=rw', uri=True, timeout=10) as connection:
        connection.execute('PRAGMA busy_timeout=10000')
        connection.execute('BEGIN IMMEDIATE')
        before = connection.execute('SELECT count(*) FROM index_catalog').fetchone()[0]
        ensure_code_search_path_index(connection)
        after = connection.execute('SELECT count(*) FROM index_catalog').fetchone()[0]
        assert before == after
        connection.commit()
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT DISTINCT rel_path FROM index_catalog "
            "WHERE owner_user_id=? AND kind='code' AND deleted_at IS NULL AND chunk_count>0",
            ('internal-admin',),
        ).fetchall()
    print(json.dumps({'database': str(database), 'index': INDEX_NAME,
        'rows_before': before, 'rows_after': after,
        'elapsed_s': time.monotonic() - started, 'query_plan': plan}))


if __name__ == '__main__':
    main()
