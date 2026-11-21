#!/usr/bin/env python
import sqlite3
import sys

db_path = r'E:\JETB\pythons\sf\retrival\data\index\chroma\chroma.sqlite3'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print('Tables:', [t[0] for t in tables])

    cursor.execute('SELECT * FROM collections')
    for c in cursor.fetchall():
        print(f'Collection: {c}')

    cursor.execute('SELECT COUNT(*) FROM embeddings')
    print(f'Embeddings count: {cursor.fetchone()[0]}')

    # Get sample embedding metadata
    cursor.execute('SELECT * FROM embedding_metadata LIMIT 3')
    print('\nSample metadata:')
    for row in cursor.fetchall():
        print(f'  {row}')

    conn.close()
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
