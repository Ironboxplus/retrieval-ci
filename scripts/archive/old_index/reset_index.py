"""
重置索引 - 清理失败的文档记录和Whoosh索引
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.db import get_db_session, init_db
from core.db.models import Document, Chunk, DocumentStatus, SourceType

def main():
    init_db()

    with get_db_session() as session:
        # 删除所有pending或failed状态的webpage文档
        deleted = session.query(Document).filter(
            Document.source_type == SourceType.webpage,
            Document.status.in_([DocumentStatus.pending, DocumentStatus.failed])
        ).delete(synchronize_session='fetch')

        print(f"Deleted {deleted} failed/pending documents")
        session.commit()

if __name__ == "__main__":
    main()
