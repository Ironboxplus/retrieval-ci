"""Server-side code knowledge base (4090 Qdrant `code_chunks`)."""
from app.services.code_kb.service import CodeKBService, get_code_kb_service

__all__ = ["CodeKBService", "get_code_kb_service"]
