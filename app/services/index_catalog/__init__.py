"""Structured index-inventory catalog (user → host → project → dir → file)."""
from app.services.index_catalog.service import IndexCatalogService, get_index_catalog_service

__all__ = ["IndexCatalogService", "get_index_catalog_service"]
