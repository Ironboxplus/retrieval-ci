"""
PDF文件解析器
"""
from pypdf import PdfReader
from typing import List, Tuple
import re

from app.models.schemas import Document
from config import settings


class PDFParser:
    """解析PDF文件"""

    def __init__(self):
        pass

    def parse(self, file_path: str) -> Tuple[str, dict]:
        """
        解析PDF文件

        Args:
            file_path: PDF文件路径

        Returns:
            (文本内容, 元数据)
        """
        reader = PdfReader(file_path)

        # 提取元数据
        metadata = {}
        if reader.metadata:
            metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
                "producer": reader.metadata.get("/Producer", ""),
                "creation_date": str(reader.metadata.get("/CreationDate", "")),
            }

        # 提取文本
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # 清理文本
                page_text = self._clean_text(page_text)
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

        full_text = "\n\n".join(text_parts)

        # 添加页数信息
        metadata["page_count"] = len(reader.pages)

        return full_text, metadata

    def _clean_text(self, text: str) -> str:
        """清理PDF提取的文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 修复断行
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        # 恢复段落
        text = re.sub(r'\.\s+([A-Z])', r'.\n\n\1', text)
        return text.strip()

    def parse_to_pages(self, file_path: str) -> List[Tuple[int, str]]:
        """
        按页解析PDF

        Args:
            file_path: PDF文件路径

        Returns:
            [(页码, 文本内容), ...]
        """
        reader = PdfReader(file_path)
        pages = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = self._clean_text(page_text)
                pages.append((page_num + 1, page_text))

        return pages
