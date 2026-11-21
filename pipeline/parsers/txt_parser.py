"""
TXT文件解析器
"""
from typing import Tuple
import chardet
import re


class TXTParser:
    """解析TXT文件"""

    def __init__(self):
        pass

    def parse(self, file_path: str) -> Tuple[str, dict]:
        """
        解析TXT文件

        Args:
            file_path: TXT文件路径

        Returns:
            (文本内容, 元数据)
        """
        # 检测编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')

        # 读取内容
        try:
            content = raw_data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # 回退到utf-8
            content = raw_data.decode('utf-8', errors='ignore')

        # 清理内容
        content = self._clean_text(content)

        metadata = {
            "encoding": encoding,
            "confidence": detected.get('confidence', 0),
            "char_count": len(content),
            "line_count": content.count('\n') + 1
        }

        return content, metadata

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 移除空白行过多的情况
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def parse_bytes(self, data: bytes) -> Tuple[str, dict]:
        """
        解析字节数据

        Args:
            data: 字节数据

        Returns:
            (文本内容, 元数据)
        """
        detected = chardet.detect(data)
        encoding = detected.get('encoding', 'utf-8')

        try:
            content = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            content = data.decode('utf-8', errors='ignore')

        content = self._clean_text(content)

        metadata = {
            "encoding": encoding,
            "confidence": detected.get('confidence', 0),
            "char_count": len(content),
            "line_count": content.count('\n') + 1
        }

        return content, metadata
