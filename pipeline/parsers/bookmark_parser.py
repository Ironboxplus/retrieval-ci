"""
HTML书签解析器
支持 Chrome/Firefox 导出的书签HTML格式
"""
from bs4 import BeautifulSoup
from typing import List, Optional
from datetime import datetime
import re

from app.models.schemas import BookmarkItem


class BookmarkParser:
    """解析HTML书签文件"""

    def __init__(self):
        self.bookmarks: List[BookmarkItem] = []

    def parse(self, html_content: str) -> List[BookmarkItem]:
        """
        解析HTML书签内容

        Args:
            html_content: HTML书签文件内容

        Returns:
            书签列表
        """
        self.bookmarks = []
        soup = BeautifulSoup(html_content, 'lxml')

        # 递归解析书签结构
        self._parse_dl(soup.find('dl'), folder_path="")

        return self.bookmarks

    def _parse_dl(self, dl_element, folder_path: str = ""):
        """递归解析<dl>元素"""
        if dl_element is None:
            return

        current_folder = folder_path

        for child in dl_element.children:
            if child.name == 'dt':
                # 检查是否是文件夹
                h3 = child.find('h3', recursive=False)
                if h3:
                    # 这是一个文件夹
                    folder_name = h3.get_text(strip=True)
                    new_folder_path = f"{folder_path}/{folder_name}" if folder_path else folder_name

                    # 查找文件夹下的<dl>
                    sub_dl = child.find('dl', recursive=False)
                    if sub_dl:
                        self._parse_dl(sub_dl, new_folder_path)
                else:
                    # 这是一个书签链接
                    a_tag = child.find('a', recursive=False)
                    if a_tag:
                        bookmark = self._parse_bookmark(a_tag, current_folder)
                        if bookmark:
                            self.bookmarks.append(bookmark)

    def _parse_bookmark(self, a_tag, folder: str) -> Optional[BookmarkItem]:
        """解析单个书签"""
        url = a_tag.get('href', '')
        if not url or not url.startswith(('http://', 'https://')):
            return None

        title = a_tag.get_text(strip=True) or url

        # 解析添加时间
        add_date = None
        add_date_str = a_tag.get('add_date')
        if add_date_str:
            try:
                # Chrome书签使用Unix时间戳(秒)
                timestamp = int(add_date_str)
                add_date = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                pass

        # 解析标签
        tags = []
        tags_str = a_tag.get('tags', '')
        if tags_str:
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]

        return BookmarkItem(
            title=title,
            url=url,
            add_date=add_date,
            folder=folder if folder else None,
            tags=tags
        )

    def parse_file(self, file_path: str) -> List[BookmarkItem]:
        """
        解析书签文件

        Args:
            file_path: 文件路径

        Returns:
            书签列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse(content)

    def get_urls(self) -> List[str]:
        """获取所有URL列表"""
        return [b.url for b in self.bookmarks]

    def get_by_folder(self, folder: str) -> List[BookmarkItem]:
        """按文件夹筛选书签"""
        return [b for b in self.bookmarks if b.folder and folder in b.folder]
