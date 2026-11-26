"""
PDF to Markdown Converter - PDF 转 Markdown 转换器

使用 marker 库（Meta 开源）进行高质量的 PDF 到 Markdown 转换。
支持：
- 保留 LaTeX 公式
- 保留表格结构
- 保留图片引用
- 批量转换
- 失败回退到 pymupdf 简单提取

依赖安装:
    pip install marker-pdf pymupdf

存储路径:
    data/papers/markdown/{conference}/{year}/{paper_id}.md
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re

# 延迟导入，避免启动时加载大型模型
_marker_available = None
_pymupdf_available = None


def _check_marker_available() -> bool:
    """检查 marker 库是否可用"""
    global _marker_available
    if _marker_available is None:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            _marker_available = True
        except ImportError:
            _marker_available = False
    return _marker_available


def _check_pymupdf_available() -> bool:
    """检查 pymupdf 库是否可用"""
    global _pymupdf_available
    if _pymupdf_available is None:
        try:
            import pymupdf
            _pymupdf_available = True
        except ImportError:
            try:
                import fitz
                _pymupdf_available = True
            except ImportError:
                _pymupdf_available = False
    return _pymupdf_available


class ConversionMethod(Enum):
    """转换方法枚举"""
    MARKER = "marker"
    PYMUPDF = "pymupdf"
    UNKNOWN = "unknown"


class ConversionStatus(Enum):
    """转换状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"  # 主方法失败，使用回退方法成功
    SKIPPED = "skipped"  # 跳过（如文件已存在）


@dataclass
class ConversionResult:
    """单个文件的转换结果"""
    pdf_path: Path
    markdown_path: Optional[Path] = None
    status: ConversionStatus = ConversionStatus.FAILED
    method: ConversionMethod = ConversionMethod.UNKNOWN
    error: Optional[str] = None
    elapsed_time: float = 0.0

    # 转换统计信息
    page_count: int = 0
    has_equations: bool = False
    has_tables: bool = False
    has_images: bool = False
    image_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pdf_path": str(self.pdf_path),
            "markdown_path": str(self.markdown_path) if self.markdown_path else None,
            "status": self.status.value,
            "method": self.method.value,
            "error": self.error,
            "elapsed_time": self.elapsed_time,
            "page_count": self.page_count,
            "has_equations": self.has_equations,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "image_count": self.image_count,
        }


@dataclass
class BatchConversionResult:
    """批量转换结果"""
    results: List[ConversionResult] = field(default_factory=list)
    total_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_elapsed_time: float = 0.0

    def add_result(self, result: ConversionResult):
        """添加单个结果"""
        self.results.append(result)
        self.total_count += 1
        self.total_elapsed_time += result.elapsed_time

        if result.status == ConversionStatus.SUCCESS:
            self.success_count += 1
        elif result.status == ConversionStatus.FALLBACK:
            self.fallback_count += 1
        elif result.status == ConversionStatus.FAILED:
            self.failed_count += 1
        elif result.status == ConversionStatus.SKIPPED:
            self.skipped_count += 1

    @property
    def success_rate(self) -> float:
        """成功率（包含回退成功）"""
        if self.total_count == 0:
            return 0.0
        return (self.success_count + self.fallback_count) / self.total_count * 100

    def summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        return {
            "total": self.total_count,
            "success": self.success_count,
            "fallback": self.fallback_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_time": f"{self.total_elapsed_time:.2f}s",
            "avg_time": f"{self.total_elapsed_time / max(self.total_count, 1):.2f}s",
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


class PDFToMarkdown:
    """PDF 转 Markdown 转换器

    使用 marker 库进行高质量转换，失败时回退到 pymupdf 简单提取。

    使用示例:
        >>> from pipeline.paper_crawler.config import Config
        >>> config = Config()
        >>> converter = PDFToMarkdown(config)
        >>>
        >>> # 单个文件转换
        >>> result_path = converter.convert_single(Path("paper.pdf"))
        >>>
        >>> # 批量转换
        >>> pdf_files = [Path("paper1.pdf"), Path("paper2.pdf")]
        >>> results = converter.convert_batch(pdf_files)
        >>> print(results.summary())
    """

    # marker 模型缓存（避免重复加载）
    _marker_model_dict = None
    _marker_converter = None

    def __init__(
        self,
        config,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
        save_images: bool = True,
        image_format: str = "png",
        use_llm: bool = False,
    ):
        """初始化转换器

        Args:
            config: 配置对象，需包含 markdown_dir 属性
            output_dir: 输出目录（优先于 config.markdown_dir）
            overwrite: 是否覆盖已存在的文件
            save_images: 是否保存提取的图片
            image_format: 图片保存格式 (png, jpg, webp)
            use_llm: 是否使用 LLM 增强转换（需要 API Key）
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path(config.markdown_dir)
        self.overwrite = overwrite
        self.save_images = save_images
        self.image_format = image_format
        self.use_llm = use_llm

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 导入日志模块
        try:
            from ..utils.logger import get_logger
            self.logger = get_logger("converter")
        except ImportError:
            import logging
            self.logger = logging.getLogger(__name__)

        # 检查依赖可用性
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖库的可用性"""
        marker_ok = _check_marker_available()
        pymupdf_ok = _check_pymupdf_available()

        if not marker_ok and not pymupdf_ok:
            self.logger.warning(
                "marker 和 pymupdf 均未安装，请运行: pip install marker-pdf pymupdf"
            )
        elif not marker_ok:
            self.logger.warning(
                "marker 未安装，将只使用 pymupdf 进行简单提取。"
                "建议安装: pip install marker-pdf"
            )
        elif not pymupdf_ok:
            self.logger.warning(
                "pymupdf 未安装，如果 marker 失败将无法回退。"
                "建议安装: pip install pymupdf"
            )

    def _get_output_path(
        self,
        pdf_path: Path,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
    ) -> Path:
        """根据 PDF 路径生成输出 Markdown 路径

        Args:
            pdf_path: PDF 文件路径
            conference: 会议名称（可选）
            year: 年份（可选）
            paper_id: 论文 ID（可选）

        Returns:
            Markdown 文件输出路径
        """
        # 如果提供了完整信息，使用结构化路径
        if conference and year and paper_id:
            output_path = self.output_dir / conference / str(year) / f"{paper_id}.md"
        elif conference and year:
            # 使用 PDF 文件名作为 paper_id
            paper_id = pdf_path.stem
            output_path = self.output_dir / conference / str(year) / f"{paper_id}.md"
        else:
            # 尝试从 PDF 路径推断结构
            # 假设 PDF 路径结构: .../pdf/{conference}/{year}/{paper_id}.pdf
            try:
                parts = pdf_path.parts
                pdf_idx = None
                for i, part in enumerate(parts):
                    if part.lower() == "pdf":
                        pdf_idx = i
                        break

                if pdf_idx is not None and len(parts) > pdf_idx + 3:
                    conference = parts[pdf_idx + 1]
                    year = parts[pdf_idx + 2]
                    output_path = self.output_dir / conference / year / f"{pdf_path.stem}.md"
                else:
                    # 使用扁平结构
                    output_path = self.output_dir / f"{pdf_path.stem}.md"
            except Exception:
                output_path = self.output_dir / f"{pdf_path.stem}.md"

        # 确保父目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def _get_images_dir(self, markdown_path: Path) -> Path:
        """获取图片存储目录

        Args:
            markdown_path: Markdown 文件路径

        Returns:
            图片存储目录路径
        """
        # 图片存储在同级的 images 目录下
        images_dir = markdown_path.parent / "images" / markdown_path.stem
        images_dir.mkdir(parents=True, exist_ok=True)
        return images_dir

    def _init_marker(self) -> bool:
        """初始化 marker 转换器（延迟加载，复用模型）

        Returns:
            是否初始化成功
        """
        if not _check_marker_available():
            return False

        # 如果已经初始化，直接返回
        if PDFToMarkdown._marker_converter is not None:
            return True

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.config.parser import ConfigParser

            self.logger.info("正在加载 marker 模型（首次加载可能需要较长时间）...")

            # 创建配置
            config = {
                "output_format": "markdown",
                "paginate_output": False,  # 不分页
            }

            # 如果需要 LLM 增强
            if self.use_llm:
                try:
                    openai_key = self.config._settings.api_keys.openai_api_key
                    if openai_key:
                        config["use_llm"] = True
                except Exception:
                    pass

            config_parser = ConfigParser(config)

            # 创建模型字典（这是最耗时的步骤）
            PDFToMarkdown._marker_model_dict = create_model_dict()

            # 创建转换器
            PDFToMarkdown._marker_converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=PDFToMarkdown._marker_model_dict,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
            )

            self.logger.info("marker 模型加载完成")
            return True

        except Exception as e:
            self.logger.error(f"初始化 marker 失败: {e}")
            PDFToMarkdown._marker_converter = None
            return False

    def _use_marker(self, pdf_path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
        """使用 marker 进行转换

        Args:
            pdf_path: PDF 文件路径

        Returns:
            (markdown 内容, 元数据字典)
        """
        if not self._init_marker():
            raise RuntimeError("marker 初始化失败")

        from marker.output import text_from_rendered

        # 执行转换
        rendered = PDFToMarkdown._marker_converter(str(pdf_path))

        # 提取文本和图片
        text, _, images = text_from_rendered(rendered)

        # 收集元数据
        metadata = {
            "page_count": getattr(rendered, "page_count", 0) if hasattr(rendered, "page_count") else 0,
            "images": images,
            "has_equations": bool(re.search(r'\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)', text, re.DOTALL)),
            "has_tables": bool(re.search(r'\|.*\|.*\|', text)),
        }

        return text, metadata

    def _use_pymupdf_fallback(self, pdf_path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
        """使用 pymupdf 进行回退提取

        Args:
            pdf_path: PDF 文件路径

        Returns:
            (markdown 内容, 元数据字典)
        """
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        doc = pymupdf.open(str(pdf_path))

        markdown_parts = []
        page_count = len(doc)
        image_count = 0

        for page_num, page in enumerate(doc, 1):
            # 添加页面分隔注释
            if page_num > 1:
                markdown_parts.append(f"\n\n<!-- Page {page_num} -->\n\n")

            # 提取文本
            text = page.get_text("text")

            # 基本的 Markdown 格式化
            # 尝试识别标题（通常是较短的独立行，可能全大写）
            lines = text.split('\n')
            formatted_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    formatted_lines.append("")
                    continue

                # 简单的标题检测
                if len(line) < 100 and line.isupper():
                    formatted_lines.append(f"## {line.title()}")
                else:
                    formatted_lines.append(line)

            markdown_parts.append('\n'.join(formatted_lines))

            # 统计图片数量
            image_list = page.get_images()
            image_count += len(image_list)

        doc.close()

        content = '\n'.join(markdown_parts)

        # 简单的后处理
        # 合并断行（论文中常见的列排版导致的断行）
        content = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', content)

        # 移除多余空行
        content = re.sub(r'\n{3,}', '\n\n', content)

        metadata = {
            "page_count": page_count,
            "images": {},
            "image_count": image_count,
            "has_equations": False,  # pymupdf 简单提取无法保留公式
            "has_tables": False,  # pymupdf 简单提取无法保留表格结构
        }

        return content, metadata

    def _save_images(
        self,
        images: Dict[str, Any],
        images_dir: Path,
        markdown_content: str,
    ) -> str:
        """保存图片并更新 Markdown 中的图片引用

        Args:
            images: 图片字典 {filename: image_data}
            images_dir: 图片保存目录
            markdown_content: Markdown 内容

        Returns:
            更新后的 Markdown 内容
        """
        if not images or not self.save_images:
            return markdown_content

        for img_name, img_data in images.items():
            try:
                # 确定图片路径
                img_ext = Path(img_name).suffix or f".{self.image_format}"
                img_path = images_dir / f"{Path(img_name).stem}{img_ext}"

                # 保存图片
                if hasattr(img_data, 'save'):
                    # PIL Image 对象
                    img_data.save(str(img_path))
                elif isinstance(img_data, bytes):
                    # 二进制数据
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                else:
                    self.logger.warning(f"未知的图片数据类型: {type(img_data)}")
                    continue

                # 更新 Markdown 中的图片引用
                # 使用相对路径
                relative_path = f"images/{images_dir.name}/{img_path.name}"
                markdown_content = markdown_content.replace(
                    f"![{img_name}]",
                    f"![{img_name}]({relative_path})"
                )

            except Exception as e:
                self.logger.warning(f"保存图片失败 {img_name}: {e}")

        return markdown_content

    def _add_metadata_header(
        self,
        content: str,
        pdf_path: Path,
        method: ConversionMethod,
        metadata: Dict[str, Any],
    ) -> str:
        """添加 YAML front matter 元数据头

        Args:
            content: Markdown 内容
            pdf_path: PDF 文件路径
            method: 使用的转换方法
            metadata: 元数据

        Returns:
            带有元数据头的 Markdown 内容
        """
        front_matter = f"""---
source_pdf: {pdf_path.name}
conversion_method: {method.value}
conversion_time: {datetime.now().isoformat()}
page_count: {metadata.get('page_count', 0)}
has_equations: {metadata.get('has_equations', False)}
has_tables: {metadata.get('has_tables', False)}
---

"""
        return front_matter + content

    def convert_single(
        self,
        pdf_path: Path,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
        add_metadata: bool = True,
    ) -> Optional[Path]:
        """转换单个 PDF 文件

        Args:
            pdf_path: PDF 文件路径
            conference: 会议名称（用于组织输出目录）
            year: 年份
            paper_id: 论文 ID
            add_metadata: 是否添加元数据头

        Returns:
            成功时返回 Markdown 文件路径，失败返回 None
        """
        result = self.convert_single_with_result(
            pdf_path=pdf_path,
            conference=conference,
            year=year,
            paper_id=paper_id,
            add_metadata=add_metadata,
        )

        if result.status in (ConversionStatus.SUCCESS, ConversionStatus.FALLBACK):
            return result.markdown_path
        return None

    def convert_single_with_result(
        self,
        pdf_path: Path,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
        add_metadata: bool = True,
    ) -> ConversionResult:
        """转换单个 PDF 文件并返回详细结果

        Args:
            pdf_path: PDF 文件路径
            conference: 会议名称
            year: 年份
            paper_id: 论文 ID
            add_metadata: 是否添加元数据头

        Returns:
            ConversionResult 对象
        """
        pdf_path = Path(pdf_path)
        start_time = datetime.now()

        result = ConversionResult(pdf_path=pdf_path)

        # 检查 PDF 文件是否存在
        if not pdf_path.exists():
            result.error = f"PDF 文件不存在: {pdf_path}"
            self.logger.error(result.error)
            return result

        if not pdf_path.suffix.lower() == '.pdf':
            result.error = f"不是 PDF 文件: {pdf_path}"
            self.logger.error(result.error)
            return result

        # 确定输出路径
        output_path = self._get_output_path(pdf_path, conference, year, paper_id)
        result.markdown_path = output_path

        # 检查是否需要跳过
        if output_path.exists() and not self.overwrite:
            result.status = ConversionStatus.SKIPPED
            result.elapsed_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"跳过已存在的文件: {output_path}")
            return result

        content = None
        metadata = {}

        # 尝试使用 marker 转换
        if _check_marker_available():
            try:
                self.logger.info(f"使用 marker 转换: {pdf_path.name}")
                content, metadata = self._use_marker(pdf_path)
                result.method = ConversionMethod.MARKER
                result.status = ConversionStatus.SUCCESS
                self.logger.info(f"marker 转换成功: {pdf_path.name}")
            except Exception as e:
                self.logger.warning(f"marker 转换失败: {e}")
                content = None

        # 如果 marker 失败，尝试 pymupdf 回退
        if content is None and _check_pymupdf_available():
            try:
                self.logger.info(f"使用 pymupdf 回退: {pdf_path.name}")
                content, metadata = self._use_pymupdf_fallback(pdf_path)
                result.method = ConversionMethod.PYMUPDF

                # 如果之前尝试过 marker，标记为回退
                if result.status != ConversionStatus.SUCCESS:
                    result.status = ConversionStatus.FALLBACK
                else:
                    result.status = ConversionStatus.SUCCESS

                self.logger.info(f"pymupdf 提取成功: {pdf_path.name}")
            except Exception as e:
                self.logger.error(f"pymupdf 提取失败: {e}")
                result.error = str(e)

        # 如果仍然没有内容，标记为失败
        if content is None:
            result.status = ConversionStatus.FAILED
            if not result.error:
                result.error = "所有转换方法均失败"
            result.elapsed_time = (datetime.now() - start_time).total_seconds()
            return result

        # 保存图片
        if self.save_images and metadata.get("images"):
            images_dir = self._get_images_dir(output_path)
            content = self._save_images(metadata["images"], images_dir, content)

        # 添加元数据头
        if add_metadata:
            content = self._add_metadata_header(content, pdf_path, result.method, metadata)

        # 写入文件
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Markdown 保存成功: {output_path}")
        except Exception as e:
            result.status = ConversionStatus.FAILED
            result.error = f"保存文件失败: {e}"
            self.logger.error(result.error)
            result.elapsed_time = (datetime.now() - start_time).total_seconds()
            return result

        # 填充结果统计
        result.page_count = metadata.get("page_count", 0)
        result.has_equations = metadata.get("has_equations", False)
        result.has_tables = metadata.get("has_tables", False)
        result.image_count = len(metadata.get("images", {}))
        result.has_images = result.image_count > 0
        result.elapsed_time = (datetime.now() - start_time).total_seconds()

        return result

    def convert_batch(
        self,
        pdf_paths: List[Path],
        conference: Optional[str] = None,
        year: Optional[int] = None,
        add_metadata: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> BatchConversionResult:
        """批量转换 PDF 文件

        Args:
            pdf_paths: PDF 文件路径列表
            conference: 会议名称（可选，用于所有文件）
            year: 年份（可选，用于所有文件）
            add_metadata: 是否添加元数据头
            progress_callback: 进度回调函数 (current, total, result)

        Returns:
            BatchConversionResult 对象
        """
        batch_result = BatchConversionResult()
        total = len(pdf_paths)

        self.logger.info(f"开始批量转换 {total} 个 PDF 文件")

        for i, pdf_path in enumerate(pdf_paths, 1):
            pdf_path = Path(pdf_path)

            result = self.convert_single_with_result(
                pdf_path=pdf_path,
                conference=conference,
                year=year,
                paper_id=pdf_path.stem,
                add_metadata=add_metadata,
            )

            batch_result.add_result(result)

            # 调用进度回调
            if progress_callback:
                try:
                    progress_callback(i, total, result)
                except Exception as e:
                    self.logger.warning(f"进度回调失败: {e}")

            # 日志进度
            if i % 10 == 0 or i == total:
                self.logger.info(
                    f"转换进度: {i}/{total} ({i/total*100:.1f}%) | "
                    f"成功: {batch_result.success_count} | "
                    f"回退: {batch_result.fallback_count} | "
                    f"失败: {batch_result.failed_count}"
                )

        self.logger.info(
            f"批量转换完成 | {batch_result.summary()}"
        )

        return batch_result

    def convert_directory(
        self,
        input_dir: Path,
        recursive: bool = True,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        add_metadata: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> BatchConversionResult:
        """转换目录下的所有 PDF 文件

        Args:
            input_dir: 输入目录
            recursive: 是否递归处理子目录
            conference: 会议名称
            year: 年份
            add_metadata: 是否添加元数据头
            progress_callback: 进度回调函数

        Returns:
            BatchConversionResult 对象
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            self.logger.error(f"目录不存在: {input_dir}")
            return BatchConversionResult()

        # 收集所有 PDF 文件
        if recursive:
            pdf_files = list(input_dir.rglob("*.pdf"))
        else:
            pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"目录中没有找到 PDF 文件: {input_dir}")
            return BatchConversionResult()

        self.logger.info(f"在 {input_dir} 中找到 {len(pdf_files)} 个 PDF 文件")

        return self.convert_batch(
            pdf_paths=pdf_files,
            conference=conference,
            year=year,
            add_metadata=add_metadata,
            progress_callback=progress_callback,
        )


# 向后兼容的类别名
class PDFToMarkdownConverter(PDFToMarkdown):
    """向后兼容的类名别名

    保持与原有代码的兼容性。
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        use_marker: bool = True,
    ):
        """初始化转换器（向后兼容接口）

        Args:
            output_dir: 输出目录
            use_marker: 是否使用marker库
        """
        # 导入配置
        try:
            from ..config import config
            self._compat_config = config
        except ImportError:
            # 创建一个简单的配置对象
            class SimpleConfig:
                def __init__(self, output_dir):
                    self.markdown_dir = output_dir or Path("./output")
            self._compat_config = SimpleConfig(output_dir)

        super().__init__(
            config=self._compat_config,
            output_dir=output_dir,
            overwrite=False,
        )
        self._use_marker_flag = use_marker

    def convert(self, pdf_path: Path, output_name: Optional[str] = None) -> Optional[Path]:
        """转换PDF为Markdown（向后兼容接口）

        Args:
            pdf_path: PDF文件路径
            output_name: 输出文件名（不含扩展名）

        Returns:
            Markdown文件路径或None
        """
        return self.convert_single(
            pdf_path=pdf_path,
            paper_id=output_name,
        )

    def convert_paper(self, paper, output_dir: Optional[Path] = None) -> Optional[Path]:
        """转换论文PDF为Markdown（向后兼容接口）

        Args:
            paper: 论文对象（必须已下载PDF）
            output_dir: 输出目录（可选，覆盖默认配置）

        Returns:
            Markdown文件路径或None
        """
        if not paper.pdf_path:
            self.logger.error(f"Paper has no PDF path: {paper.title}")
            return None

        # If output_dir is specified, temporarily override self.output_dir
        original_output_dir = self.output_dir
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(paper.pdf_path)

        # Get conference and year from paper object
        conference = None
        year = None

        if hasattr(paper, 'conference') and paper.conference:
            if hasattr(paper.conference, 'value'):
                conference = paper.conference.value
            else:
                conference = str(paper.conference)

        if hasattr(paper, 'year') and paper.year:
            year = paper.year
        elif hasattr(paper, 'metadata') and paper.metadata and hasattr(paper.metadata, 'year'):
            year = paper.metadata.year

        try:
            result = self.convert_single(
                pdf_path=pdf_path,
                conference=conference,
                year=year,
                paper_id=paper.id,
            )
            return result
        finally:
            # Restore original output_dir
            if output_dir:
                self.output_dir = original_output_dir


# 便捷函数
def convert_pdf_to_markdown(
    pdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    use_marker: bool = True,
) -> Optional[str]:
    """便捷函数：转换单个 PDF 到 Markdown

    Args:
        pdf_path: PDF 文件路径
        output_path: 输出路径（可选）
        use_marker: 是否优先使用 marker

    Returns:
        Markdown 内容，失败返回 None
    """
    pdf_path = Path(pdf_path)

    content = None

    # 尝试 marker
    if use_marker and _check_marker_available():
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(str(pdf_path))
            content, _, _ = text_from_rendered(rendered)
        except Exception:
            content = None

    # 回退到 pymupdf
    if content is None and _check_pymupdf_available():
        try:
            try:
                import pymupdf
            except ImportError:
                import fitz as pymupdf

            doc = pymupdf.open(str(pdf_path))
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            doc.close()
            content = '\n\n'.join(parts)
        except Exception:
            content = None

    # 保存到文件
    if content and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    return content


# 导出
__all__ = [
    "PDFToMarkdown",
    "PDFToMarkdownConverter",  # 向后兼容
    "ConversionResult",
    "BatchConversionResult",
    "ConversionMethod",
    "ConversionStatus",
    "convert_pdf_to_markdown",
]
