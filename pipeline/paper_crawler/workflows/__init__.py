"""
Workflows - LangGraph 工作流模块

定义论文爬取、处理的自动化工作流。

功能:
- PaperPipeline: 论文处理管道类
- PaperState: 工作流状态类型
- run_pipeline: 便捷运行函数
- app: 预编译的工作流实例
"""

from pipeline.paper_crawler.workflows.paper_pipeline import (
    # 主类
    PaperPipeline,
    PipelineConfig,
    PipelineContext,

    # 状态类型
    PaperState,
    SinglePaperState,

    # 节点函数
    get_paper_list,
    download_papers,
    convert_to_markdown,
    build_index,
    save_metadata,

    # 构建函数
    build_paper_pipeline,
    build_parallel_pipeline,

    # 便捷函数
    run_pipeline,
    get_app,
    get_context,
    set_context,

    # 预编译的工作流实例
    app,
)

__all__ = [
    # 主类
    "PaperPipeline",
    "PipelineConfig",
    "PipelineContext",

    # 状态类型
    "PaperState",
    "SinglePaperState",

    # 节点函数
    "get_paper_list",
    "download_papers",
    "convert_to_markdown",
    "build_index",
    "save_metadata",

    # 构建函数
    "build_paper_pipeline",
    "build_parallel_pipeline",

    # 便捷函数
    "run_pipeline",
    "get_app",
    "get_context",
    "set_context",

    # 预编译实例
    "app",
]
