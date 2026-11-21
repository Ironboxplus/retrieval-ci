"""
Google Scholar 引用数获取示例

本脚本展示如何使用 GoogleScholarScraper 获取论文引用数。

支持三种获取方式:
1. requests + BeautifulSoup (主要方式)
2. scholarly 库 (备选方式)
3. Chrome MCP (反爬备用方式)

使用方法:
    python example_google_scholar.py
    python example_google_scholar.py --method scholarly
    python example_google_scholar.py --batch

依赖安装:
    pip install requests beautifulsoup4
    pip install scholarly  # 可选
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.paper_crawler.scrapers import (
    GoogleScholarScraper,
    GoogleScholarConfig,
    FetchMethod,
    get_citation_count,
    get_citation_count_batch,
)


def example_single_paper():
    """示例1: 获取单篇论文引用数"""
    print("=" * 60)
    print("示例1: 获取单篇论文引用数")
    print("=" * 60)

    # 使用便捷函数
    title = "Attention is All You Need"
    print(f"\n标题: {title}")

    count = get_citation_count(title)
    print(f"引用数: {count}")


def example_with_scraper():
    """示例2: 使用 GoogleScholarScraper 类"""
    print("\n" + "=" * 60)
    print("示例2: 使用 GoogleScholarScraper 类")
    print("=" * 60)

    # 自定义配置
    config = GoogleScholarConfig(
        min_delay=4.0,
        max_delay=10.0,
        similarity_threshold=0.6,
        preferred_method=FetchMethod.REQUESTS,
    )

    with GoogleScholarScraper(config=config) as scraper:
        titles = [
            "Attention is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ]

        for title in titles:
            print(f"\n标题: {title[:60]}...")
            count = scraper.get_citation_count(title)
            print(f"引用数: {count}")

        # 打印统计信息
        print("\n统计信息:")
        for key, value in scraper.get_stats().items():
            print(f"  {key}: {value}")


def example_batch():
    """示例3: 批量获取引用数"""
    print("\n" + "=" * 60)
    print("示例3: 批量获取引用数")
    print("=" * 60)

    titles = [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "Language Models are Few-Shot Learners",
        "Deep Residual Learning for Image Recognition",
        "ImageNet Classification with Deep Convolutional Neural Networks",
    ]

    print(f"\n待获取论文数: {len(titles)}")

    # 使用便捷函数
    results = get_citation_count_batch(titles)

    print("\n结果:")
    print("-" * 60)
    for title, count in results.items():
        short_title = title[:50] + "..." if len(title) > 50 else title
        print(f"  {short_title}: {count}")


def example_scholarly():
    """示例4: 使用 scholarly 库"""
    print("\n" + "=" * 60)
    print("示例4: 使用 scholarly 库 (备选方式)")
    print("=" * 60)

    config = GoogleScholarConfig(
        preferred_method=FetchMethod.SCHOLARLY,
    )

    try:
        with GoogleScholarScraper(config=config) as scraper:
            title = "Attention is All You Need"
            print(f"\n标题: {title}")

            count = scraper.get_citation_count(title)
            print(f"引用数: {count}")
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 请确保已安装 scholarly 库: pip install scholarly")


def example_with_proxy():
    """示例5: 使用代理"""
    print("\n" + "=" * 60)
    print("示例5: 使用代理")
    print("=" * 60)

    config = GoogleScholarConfig(
        proxy="http://127.0.0.1:7890",  # 替换为你的代理地址
        min_delay=2.0,
        max_delay=5.0,
    )

    print(f"\n代理: {config.proxy}")
    print("注意: 请确保代理可用，否则会连接失败")

    # 这里只是展示配置，不实际运行
    print("(跳过实际请求，仅展示配置)")


def example_chrome_mcp():
    """示例6: Chrome MCP 使用说明 (用于反爬)"""
    print("\n" + "=" * 60)
    print("示例6: Chrome MCP 使用说明")
    print("=" * 60)

    print("""
Chrome MCP 适用于 Google Scholar 反爬严重时的备用方式。
需要在 Claude Code 环境中使用，配合 Chrome DevTools MCP 工具。

使用步骤:
1. 确保 Chrome 浏览器已打开并连接 MCP
2. 调用 get_citation_count_with_chrome_mcp() 方法

示例代码:
```python
from pipeline.paper_crawler.scrapers import GoogleScholarScraper

scraper = GoogleScholarScraper()

# Chrome MCP 工具字典 (由 Claude Code 提供)
chrome_tools = {
    "navigate_page": mcp__chrome_devtools__navigate_page,
    "take_snapshot": mcp__chrome_devtools__take_snapshot,
    "wait_for": mcp__chrome_devtools__wait_for,
}

title = "Attention is All You Need"
count = scraper.get_citation_count_with_chrome_mcp(title, chrome_tools)
print(f"引用数: {count}")
```

注意:
- Chrome MCP 模拟真人浏览器操作，可以绕过大多数反爬措施
- 但速度较慢，适合少量论文查询
- 大批量查询仍建议使用 OpenAlex API
""")


def example_integration_with_openalex():
    """示例7: 与 OpenAlex 集成使用"""
    print("\n" + "=" * 60)
    print("示例7: 与 OpenAlex 集成使用")
    print("=" * 60)

    print("""
可以通过配置 citation_source 选择引用数来源:

.env 配置:
```
GOOGLE_SCHOLAR_CITATION_SOURCE=openalex     # 仅使用 OpenAlex (默认)
GOOGLE_SCHOLAR_CITATION_SOURCE=google_scholar  # 仅使用 Google Scholar
GOOGLE_SCHOLAR_CITATION_SOURCE=both         # 两者都尝试，取较大值
```

代码示例:
```python
from pipeline.paper_crawler.config import get_settings

settings = get_settings()

if settings.google_scholar.citation_source == "google_scholar":
    # 使用 Google Scholar
    from pipeline.paper_crawler.scrapers import get_citation_count
    count = get_citation_count(title)
elif settings.google_scholar.citation_source == "openalex":
    # 使用 OpenAlex (参考 update_citations.py)
    count = get_citation_count_openalex(title, session)
else:
    # 两者都尝试
    gs_count = get_citation_count(title)
    oa_count = get_citation_count_openalex(title, session)
    count = max(gs_count, oa_count)
```
""")


def main():
    parser = argparse.ArgumentParser(description="Google Scholar 引用数获取示例")
    parser.add_argument(
        "--method",
        choices=["requests", "scholarly", "all"],
        default="requests",
        help="获取方式: requests (默认), scholarly, all (所有示例)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="运行批量获取示例"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="仅打印说明，不实际运行爬虫"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Google Scholar 引用数获取示例")
    print("=" * 60)

    if args.no_run:
        # 仅打印说明
        example_with_proxy()
        example_chrome_mcp()
        example_integration_with_openalex()
        return

    if args.method == "all":
        example_single_paper()
        example_with_scraper()
        if args.batch:
            example_batch()
        example_scholarly()
        example_with_proxy()
        example_chrome_mcp()
        example_integration_with_openalex()
    elif args.method == "scholarly":
        example_scholarly()
    elif args.batch:
        example_batch()
    else:
        example_single_paper()
        example_with_scraper()

    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
