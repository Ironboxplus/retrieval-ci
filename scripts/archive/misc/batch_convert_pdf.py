"""
Batch PDF to Markdown Converter

Batch convert downloaded PDF papers to Markdown format.
Features:
- Resume from breakpoint (skip already converted files)
- Detailed logging to data/logs/conversion_{date}.log
- Progress bar display
- Error handling with continuation
- Generate conversion report to data/papers/metadata/conversion_report.json

Usage:
    python scripts/batch_convert_pdf.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from pipeline.paper_crawler.config import Config, settings
from pipeline.paper_crawler.converters import (
    PDFToMarkdown,
    ConversionResult,
    ConversionStatus,
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"conversion_{date_str}.log"

    # Create logger
    logger = logging.getLogger("batch_converter")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - info level only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_pending_files(pdf_dir: Path, md_dir: Path) -> List[Path]:
    """Get list of PDF files that haven't been converted yet.

    Args:
        pdf_dir: Directory containing PDF files
        md_dir: Directory containing Markdown files

    Returns:
        List of PDF file paths that need conversion
    """
    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))

    # Get already converted file stems
    converted_stems = {f.stem for f in md_dir.glob("*.md")}

    # Filter out already converted files
    pending_files = [f for f in pdf_files if f.stem not in converted_stems]

    return sorted(pending_files)


def generate_report(
    results: List[Dict[str, Any]],
    output_path: Path,
    total_pdf: int,
    total_md_before: int,
    start_time: datetime,
    end_time: datetime,
) -> Dict[str, Any]:
    """Generate conversion report.

    Args:
        results: List of conversion result dictionaries
        output_path: Path to save the report
        total_pdf: Total PDF count
        total_md_before: MD count before conversion
        start_time: Conversion start time
        end_time: Conversion end time

    Returns:
        Report dictionary
    """
    success_count = sum(
        1 for r in results
        if r["status"] in ("success", "fallback")
    )
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")

    failed_files = [
        {"file": r["pdf_path"], "error": r.get("error", "Unknown error")}
        for r in results
        if r["status"] == "failed"
    ]

    report = {
        "conversion_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
        },
        "statistics": {
            "total_pdf_files": total_pdf,
            "md_before_conversion": total_md_before,
            "attempted": len(results),
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "success_rate": f"{success_count / max(len(results), 1) * 100:.1f}%",
        },
        "failed_files": failed_files,
        "all_results": results,
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    """Main entry point for batch conversion."""
    # Initialize configuration
    config = Config()

    pdf_dir = config.pdf_dir
    md_dir = config.markdown_dir
    log_dir = config.log_dir
    metadata_dir = config.metadata_dir

    # Setup logging
    logger = setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("PDF to Markdown Batch Converter Started")
    logger.info("=" * 60)

    # Count files
    all_pdf_files = list(pdf_dir.glob("*.pdf"))
    all_md_files = list(md_dir.glob("*.md"))
    total_pdf = len(all_pdf_files)
    total_md_before = len(all_md_files)

    logger.info(f"PDF directory: {pdf_dir}")
    logger.info(f"Markdown directory: {md_dir}")
    logger.info(f"Total PDF files: {total_pdf}")
    logger.info(f"Already converted: {total_md_before}")

    # Get pending files
    pending_files = get_pending_files(pdf_dir, md_dir)
    pending_count = len(pending_files)

    logger.info(f"Files to convert: {pending_count}")

    if pending_count == 0:
        logger.info("No files to convert. All PDFs have been converted.")
        return

    # Initialize converter
    logger.info("Initializing PDF converter...")
    converter = PDFToMarkdown(
        config=config,
        output_dir=md_dir,
        overwrite=False,
        save_images=True,
    )

    # Track results
    results: List[Dict[str, Any]] = []
    start_time = datetime.now()

    # Convert with progress bar
    logger.info("Starting conversion...")
    print()  # New line before progress bar

    success_count = 0
    failed_count = 0

    with tqdm(
        total=pending_count,
        desc="Converting PDFs",
        unit="file",
        ncols=100,
    ) as pbar:
        for pdf_path in pending_files:
            try:
                result = converter.convert_single_with_result(
                    pdf_path=pdf_path,
                    add_metadata=True,
                )

                result_dict = result.to_dict()
                results.append(result_dict)

                if result.status in (ConversionStatus.SUCCESS, ConversionStatus.FALLBACK):
                    success_count += 1
                    logger.debug(
                        f"SUCCESS: {pdf_path.name} -> {result.markdown_path.name} "
                        f"({result.elapsed_time:.2f}s, method={result.method.value})"
                    )
                else:
                    failed_count += 1
                    logger.error(
                        f"FAILED: {pdf_path.name} - {result.error}"
                    )

            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                logger.error(f"EXCEPTION: {pdf_path.name} - {error_msg}")
                results.append({
                    "pdf_path": str(pdf_path),
                    "markdown_path": None,
                    "status": "failed",
                    "method": "unknown",
                    "error": error_msg,
                    "elapsed_time": 0,
                })

            # Update progress bar
            pbar.set_postfix({
                "success": success_count,
                "failed": failed_count,
            })
            pbar.update(1)

    end_time = datetime.now()

    # Generate report
    report_path = metadata_dir / "conversion_report.json"
    report = generate_report(
        results=results,
        output_path=report_path,
        total_pdf=total_pdf,
        total_md_before=total_md_before,
        start_time=start_time,
        end_time=end_time,
    )

    # Print summary
    print()
    logger.info("=" * 60)
    logger.info("Conversion Complete")
    logger.info("=" * 60)
    logger.info(f"Duration: {report['conversion_info']['duration_seconds']:.2f} seconds")
    logger.info(f"Attempted: {report['statistics']['attempted']}")
    logger.info(f"Success: {report['statistics']['success']}")
    logger.info(f"Failed: {report['statistics']['failed']}")
    logger.info(f"Success Rate: {report['statistics']['success_rate']}")
    logger.info(f"Report saved to: {report_path}")

    # Log failed files
    if report["failed_files"]:
        logger.info("-" * 60)
        logger.info("Failed files:")
        for item in report["failed_files"]:
            logger.info(f"  - {Path(item['file']).name}: {item['error']}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
