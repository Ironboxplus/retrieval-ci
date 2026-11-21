"""Simple PDF downloader script for paper_crawler."""

import json
import httpx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys

def download_pdf(paper_data, output_dir, timeout=60):
    """Download a single PDF."""
    paper_id = paper_data.get("id", "unknown")
    pdf_url = paper_data.get("pdf_url")
    title = paper_data.get("title", "Unknown")[:50]

    if not pdf_url:
        return paper_id, False, "No PDF URL"

    # Determine save path
    conference = paper_data.get("conference", "unknown")
    if isinstance(conference, dict):
        conference = conference.get("value", "unknown")
    year = paper_data.get("year", "unknown")

    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper_id)[:200]
    save_dir = output_dir / str(conference).lower() / str(year)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{safe_id}.pdf"

    # Skip if already exists
    if save_path.exists() and save_path.stat().st_size > 1000:
        return paper_id, True, "Skipped (exists)"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(pdf_url, headers=headers)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(response.content)

            # Verify PDF header
            with open(save_path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    save_path.unlink()
                    return paper_id, False, "Invalid PDF format"

            return paper_id, True, f"Downloaded {len(response.content)} bytes"

    except Exception as e:
        return paper_id, False, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download PDFs for papers")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file with papers")
    parser.add_argument("-o", "--output", default="E:/JETB/pythons/sf/retrival/data/papers/pdf", help="Output directory")
    parser.add_argument("-n", "--concurrent", type=int, default=5, help="Number of concurrent downloads")
    parser.add_argument("-l", "--limit", type=int, default=0, help="Limit number of papers (0 = all)")
    args = parser.parse_args()

    # Load papers
    with open(args.input, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers")

    # Filter papers with PDF URLs
    papers_with_pdf = [p for p in papers if p.get("pdf_url")]
    print(f"{len(papers_with_pdf)} papers have PDF URLs")

    if args.limit > 0:
        papers_with_pdf = papers_with_pdf[:args.limit]
        print(f"Limited to {len(papers_with_pdf)} papers")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download with ThreadPoolExecutor
    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(download_pdf, paper, output_dir): paper
            for paper in papers_with_pdf
        }

        for i, future in enumerate(as_completed(futures), 1):
            paper_id, ok, msg = future.result()
            if ok:
                if "Skipped" in msg:
                    skipped += 1
                else:
                    success += 1
                    print(f"[{i}/{len(futures)}] OK: {paper_id} - {msg}")
            else:
                failed += 1
                print(f"[{i}/{len(futures)}] FAIL: {paper_id} - {msg}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()
