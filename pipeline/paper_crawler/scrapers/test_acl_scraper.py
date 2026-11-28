"""
ACL Scraper Test Script

Tests the ACL Scraper implementation by fetching papers from ACL 2024.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.paper_crawler.scrapers.acl_scraper import ACLScraper


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_get_paper_list():
    """Test fetching paper list from ACL 2024."""
    print("\n" + "="*60)
    print("TEST 1: Fetching ACL 2024 Long Papers (limit 5)")
    print("="*60)

    scraper = ACLScraper()

    try:
        papers = await scraper.get_paper_list(
            conference="ACL",
            year=2024,
            track="long",
            limit=5
        )

        print(f"\nFetched {len(papers)} papers:\n")

        for i, paper in enumerate(papers, 1):
            print(f"--- Paper {i} ---")
            print(f"ID: {paper.id}")
            print(f"Title: {paper.title[:80]}..." if len(paper.title) > 80 else f"Title: {paper.title}")
            print(f"Authors: {', '.join(a.name for a in paper.authors[:3])}" + ("..." if len(paper.authors) > 3 else ""))
            print(f"Year: {paper.year}")
            print(f"Conference: {paper.conference.value}")
            print(f"Type: {paper.paper_type.value}")
            print(f"PDF URL: {paper.pdf_url}")
            print()

        return len(papers) > 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_get_paper_details():
    """Test fetching single paper details."""
    print("\n" + "="*60)
    print("TEST 2: Fetching Single Paper Details")
    print("="*60)

    scraper = ACLScraper()

    paper_id = "2024.acl-long.1"
    print(f"\nFetching paper: {paper_id}")

    try:
        paper = await scraper.get_paper_details(paper_id)

        print(f"\n--- Paper Details ---")
        print(f"ID: {paper.id}")
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(a.name for a in paper.authors)}")
        print(f"Year: {paper.year}")
        print(f"Conference: {paper.conference.value}")
        print(f"Type: {paper.paper_type.value}")
        print(f"PDF URL: {paper.pdf_url}")
        print(f"DOI: {paper.doi}")
        print(f"Abstract: {paper.abstract[:200]}..." if len(paper.abstract) > 200 else f"Abstract: {paper.abstract}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_get_pdf_url():
    """Test getting PDF URL."""
    print("\n" + "="*60)
    print("TEST 3: Getting PDF URL")
    print("="*60)

    scraper = ACLScraper()

    paper_id = "2024.acl-long.1"
    print(f"\nGetting PDF URL for: {paper_id}")

    try:
        paper = await scraper.get_paper_details(paper_id)
        pdf_url = await scraper.get_pdf_url(paper)

        print(f"PDF URL: {pdf_url}")

        # Verify URL format
        assert "aclanthology.org" in pdf_url
        assert paper_id in pdf_url
        assert ".pdf" in pdf_url

        print("PDF URL format verified!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_search():
    """Test paper search."""
    print("\n" + "="*60)
    print("TEST 4: Searching Papers")
    print("="*60)

    scraper = ACLScraper()

    query = "large language model"
    print(f"\nSearching for: '{query}'")

    try:
        papers = await scraper.search(query, limit=5)

        print(f"\nFound {len(papers)} papers:\n")

        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.title[:60]}..." if len(paper.title) > 60 else f"{i}. {paper.title}")
            print(f"   ID: {paper.id}, Year: {paper.year}")

        return len(papers) > 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


async def test_venue_url():
    """Test venue URL generation."""
    print("\n" + "="*60)
    print("TEST 5: Venue URL Generation")
    print("="*60)

    scraper = ACLScraper()

    venues = [
        ("ACL", 2024),
        ("EMNLP", 2024),
        ("NAACL", 2024),
    ]

    print()
    for conf, year in venues:
        url = scraper.get_venue_url(conf, year)
        print(f"{conf} {year}: {url}")

    return True


async def test_paper_id_parsing():
    """Test paper ID parsing."""
    print("\n" + "="*60)
    print("TEST 6: Paper ID Parsing")
    print("="*60)

    scraper = ACLScraper()

    paper_ids = [
        "2024.acl-long.1",
        "2024.emnlp-main.100",
        "2023.findings-acl.50",
    ]

    print()
    for paper_id in paper_ids:
        year, venue, track, number = scraper.parse_paper_id(paper_id)
        print(f"{paper_id} -> year={year}, venue={venue}, track={track}, number={number}")

    return True


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ACL SCRAPER TEST SUITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = {}

    # Run tests
    results["get_paper_list"] = await test_get_paper_list()
    results["get_paper_details"] = await test_get_paper_details()
    results["get_pdf_url"] = await test_get_pdf_url()
    results["search"] = await test_search()
    results["venue_url"] = await test_venue_url()
    results["paper_id_parsing"] = await test_paper_id_parsing()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
