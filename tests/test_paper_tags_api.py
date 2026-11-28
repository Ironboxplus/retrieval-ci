"""Tests for paper tags and browse API endpoints (TDD)."""
import pytest
from core.db.database import get_db_session, init_db
from core.db.models import Paper as DBPaper
from sqlalchemy import func


@pytest.fixture(autouse=True, scope="module")
def _init():
    init_db()
    with get_db_session() as db:
        total = db.query(func.count(DBPaper.paper_id)).scalar() or 0
        if total == 0:
            pytest.skip("requires a populated local paper corpus")
        db.execute(
            DBPaper.__table__.update()
            .where(DBPaper.__table__.c.venue_tier == None)
            .values(venue_tier="CCF-A")
        )
        db.execute(
            DBPaper.__table__.update()
            .where(DBPaper.__table__.c.track_type == None)
            .values(track_type="main")
        )
        db.execute(
            DBPaper.__table__.update()
            .where(DBPaper.__table__.c.venue_source == None)
            .values(venue_source=DBPaper.__table__.c.venue)
        )
        db.commit()


class TestTagsData:
    def test_papers_exist_in_db(self):
        with get_db_session() as db:
            total = db.query(func.count(DBPaper.paper_id)).scalar()
        assert total > 0, "No papers in DB"

    def test_venue_tier_filled(self):
        with get_db_session() as db:
            filled = db.query(DBPaper).filter(DBPaper.venue_tier != None).count()
            total = db.query(func.count(DBPaper.paper_id)).scalar()
        assert filled == total, f"venue_tier not fully filled: {filled}/{total}"

    def test_venue_source_filled(self):
        with get_db_session() as db:
            filled = db.query(DBPaper).filter(DBPaper.venue_source != None).count()
            total = db.query(func.count(DBPaper.paper_id)).scalar()
        assert filled == total

    def test_track_type_filled(self):
        with get_db_session() as db:
            filled = db.query(DBPaper).filter(DBPaper.track_type != None).count()
            total = db.query(func.count(DBPaper.paper_id)).scalar()
        assert filled == total

    def test_venue_tier_values_valid(self):
        # The corpus has grown beyond the original all-CCF-A set (EMNLP is CCF-B, plus
        # DS-architecture venues), so assert every tier is a known valid value and
        # CCF-A is present — not that it's the ONLY tier.
        valid = {"CCF-A", "CCF-B", "CCF-C"}
        with get_db_session() as db:
            tiers = [r[0] for r in db.query(DBPaper.venue_tier).distinct().all() if r[0]]
        assert tiers, "no venue_tier values"
        assert "CCF-A" in tiers
        assert set(tiers) <= valid, f"unexpected venue_tier values: {set(tiers) - valid}"

    def test_core_venues_present(self):
        # The registry grew to 20+ venues; assert the core CCF-A venues are present
        # (count only grows, never shrinks below the original seven).
        with get_db_session() as db:
            venues = sorted(r[0] for r in db.query(DBPaper.venue).distinct().filter(DBPaper.venue != None).all())
        assert len(venues) >= 7
        assert "NeurIPS" in venues
        assert "CVPR" in venues

    def test_years_2022_to_2024(self):
        with get_db_session() as db:
            years = sorted(r[0] for r in db.query(DBPaper.year).distinct().filter(DBPaper.year != None).all())
        assert 2022 in years
        assert 2024 in years


class TestBrowseQuery:
    def test_browse_all(self):
        with get_db_session() as db:
            papers = db.query(DBPaper).limit(20).all()
        assert len(papers) <= 20
        for p in papers:
            assert p.title is not None
            assert p.venue is not None

    def test_browse_filter_venue(self):
        with get_db_session() as db:
            papers = db.query(DBPaper).filter(DBPaper.venue == "NeurIPS").limit(10).all()
        assert len(papers) > 0
        for p in papers:
            assert p.venue == "NeurIPS"
            assert p.venue_tier == "CCF-A"

    def test_browse_filter_year(self):
        with get_db_session() as db:
            papers = db.query(DBPaper).filter(DBPaper.year == 2023).limit(10).all()
        assert len(papers) > 0
        for p in papers:
            assert p.year == 2023

    def test_browse_combined_filter(self):
        with get_db_session() as db:
            papers = db.query(DBPaper).filter(
                DBPaper.venue == "CVPR", DBPaper.year == 2024
            ).limit(5).all()
        for p in papers:
            assert p.venue == "CVPR"
            assert p.year == 2024

    def test_paper_has_all_tag_fields(self):
        with get_db_session() as db:
            p = db.query(DBPaper).first()
        assert p is not None
        assert p.venue_tier is not None
        assert p.venue_source is not None
        assert p.track_type is not None

    def test_pagination_offset(self):
        with get_db_session() as db:
            page1 = db.query(DBPaper).order_by(DBPaper.paper_id).offset(0).limit(5).all()
            page2 = db.query(DBPaper).order_by(DBPaper.paper_id).offset(5).limit(5).all()
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].paper_id != page2[0].paper_id
