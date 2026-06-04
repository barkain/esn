"""Tests for elite and frontier archive managers."""

from esn.core.archives import EliteArchive, FrontierArchive
from esn.core.enums import SearchMode
from esn.core.models import CandidateRecord


def _make_candidate(
    id: str = "c1",
    score: float | None = 1.0,
    success: bool | None = True,
    generation: int = 0,
) -> CandidateRecord:
    return CandidateRecord(
        id=id,
        generation=generation,
        search_mode=SearchMode.EXPLOIT,
        operator_name="test_op",
        object_hash="hash",
        score=score,
        success=success,
    )


# --- EliteArchive ---


class TestEliteArchive:
    def test_insert_valid_candidate(self):
        archive = EliteArchive(max_size=10)
        assert archive.insert(_make_candidate()) is True
        assert archive.size == 1

    def test_insert_below_threshold_rejected(self):
        archive = EliteArchive(max_size=10)
        assert archive.insert(_make_candidate(score=0.9), min_score=1.0) is False
        assert archive.size == 0

    def test_insert_invalid_success_false(self):
        archive = EliteArchive()
        assert archive.insert(_make_candidate(success=False)) is False
        assert archive.size == 0

    def test_insert_none_score_rejected(self):
        archive = EliteArchive()
        assert archive.insert(_make_candidate(score=None)) is False
        assert archive.size == 0

    def test_insert_success_none_rejected(self):
        archive = EliteArchive()
        assert archive.insert(_make_candidate(success=None)) is False
        assert archive.size == 0

    def test_eviction_on_full(self):
        archive = EliteArchive(max_size=3)
        archive.insert(_make_candidate(id="a", score=1.0))
        archive.insert(_make_candidate(id="b", score=2.0))
        archive.insert(_make_candidate(id="c", score=3.0))
        assert archive.size == 3
        # Insert higher score, should evict lowest (1.0)
        assert archive.insert(_make_candidate(id="d", score=4.0)) is True
        assert archive.size == 3
        ids = {c.id for c in archive.get_all()}
        assert "a" not in ids
        assert "d" in ids

    def test_eviction_rejects_lower_score(self):
        archive = EliteArchive(max_size=2)
        archive.insert(_make_candidate(id="a", score=5.0))
        archive.insert(_make_candidate(id="b", score=3.0))
        # Try to insert lower than min — should be rejected
        assert archive.insert(_make_candidate(id="c", score=2.0)) is False
        assert archive.size == 2

    def test_get_best_returns_top_n(self):
        archive = EliteArchive()
        for i in range(5):
            archive.insert(_make_candidate(id=f"c{i}", score=float(i)))
        best = archive.get_best(3)
        assert len(best) == 3
        assert best[0].score == 4.0
        assert best[1].score == 3.0
        assert best[2].score == 2.0

    def test_get_best_empty(self):
        archive = EliteArchive()
        assert archive.get_best() == []

    def test_get_all_sorted(self):
        archive = EliteArchive()
        archive.insert(_make_candidate(id="a", score=1.0))
        archive.insert(_make_candidate(id="b", score=3.0))
        archive.insert(_make_candidate(id="c", score=2.0))
        all_c = archive.get_all()
        assert [c.score for c in all_c] == [3.0, 2.0, 1.0]

    def test_size_property(self):
        archive = EliteArchive()
        assert archive.size == 0
        archive.insert(_make_candidate(id="a"))
        assert archive.size == 1


# --- FrontierArchive ---


class TestFrontierArchive:
    def test_insert_high_novelty(self):
        archive = FrontierArchive()
        assert archive.insert(_make_candidate(), novelty=0.5) is True
        assert archive.size == 1

    def test_insert_high_repairability(self):
        archive = FrontierArchive()
        assert archive.insert(_make_candidate(), repairability=0.5) is True
        assert archive.size == 1

    def test_insert_both_low_rejected(self):
        archive = FrontierArchive(novelty_threshold=0.1, repairability_threshold=0.1)
        assert archive.insert(_make_candidate(), novelty=0.05, repairability=0.05) is False
        assert archive.size == 0

    def test_insert_invalid_candidate_accepted(self):
        """Frontier does NOT require success=True."""
        archive = FrontierArchive()
        c = _make_candidate(success=False, score=None)
        assert archive.insert(c, novelty=0.5) is True

    def test_eviction_removes_oldest_lowest_novelty(self):
        archive = FrontierArchive(max_size=3)
        archive.insert(_make_candidate(id="a"), novelty=0.1)
        archive.insert(_make_candidate(id="b"), novelty=0.5)
        archive.insert(_make_candidate(id="c"), novelty=0.1)
        # Full — inserting new should evict "a" (oldest with lowest novelty 0.1)
        archive.insert(_make_candidate(id="d"), novelty=0.8)
        assert archive.size == 3
        ids = {c.id for c in archive.get_all()}
        assert "a" not in ids
        assert "d" in ids

    def test_get_repair_candidates(self):
        archive = FrontierArchive()
        archive.insert(_make_candidate(id="a"), repairability=0.3)
        archive.insert(_make_candidate(id="b"), repairability=0.9)
        archive.insert(_make_candidate(id="c"), repairability=0.6)
        top = archive.get_repair_candidates(2)
        assert len(top) == 2
        assert top[0].id == "b"
        assert top[1].id == "c"

    def test_get_novel_candidates(self):
        archive = FrontierArchive()
        archive.insert(_make_candidate(id="a"), novelty=0.2)
        archive.insert(_make_candidate(id="b"), novelty=0.8)
        archive.insert(_make_candidate(id="c"), novelty=0.5)
        top = archive.get_novel_candidates(2)
        assert len(top) == 2
        assert top[0].id == "b"
        assert top[1].id == "c"

    def test_empty_archive(self):
        archive = FrontierArchive()
        assert archive.get_all() == []
        assert archive.get_repair_candidates() == []
        assert archive.get_novel_candidates() == []

    def test_size_property(self):
        archive = FrontierArchive()
        assert archive.size == 0
        archive.insert(_make_candidate(), novelty=0.5)
        assert archive.size == 1

    def test_distinct_object_hashes(self):
        archive = FrontierArchive()
        archive.insert(_make_candidate(id="a"), novelty=0.5)
        archive.insert(_make_candidate(id="b"), novelty=0.6)
        c = _make_candidate(id="c")
        c.object_hash = "other"
        archive.insert(c, novelty=0.7)
        assert archive.distinct_object_hashes == 2

    def test_max_size_respected(self):
        archive = FrontierArchive(max_size=5)
        for i in range(10):
            archive.insert(_make_candidate(id=f"c{i}"), novelty=float(i) / 10 + 0.1)
        assert archive.size == 5
