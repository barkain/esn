"""Tests for the family tracker."""

from esn.engine.family_tracker import FamilyTracker


def test_record_single_family():
    tracker = FamilyTracker()
    tracker.record("ring", 2.0, True, "concentric rings")
    stats = tracker.get_stats("ring")
    assert stats is not None
    assert stats.best_score == 2.0
    assert stats.attempt_count == 1
    assert stats.success_count == 1
    assert stats.failure_count == 0
    assert stats.representative_summary == "concentric rings"


def test_record_failure():
    tracker = FamilyTracker()
    tracker.record("grid", 0.0, False)
    stats = tracker.get_stats("grid")
    assert stats is not None
    assert stats.failure_count == 1
    assert stats.success_count == 0


def test_multiple_families_sorted_by_best():
    tracker = FamilyTracker()
    tracker.record("ring", 2.0, True)
    tracker.record("hex", 2.5, True)
    tracker.record("grid", 1.5, True)
    summaries = tracker.get_summary()
    assert len(summaries) == 3
    assert summaries[0].startswith("hex:")
    assert summaries[1].startswith("ring:")
    assert summaries[2].startswith("grid:")


def test_plateau_detection():
    tracker = FamilyTracker()
    tracker.record("ring", 2.0, True)
    tracker.record("ring", 1.8, True)
    tracker.record("ring", 1.9, True)
    stats = tracker.get_stats("ring")
    assert stats is not None
    assert stats.plateau_gens == 2  # two gens without improvement


def test_plateau_resets_on_improvement():
    tracker = FamilyTracker()
    tracker.record("ring", 2.0, True)
    tracker.record("ring", 1.8, True)
    tracker.record("ring", 2.1, True)
    stats = tracker.get_stats("ring")
    assert stats is not None
    assert stats.plateau_gens == 0


def test_recent_scores_capped_at_5():
    tracker = FamilyTracker()
    for i in range(8):
        tracker.record("ring", float(i), True)
    stats = tracker.get_stats("ring")
    assert stats is not None
    assert len(stats.recent_scores) == 5
    assert stats.recent_scores == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_get_summary_format():
    tracker = FamilyTracker()
    tracker.record("ring", 2.12, True, "concentric placement")
    summaries = tracker.get_summary()
    assert len(summaries) == 1
    assert "ring:" in summaries[0]
    assert "best=2.1200" in summaries[0]
    assert "1 attempts (1 ok, 0 fail)" in summaries[0]
    assert "improving" in summaries[0]
    assert "last: 2.12" in summaries[0]


def test_to_dict_from_dict_roundtrip():
    tracker = FamilyTracker()
    tracker.record("ring", 2.0, True, "rings")
    tracker.record("hex", 2.5, True, "hexagonal")
    tracker.record("ring", 1.8, False)

    data = tracker.to_dict()
    restored = FamilyTracker.from_dict(data)

    assert restored.get_stats("ring") is not None
    assert restored.get_stats("hex") is not None
    ring = restored.get_stats("ring")
    assert ring.best_score == 2.0
    assert ring.attempt_count == 2
    assert ring.failure_count == 1
    hex_stats = restored.get_stats("hex")
    assert hex_stats.best_score == 2.5


def test_get_stats_unknown_family():
    tracker = FamilyTracker()
    assert tracker.get_stats("nonexistent") is None
