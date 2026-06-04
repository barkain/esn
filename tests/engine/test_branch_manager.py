"""BranchManager unit tests.

Covers two-flow accounting, lineage routing, failure routing (no centroid
fallback), retirement triggers, eviction, and persistence round-trip.
"""
# ruff: noqa: S101
# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

from esn.engine.branch_manager import (
    FAILURE_LIMIT,
    MAX_LIVE_BRANCHES,
    MIN_PLATEAU_FOR_RECOMBINE,
    STAGNATION_LIMIT,
    BranchManager,
    BranchRole,
    build_aspect_signature,
)


def _emb(*vals: float) -> list[float]:
    return [float(v) for v in vals]


class TestTwoFlowAccounting:
    def test_seed_creates_first_branch(self):
        m = BranchManager()
        a = m.register_attempt(
            parent_id=None,
            child_id="seed",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="grid",
            generation=0,
        )
        assert a.branch_id is not None
        assert a.created_new is True
        assert len(m.live_branches()) == 1
        b = m.live_branches()[0]
        assert b.num_attempts == 1
        assert b.num_successes == 1
        assert b.num_improvements == 1
        assert b.depth == 1

    def test_successful_descendant_extends_branch(self):
        m = BranchManager()
        a1 = m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="grid",
            generation=0,
        )
        a2 = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.5,
            family="grid",
            aspect_signature="grid",
            generation=1,
        )
        assert a1.branch_id == a2.branch_id
        b = m.branches[a1.branch_id]
        assert b.num_attempts == 2
        assert b.num_successes == 2
        assert b.num_improvements == 2
        assert b.best_score == 1.5
        assert b.depth == 2

    def test_failure_increments_attempts_not_successes(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="grid",
            generation=0,
        )
        m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=False,
            score=0.0,
            family="grid",
            aspect_signature="grid",
            generation=1,
        )
        b = m.live_branches()[0]
        assert b.num_attempts == 2
        assert b.num_successes == 1
        assert b.consecutive_failures == 1

    def test_success_resets_consecutive_failures(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        for i in range(3):
            m.register_attempt(
                parent_id="c1",
                child_id=f"f{i}",
                success=False,
                score=0.0,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
        m.register_attempt(
            parent_id="c1",
            child_id="ok",
            success=True,
            score=2.0,
            family="g",
            aspect_signature="g",
            generation=5,
        )
        b = m.live_branches()[0]
        assert b.consecutive_failures == 0
        assert b.num_attempts == 5
        assert b.num_successes == 2


class TestFailureRouting:
    def test_orphan_failure_does_not_charge_any_branch(self):
        m = BranchManager()
        # Live branch exists, but failed child has UNKNOWN parent
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        a = m.register_attempt(
            parent_id="ghost",
            child_id="ghost_child",
            success=False,
            score=0.0,
            family="g",
            aspect_signature="g",
            generation=1,
        )
        assert a.branch_id is None
        assert a.is_orphan is True
        assert m.orphan_failures == 1
        # The live branch must be untouched
        b = m.live_branches()[0]
        assert b.num_attempts == 1
        assert b.consecutive_failures == 0

    def test_failure_charges_retired_parent_branch(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        # Force retirement
        for i in range(FAILURE_LIMIT):
            m.register_attempt(
                parent_id="c1",
                child_id=f"f{i}",
                success=False,
                score=0.0,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
        b = list(m.branches.values())[0]
        assert not b.alive
        # Now a late failure from c1's lineage hits the retired record
        a = m.register_attempt(
            parent_id="c1",
            child_id="late",
            success=False,
            score=0.0,
            family="g",
            aspect_signature="g",
            generation=20,
        )
        assert a.branch_id == b.id
        # 1 seed success + FAILURE_LIMIT failures + 1 late failure
        assert b.num_attempts == FAILURE_LIMIT + 2


class TestRetirement:
    def test_stagnation_retires_branch(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        # Successful but non-improving children push stagnation up
        prev = "c1"
        for i in range(STAGNATION_LIMIT + 2):
            cid = f"c{i + 2}"
            m.register_attempt(
                parent_id=prev,
                child_id=cid,
                success=True,
                score=0.5,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
            prev = cid
        b = list(m.branches.values())[0]
        assert b.alive is False
        assert b.retired_reason == "stagnation"

    def test_consecutive_failures_retire(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        for i in range(FAILURE_LIMIT):
            m.register_attempt(
                parent_id="c1",
                child_id=f"f{i}",
                success=False,
                score=0.0,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
        b = list(m.branches.values())[0]
        assert b.alive is False
        assert b.retired_reason == "consecutive_failures"

    def test_split_path_enforces_stagnation_retirement(self):
        """Regression: splits charge an attempt but used to bypass _maybe_retire.

        A seed branch that only ever throws off semantic splits (no
        continuations) must still honor STAGNATION_LIMIT. Before the fix
        the split path refreshed neither stagnation nor _maybe_retire, so
        a fork-only branch lived forever despite no local improvement.
        """
        m = BranchManager()
        # Seed branch at gen 0
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=10.0,
            family="fam_a",
            aspect_signature="asp_a",
            generation=0,
            embedding=_emb(1.0, 0.0, 0.0),
        )
        seed = list(m.branches.values())[0]
        seed_id = seed.id
        # Flood the seed with cross-family splits so _should_split fires
        # every time. Each child has a different family -> new branch is
        # created via the split path, and the seed's num_attempts is
        # charged without _extend_branch running.
        for i in range(STAGNATION_LIMIT + 2):
            m.register_attempt(
                parent_id="c1",
                child_id=f"split_{i}",
                success=True,
                score=5.0,  # below seed.best_score, not an improvement
                family=f"fam_fork_{i}",  # family_diff -> carves out the score floor
                aspect_signature=f"asp_fork_{i}",
                generation=1 + i,
                embedding=_emb(0.0, 1.0, 0.0),  # far from seed centroid
            )
        seed_after = m.branches[seed_id]
        assert seed_after.alive is False, (
            "seed branch must retire on STAGNATION_LIMIT even when it only "
            "throws off splits (never extends)"
        )
        assert seed_after.retired_reason == "stagnation"

    def test_merge_path_enforces_retirement(self):
        """Regression: nearest-centroid merge used to bypass _maybe_retire.

        A branch that only grows through the no-lineage fallback (orphan
        children merged into its centroid) must still honor FAILURE_LIMIT.
        Before the fix the merge path returned early, so failures
        accumulated but never retired the branch via this path.
        """
        m = BranchManager()
        # Seed a branch with a distinctive centroid
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=10.0,
            family="g",
            aspect_signature="g",
            generation=0,
            embedding=_emb(1.0, 0.0, 0.0),
        )
        seed_id = list(m.branches.values())[0].id
        # Push consecutive failures against the seed parent to set up
        # retirement eligibility. These go through the failure path, not
        # the merge path, but they bump consecutive_failures on the seed.
        for i in range(FAILURE_LIMIT - 1):
            m.register_attempt(
                parent_id="c1",
                child_id=f"f{i}",
                success=False,
                score=0.0,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
        # Verify we're right at the threshold but still alive
        assert m.branches[seed_id].alive is True
        assert m.branches[seed_id].consecutive_failures == FAILURE_LIMIT - 1
        # Now land a successful *orphan* child (unknown parent) whose
        # centroid is close enough to merge. _extend_branch resets
        # consecutive_failures to 0, so the merge path must retire based
        # on pre-merge stagnation or not at all — simpler invariant: drive
        # pure stagnation via merges instead.
        # Reset: use a fresh manager with a stagnation-based scenario.
        m2 = BranchManager()
        m2.register_attempt(
            parent_id=None,
            child_id="root",
            success=True,
            score=10.0,
            family="g",
            aspect_signature="g",
            generation=0,
            embedding=_emb(1.0, 0.0, 0.0),
        )
        seed2_id = list(m2.branches.values())[0].id
        # Flood with successful orphan children that merge into the seed
        # via nearest-centroid, each non-improving (score below best).
        # _extend_branch bumps stagnation = generation - last_improved_gen.
        for i in range(STAGNATION_LIMIT + 2):
            m2.register_attempt(
                parent_id=f"ghost_{i}",  # unknown parent -> no-lineage path
                child_id=f"orphan_{i}",
                success=True,
                score=5.0,  # non-improving, grows stagnation on merge
                family="g",
                aspect_signature="g",
                generation=1 + i,
                embedding=_emb(1.0, 0.0, 0.0),  # identical centroid -> merge
            )
        seed2_after = m2.branches[seed2_id]
        assert seed2_after.alive is False, (
            "branch must retire on stagnation even when all growth comes "
            "through the nearest-centroid merge path"
        )
        assert seed2_after.retired_reason == "stagnation"


class TestBatchDeferredRetirement:
    """Fix 3: inside begin_batch/end_batch, branch liveness is frozen.

    Regression target: seed=123 gen-19 — ring hit stagnation mid-batch,
    killed itself before the spiral candidate in the same batch could
    lineage-attach, and the spiral fell into the no-lineage fallback.
    """

    def _seed_branch(self, m: BranchManager) -> str:
        """Create one live branch at gen 0, return its child id."""
        a = m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        assert a.branch_id is not None
        return "c1"

    def test_stagnation_retirement_is_deferred(self):
        m = BranchManager()
        self._seed_branch(m)
        # Drive stagnation up to just before the limit (pre-batch).
        prev = "c1"
        for i in range(STAGNATION_LIMIT - 1):
            cid = f"pre{i}"
            m.register_attempt(
                parent_id=prev,
                child_id=cid,
                success=True,
                score=0.5,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
            prev = cid
        b = list(m.branches.values())[0]
        assert b.alive is True  # still alive — one away from the limit

        # One more non-improving success WOULD retire outside batch mode.
        m.begin_batch()
        try:
            m.register_attempt(
                parent_id=prev,
                child_id="in_batch_1",
                success=True,
                score=0.5,
                family="g",
                aspect_signature="g",
                generation=STAGNATION_LIMIT,
            )
            # Inside the batch, branch still looks alive to later slots.
            assert b.alive is True
            # A second slot in the same batch can still lineage-attach.
            a2 = m.register_attempt(
                parent_id="in_batch_1",
                child_id="in_batch_2",
                success=True,
                score=0.6,
                family="g",
                aspect_signature="g",
                generation=STAGNATION_LIMIT,
            )
            assert a2.branch_id == b.id
            assert not a2.created_new  # NOT falling into no_lineage fallback
        finally:
            m.end_batch(generation=STAGNATION_LIMIT)

        # After end_batch, the queued stagnation retirement is applied.
        assert b.alive is False
        assert b.retired_reason == "stagnation"

    def test_failure_retirement_is_deferred(self):
        m = BranchManager()
        self._seed_branch(m)
        # Prior failures: one short of FAILURE_LIMIT.
        for i in range(FAILURE_LIMIT - 1):
            m.register_attempt(
                parent_id="c1",
                child_id=f"pf{i}",
                success=False,
                score=0.0,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
        b = list(m.branches.values())[0]
        assert b.alive is True

        m.begin_batch()
        try:
            # This would trip FAILURE_LIMIT outside batch mode.
            m.register_attempt(
                parent_id="c1",
                child_id="f_batch",
                success=False,
                score=0.0,
                family="g",
                aspect_signature="g",
                generation=FAILURE_LIMIT,
            )
            assert b.alive is True  # retirement deferred
            # And a later successful slot in the same batch still finds it.
            a = m.register_attempt(
                parent_id="c1",
                child_id="s_batch",
                success=True,
                score=1.5,
                family="g",
                aspect_signature="g",
                generation=FAILURE_LIMIT,
            )
            assert a.branch_id == b.id
        finally:
            m.end_batch(generation=FAILURE_LIMIT)

        assert b.alive is False
        assert b.retired_reason == "consecutive_failures"

    def test_end_batch_is_safe_without_begin(self):
        # Calling end_batch without begin_batch must be a no-op.
        m = BranchManager()
        self._seed_branch(m)
        m.end_batch(generation=1)  # must not raise
        b = list(m.branches.values())[0]
        assert b.alive is True

    def test_non_batch_retires_immediately(self):
        # Default path (no begin_batch) still retires inline — regression
        # guard for the two existing TestRetirementPolicy tests' semantics.
        m = BranchManager()
        self._seed_branch(m)
        prev = "c1"
        for i in range(STAGNATION_LIMIT + 1):
            cid = f"c{i + 2}"
            m.register_attempt(
                parent_id=prev,
                child_id=cid,
                success=True,
                score=0.5,
                family="g",
                aspect_signature="g",
                generation=1 + i,
            )
            prev = cid
        b = list(m.branches.values())[0]
        assert b.alive is False
        assert b.retired_reason == "stagnation"


class TestEviction:
    def test_max_live_cap(self):
        m = BranchManager()
        # Create MAX_LIVE_BRANCHES + 2 root branches
        for i in range(MAX_LIVE_BRANCHES + 2):
            m.register_attempt(
                parent_id=None,
                child_id=f"root{i}",
                success=True,
                score=float(i + 1),
                family=f"f{i}",
                aspect_signature=f"f{i}",
                generation=i,
            )
        assert len(m.live_branches()) == MAX_LIVE_BRANCHES + 2
        evicted = m.evict_excess()
        assert len(evicted) == 2
        assert len(m.live_branches()) == MAX_LIVE_BRANCHES
        # Worst branches (lowest scores) should be the ones evicted
        for b_id in evicted:
            assert m.branches[b_id].retired_reason == "max_live_cap"


class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=2.0,
            family="g",
            aspect_signature="g",
            generation=1,
        )
        m.register_attempt(
            parent_id="ghost",
            child_id="orphan",
            success=False,
            score=0.0,
            family="g",
            aspect_signature="g",
            generation=2,
        )
        path = tmp_path / "branches.json"
        m.save_to(path)

        m2 = BranchManager.load_from(path)
        assert len(m2.live_branches()) == 1
        b = m2.live_branches()[0]
        assert b.num_successes == 2
        assert b.depth == 2
        assert m2.orphan_failures == 1
        # parent_to_branch index round-trips so future descendants resolve
        a = m2.register_attempt(
            parent_id="c2",
            child_id="c3",
            success=True,
            score=3.0,
            family="g",
            aspect_signature="g",
            generation=3,
        )
        assert a.branch_id == b.id


class TestAspectSignature:
    def test_basic_signature(self):
        sig = build_aspect_signature(code="def solve(n):\n    return []")
        assert "family=straight-line" in sig
        assert "features=" in sig
        assert "cfhash=" in sig

    def test_signature_distinguishes_recursive_from_iterative(self):
        rec = build_aspect_signature(
            code="def solve(n):\n    if n <= 0:\n        return 0\n    return solve(n - 1)"
        )
        it = build_aspect_signature(code="def solve(n):\n    for i in range(n):\n        print(i)")
        assert "family=recursive-tail" in rec
        assert "family=iterative-flat" in it
        assert rec != it


class TestSemanticSplit:
    def test_split_on_distant_embedding_with_different_aspect(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        # Same lineage, but very different family + orthogonal embedding
        a = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.5,
            family="ring",
            aspect_signature="family=ring",
            generation=1,
            embedding=_emb(0, 1, 0),
        )
        assert a.created_new is True
        assert len(m.live_branches()) == 2
        # Parent branch was charged an attempt for the split
        parent_branch = next(b for b in m.live_branches() if b.family == "grid")
        assert parent_branch.num_attempts == 2

    def test_no_split_when_same_family_embedding_close(self):
        """Same-family aspect-only forks still gated by embedding distance.

        Cross-family splits bypass the distance gate (see
        test_split_across_families_bypasses_distance_gate), but within a
        single family we still require the embedding to actually move so
        cosmetic cfhash drift doesn't spawn spurious branches.
        """
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid | cfhash=aaaa",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        a = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.5,
            family="grid",
            aspect_signature="family=grid | cfhash=bbbb",
            generation=1,
            embedding=_emb(0.99, 0.01, 0),
        )
        assert a.created_new is False
        assert len(m.live_branches()) == 1

    def test_split_across_families_bypasses_distance_gate(self):
        """Family-level structural divergence is a first-class split signal.

        Programs that share identical stdin/stdout boilerplate can cluster
        under the embedding: iterative-flat / iterative-nested /
        recursive-multi elites may all land in one branch because the
        BGE-small embedding never crosses SPLIT_DISTANCE (the shared
        boilerplate dominates the embedding). Cross-family splits must not
        depend on distance alone — the family jump itself is enough
        evidence of structural divergence.
        """
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="iterative-flat",
            aspect_signature="family=iterative-flat",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        # Family jumps iterative-flat -> recursive-multi, but embedding is
        # nearly identical (shared surface text). Under the old policy
        # this absorbs into the parent branch; under the new policy it
        # must fork.
        a = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.5,
            family="recursive-multi",
            aspect_signature="family=recursive-multi",
            generation=1,
            embedding=_emb(0.99, 0.01, 0),
        )
        assert a.created_new is True
        assert len(m.live_branches()) == 2

    def test_no_split_when_same_family_below_floor(self):
        """Same-family aspect-only forks must still clear the score floor."""
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=10.0,
            family="grid",
            aspect_signature="family=grid | motifs=a",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        # Same family, different aspect, distant embedding, but weak score
        a = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid | motifs=b",
            generation=1,
            embedding=_emb(0, 1, 0),
        )
        assert a.created_new is False

    def test_split_across_families_bypasses_floor(self):
        """Novel families emerge low-scoring and must be allowed to split.

        Regression for the seed=123 lineage-leak: a spiral child scoring
        far below the ring incumbent was being absorbed into ring (or
        saved only by mid-batch retirement). Cross-family splits should
        bypass the incumbent-score floor.
        """
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=10.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        # Different family, distant embedding, score way below floor — must split
        a = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.0,
            family="ring",
            aspect_signature="family=ring",
            generation=1,
            embedding=_emb(0, 1, 0),
        )
        assert a.created_new is True
        assert len(m.live_branches()) == 2
        # New branch should be the low-scoring ring fork
        ring = next(b for b in m.live_branches() if b.family == "ring")
        assert ring.best_score == 1.0


class TestNoLineageFallback:
    def test_merges_into_nearest_live_branch(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="root_a",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        # Orphan child whose embedding is near root_a's centroid
        a = m.register_attempt(
            parent_id="ghost",
            child_id="orphan_close",
            success=True,
            score=2.0,
            family="grid",
            aspect_signature="family=grid",
            generation=1,
            embedding=_emb(0.95, 0.05, 0),
        )
        assert a.created_new is False
        assert len(m.live_branches()) == 1
        b = m.live_branches()[0]
        assert b.depth == 2

    def test_creates_new_when_far(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="root_a",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        a = m.register_attempt(
            parent_id="ghost",
            child_id="orphan_far",
            success=True,
            score=2.0,
            family="hex",
            aspect_signature="family=hex",
            generation=1,
            embedding=_emb(0, 0, 1),
        )
        assert a.created_new is True
        assert len(m.live_branches()) == 2


class TestDominanceRetirement:
    def test_retires_dominated_same_family_close_centroid(self):
        m = BranchManager()
        # Weak branch
        m.register_attempt(
            parent_id=None,
            child_id="weak",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        # Stronger branch on a separate orthogonal embedding so the no-lineage
        # fallback creates a fresh branch instead of merging.
        m.register_attempt(
            parent_id=None,
            child_id="strong",
            success=True,
            score=2.0,
            family="grid",
            aspect_signature="family=grid",
            generation=5,
            embedding=_emb(0, 0, 1),
        )
        # Now move the strong branch's centroid close to the weak one to
        # exercise the dominance overlap rule deterministically.
        strong = next(b for b in m.branches.values() if b.root_parent_id == "strong")
        strong.centroid_embedding = _emb(0.97, 0.03, 0)
        retired = m.apply_dominance_retirement()
        assert len(retired) == 1
        weak = next(b for b in m.branches.values() if b.root_parent_id == "weak")
        assert not weak.alive
        assert weak.retired_reason == "dominance"

    def test_no_retirement_without_semantic_overlap(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="a",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        m.register_attempt(
            parent_id=None,
            child_id="b",
            success=True,
            score=2.0,
            family="hex",
            aspect_signature="family=hex",
            generation=5,
            embedding=_emb(0, 1, 0),
        )
        retired = m.apply_dominance_retirement()
        assert retired == []
        assert all(b.alive for b in m.branches.values())


class TestBranchRoleSampler:
    def _make_multi_branch_mgr(self) -> BranchManager:
        m = BranchManager()
        # Anchor: strong, improving
        m.register_attempt(
            parent_id=None,
            child_id="a1",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        m.register_attempt(
            parent_id="a1",
            child_id="a2",
            success=True,
            score=3.0,
            family="grid",
            aspect_signature="family=grid",
            generation=1,
            embedding=_emb(1, 0, 0),
        )
        # Orthogonal breakout: viable but stagnant
        m.register_attempt(
            parent_id=None,
            child_id="b1",
            success=True,
            score=1.8,
            family="hex",
            aspect_signature="family=hex",
            generation=0,
            embedding=_emb(0, 1, 0),
        )
        # Drive stagnation on b1 without improvement
        m.register_attempt(
            parent_id="b1",
            child_id="b2",
            success=True,
            score=1.2,
            family="hex",
            aspect_signature="family=hex",
            generation=5,
            embedding=_emb(0, 1, 0),
        )
        # Diversity: orthogonal and weaker
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.1,
            family="spiral",
            aspect_signature="family=spiral",
            generation=0,
            embedding=_emb(0, 0, 1),
        )
        return m

    def test_anchor_picks_global_best_branch(self):
        m = self._make_multi_branch_mgr()
        picks = m.sample_branch_parents([BranchRole.ANCHOR], global_best=3.0)
        assert len(picks) == 1
        role, branch = picks[0]
        assert role is BranchRole.ANCHOR
        assert branch.best_score == 3.0
        assert branch.family == "grid"

    def test_breakout_picks_stagnant_viable_branch(self):
        m = self._make_multi_branch_mgr()
        picks = m.sample_branch_parents([BranchRole.ANCHOR, BranchRole.BREAKOUT], global_best=3.0)
        # Breakout must be a non-anchor stagnant viable branch
        assert len(picks) == 2
        breakout_role, breakout_branch = picks[1]
        assert breakout_role is BranchRole.BREAKOUT
        assert breakout_branch.family == "hex"
        assert breakout_branch.stagnation > 0

    def test_diversity_picks_farthest_from_anchor(self):
        m = self._make_multi_branch_mgr()
        picks = m.sample_branch_parents([BranchRole.ANCHOR, BranchRole.DIVERSITY], global_best=3.0)
        assert len(picks) == 2
        div_role, div_branch = picks[1]
        assert div_role is BranchRole.DIVERSITY
        # Grid anchor centroid=(1,0,0); both hex (0,1,0) and spiral (0,0,1)
        # are at distance 1.0. Either is fine; just ensure not the anchor.
        assert div_branch.family != "grid"

    def test_dedup_across_roles(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="only",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        picks = m.sample_branch_parents(
            [
                BranchRole.ANCHOR,
                BranchRole.CONTINUATION,
                BranchRole.BREAKOUT,
                BranchRole.DIVERSITY,
            ],
            global_best=1.0,
        )
        # Only one branch exists → only one pick (anchor), others dedupe out.
        assert len(picks) == 1
        assert picks[0][0] is BranchRole.ANCHOR

    def test_render_report_section_has_live_table(self):
        m = self._make_multi_branch_mgr()
        lines = m.render_report_section()
        text = "\n".join(lines)
        assert "## Branch Preservation" in text
        assert "Live branches" in text
        assert "### Live" in text


class TestRecombinationGates:
    def _make_three_strong_branches(self, global_best: float = 3.0) -> BranchManager:
        m = BranchManager()
        # Three orthogonal, high-quality branches
        m.register_attempt(
            parent_id=None,
            child_id="a1",
            success=True,
            score=3.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        m.register_attempt(
            parent_id=None,
            child_id="b1",
            success=True,
            score=2.8,
            family="hex",
            aspect_signature="family=hex",
            generation=0,
            embedding=_emb(0, 1, 0),
        )
        m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=2.7,
            family="spiral",
            aspect_signature="family=spiral",
            generation=0,
            embedding=_emb(0, 0, 1),
        )
        return m

    def test_gate_insufficient_live_branches(self):
        m = BranchManager()
        m.register_attempt(
            parent_id=None,
            child_id="a",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
            embedding=_emb(1, 0),
        )
        pair = m.find_recombination_pair(global_best=1.0, global_stagnation=10, current_gen=20)
        assert pair is None
        assert m.live_branches()  # sanity

    def test_gate_plateau_not_met(self):
        m = self._make_three_strong_branches()
        pair = m.find_recombination_pair(global_best=3.0, global_stagnation=0, current_gen=10)
        assert pair is None

    def test_gate_spectral_undersampled(self):
        m = self._make_three_strong_branches()
        pair = m.find_recombination_pair(
            global_best=3.0,
            global_stagnation=MIN_PLATEAU_FOR_RECOMBINE,
            current_gen=10,
            spectral_undersampled=True,
        )
        assert pair is None

    def test_gate_cooldown_blocks(self):
        m = self._make_three_strong_branches()
        # First call passes
        pair1 = m.find_recombination_pair(
            global_best=3.0,
            global_stagnation=MIN_PLATEAU_FOR_RECOMBINE,
            current_gen=10,
        )
        assert pair1 is not None
        anchor, donor, _diag = pair1
        # Simulate a recombination event that sets the cooldown
        m.record_recombination_event(
            anchor_branch_id=anchor.id,
            donor_branch_id=donor.id,
            child_id="c2",
            success=True,
            score=3.5,
            created_new_branch=True,
            generation=10,
        )
        # Immediately attempting another must fail on cooldown
        pair2 = m.find_recombination_pair(
            global_best=3.5,
            global_stagnation=MIN_PLATEAU_FOR_RECOMBINE,
            current_gen=11,
        )
        assert pair2 is None

    def test_pair_returns_anchor_and_distant_donor(self):
        m = self._make_three_strong_branches()
        pair = m.find_recombination_pair(
            global_best=3.0,
            global_stagnation=MIN_PLATEAU_FOR_RECOMBINE,
            current_gen=10,
        )
        assert pair is not None
        anchor, donor, diag = pair
        assert anchor.best_score == 3.0
        assert donor.id != anchor.id
        assert diag["reason"] == "gates_passed"
        assert diag["diversity"] >= 0.4

    def test_quality_floor_rejects_weak_donors(self):
        m = BranchManager()
        # One strong branch, two very weak ones
        m.register_attempt(
            parent_id=None,
            child_id="strong",
            success=True,
            score=10.0,
            family="grid",
            aspect_signature="family=grid",
            generation=0,
            embedding=_emb(1, 0, 0),
        )
        m.register_attempt(
            parent_id=None,
            child_id="weak1",
            success=True,
            score=0.1,
            family="hex",
            aspect_signature="family=hex",
            generation=0,
            embedding=_emb(0, 1, 0),
        )
        m.register_attempt(
            parent_id=None,
            child_id="weak2",
            success=True,
            score=0.2,
            family="spiral",
            aspect_signature="family=spiral",
            generation=0,
            embedding=_emb(0, 0, 1),
        )
        pair = m.find_recombination_pair(
            global_best=10.0,
            global_stagnation=MIN_PLATEAU_FOR_RECOMBINE,
            current_gen=10,
        )
        assert pair is None

    def test_event_log_and_persistence_roundtrip(self, tmp_path):
        m = self._make_three_strong_branches()
        pair = m.find_recombination_pair(
            global_best=3.0,
            global_stagnation=MIN_PLATEAU_FOR_RECOMBINE,
            current_gen=10,
        )
        assert pair is not None
        anchor, donor, _ = pair
        m.record_recombination_event(
            anchor_branch_id=anchor.id,
            donor_branch_id=donor.id,
            child_id="rec_child",
            success=True,
            score=3.5,
            created_new_branch=True,
            generation=10,
        )
        path = tmp_path / "branches.json"
        m.save_to(path)
        m2 = BranchManager.load_from(path)
        assert len(m2.recombination_events) == 1
        assert m2.last_recombination_gen == 10


class TestRecombinationDonor:
    def test_donor_counters_separate(self):
        m = BranchManager()
        a = m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="g",
            aspect_signature="g",
            generation=0,
        )
        before = m.branches[a.branch_id]
        ns_before = before.num_successes
        m.register_recombination_donor(donor_branch_id=a.branch_id, success=True)
        after = m.branches[a.branch_id]
        assert after.num_donor_attempts == 1
        assert after.num_donor_successes == 1
        assert after.num_successes == ns_before  # primary lineage untouched


class TestIdentityRefreshOnImprovement:
    """Issue #8 (Option E): branch.family / branch.aspect_signature must
    refresh when an extending child improves the branch's best score, and
    stay stable when the extension does not improve it."""

    def test_identity_refreshes_when_extension_improves_best(self):
        m = BranchManager()
        a1 = m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=1.0,
            family="grid",
            aspect_signature="grid-v1",
            generation=0,
        )
        # Extend the same branch with an improving child that carries a
        # different family/aspect_signature. No embeddings -> _should_split
        # short-circuits to False, guaranteeing the extension path runs.
        a2 = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=2.0,
            family="rings",
            aspect_signature="rings-v1",
            generation=1,
        )
        assert a2.branch_id == a1.branch_id
        assert a2.created_new is False
        b = m.branches[a1.branch_id]
        assert b.best_score == 2.0
        assert b.family == "rings"
        assert b.aspect_signature == "rings-v1"

    def test_identity_stable_when_extension_does_not_improve(self):
        m = BranchManager()
        a1 = m.register_attempt(
            parent_id=None,
            child_id="c1",
            success=True,
            score=2.0,
            family="grid",
            aspect_signature="grid-v1",
            generation=0,
        )
        # Extend with a non-improving child carrying different identity.
        a2 = m.register_attempt(
            parent_id="c1",
            child_id="c2",
            success=True,
            score=1.0,
            family="rings",
            aspect_signature="rings-v1",
            generation=1,
        )
        assert a2.branch_id == a1.branch_id
        assert a2.created_new is False
        b = m.branches[a1.branch_id]
        assert b.best_score == 2.0
        assert b.family == "grid"
        assert b.aspect_signature == "grid-v1"
