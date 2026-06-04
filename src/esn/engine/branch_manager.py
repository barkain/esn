# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Branch preservation for ESN engine search.

This module provides lineage-anchored branch tracking with two-flow
accounting (attempt vs promotion), failure routing, retirement, and
persistence, plus semantic-split / centroid-based identity.

The implementation here is intentionally minimal: every
evaluated child is registered, branches are created on root mutations
or when the parent is unknown to a *live* branch, and retirement runs
on stagnation or repeated failure.
"""

from __future__ import annotations

import enum
import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class BranchRole(str, enum.Enum):
    """Roles used by branch-aware parent selection."""

    ANCHOR = "anchor"  # global-best branch — exploit latest parent
    CONTINUATION = "continuation"  # healthy improving branch (not the anchor)
    BREAKOUT = "breakout"  # stagnant-but-viable branch — push to escape
    DIVERSITY = "diversity"  # farthest-centroid branch from anchor


# --- Tuning constants (named for discoverability) -------------------------

STAGNATION_LIMIT = 10  # gens since last_improved_gen → retire
FAILURE_LIMIT = 8  # consecutive failures → retire
MAX_LIVE_BRANCHES = 8  # cap on live branches; eviction by composite
DOMINANCE_MARGIN_FRAC = 0.05  # |best_score| fraction for dominance check
DOMINANCE_DISTANCE = 0.10  # cosine distance threshold for semantic dominance
BREAKTHROUGH_RATIO = 1.02  # child / global_best ratio for breakthrough
BREAKTHROUGH_STAGNATION = 5  # parent branch stagnation gens for breakthrough

# Semantic identity
# Thresholds calibrated against deterministic AST feature vectors (30-dim,
# L2-normalised). Cross-family distances ~0.45-0.75, same-family with
# different features ~0.15-0.30, near-identical ~0.00-0.05.
SPLIT_DISTANCE = 0.15  # cosine distance to fork into a new branch
MERGE_DISTANCE = 0.30  # max cosine distance for no-lineage fallback merge
MIN_BRANCH_SCORE_FRAC = 0.5  # child.score / parent.best_score floor for split
CENTROID_EMA = 0.8  # weight on existing centroid in EMA update

# Recombination activation gates
MIN_LIVE_FOR_RECOMBINE = 3  # need at least N live branches
MIN_PLATEAU_FOR_RECOMBINE = 4  # gens of no global improvement before allowed
RECOMBINE_QUALITY_FLOOR = 0.7  # donor.best_score / global_best minimum
MIN_DIVERSITY = 0.15  # cosine distance floor between anchor/donor
RECOMBINE_COOLDOWN = 2  # gens between recombination attempts


def _cosine_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    """Return 1 - cos(a, b), or None if either is missing/zero."""
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 1e-24 or nb <= 1e-24:
        return None
    return max(0.0, 1.0 - dot / (math.sqrt(na) * math.sqrt(nb)))


def build_aspect_signature(*, code: str, family: str | None = None) -> str:
    """Domain-agnostic structural fingerprint for the branch identity tag.

    Format: ``family=X | features=a,b,c | cfhash=abcd1234``

    The ``family`` parameter is accepted for backwards-compat but ignored —
    the family is derived structurally from ``code`` via ``extract_ast_features``.
    """
    from esn.engine.ast_features import extract_ast_features  # type: ignore[import-not-found]

    result = extract_ast_features(code or "")
    parts = [
        f"family={result['family']}",
        "features=" + ",".join(result["features"]),
        f"cfhash={result['cfhash']}",
    ]
    return " | ".join(parts)


@dataclass
class BranchRecord:
    """Persistent state for one search line.

    The lineage-only mode keeps centroid_embedding
    as None and aspect_signature as the family snapshot only.
    """

    id: str
    root_parent_id: str
    latest_parent_id: str
    family: str
    aspect_signature: str = ""
    centroid_embedding: list[float] | None = None
    created_gen: int = 0
    last_improved_gen: int = 0
    best_score: float = 0.0
    latest_score: float = 0.0
    depth: int = 0
    num_attempts: int = 0
    num_successes: int = 0
    num_improvements: int = 0
    num_donor_attempts: int = 0
    num_donor_successes: int = 0
    consecutive_failures: int = 0
    stagnation: int = 0
    alive: bool = True
    retired_reason: str | None = None
    created_via: str = "split"
    recombination_parents: tuple[str, str] | None = None


@dataclass
class BranchAssignment:
    """Returned by register_attempt so the engine can stamp the child."""

    branch_id: str | None  # None when child went into the orphan bucket
    created_new: bool = False
    is_orphan: bool = False


class BranchManager:
    """Tracks live and retired branches.

    Lineage-only identity: a child inherits its parent's branch
    via `parent_to_branch` lookup. If the parent id is unknown to any live
    branch but the parent points to a *retired* branch (still in the
    persistence dict), failed children update the retired record only;
    successful children create a fresh branch with no lineage. Orphan
    failures (parent id never seen) go into a counter, never a branch.
    """

    def __init__(self) -> None:
        self._branches: dict[str, BranchRecord] = {}
        # parent program_id → owning branch id (for live AND retired)
        self._parent_to_branch: dict[str, str] = {}
        self._orphan_failures: int = 0
        self._orphan_successes: int = 0
        # recombination bookkeeping
        self._last_recombination_gen: int = -1 - RECOMBINE_COOLDOWN
        self._recombination_events: list[dict[str, Any]] = []
        # Fix 3: deferred retirement. When _batch_mode is True, _maybe_retire
        # collects pending retirements instead of flipping branches dead
        # immediately. The engine calls begin_batch() before processing a
        # batch's register_attempt calls and end_batch() after, so every slot
        # in the batch sees the same branch-liveness universe. Not serialized:
        # always starts False on a fresh or reloaded manager.
        self._batch_mode: bool = False
        self._pending_retirements: dict[str, str] = {}

    # --- Public API ------------------------------------------------------

    @property
    def branches(self) -> dict[str, BranchRecord]:
        return self._branches

    @property
    def orphan_failures(self) -> int:
        return self._orphan_failures

    @property
    def orphan_successes(self) -> int:
        return self._orphan_successes

    def live_branches(self) -> list[BranchRecord]:
        return [b for b in self._branches.values() if b.alive]

    def retired_branches(self) -> list[BranchRecord]:
        return [b for b in self._branches.values() if not b.alive]

    def begin_batch(self) -> None:
        """Enter deferred-retirement mode for a batch of register_attempt calls.

        While in batch mode, _maybe_retire queues retirements instead of
        applying them, so branch liveness is stable across all slots in the
        batch. Idempotent.
        """
        self._batch_mode = True
        self._pending_retirements = {}

    def end_batch(self, generation: int) -> None:
        """Exit batch mode and apply any retirements queued during the batch.

        Safe to call even if begin_batch was never called — just clears state.
        """
        pending = self._pending_retirements
        self._batch_mode = False
        self._pending_retirements = {}
        for branch_id, reason in pending.items():
            branch = self._branches.get(branch_id)
            if branch is None or not branch.alive:
                continue
            branch.alive = False
            branch.retired_reason = reason

    def register_attempt(
        self,
        *,
        parent_id: str | None,
        child_id: str,
        success: bool,
        score: float,
        family: str,
        aspect_signature: str,
        generation: int,
        embedding: list[float] | None = None,
    ) -> BranchAssignment:
        """Register an evaluated child against its branch.

        Called for EVERY child, success or failure. Runs Flow 1 (attempt
        accounting) on every call. On success, also runs Flow 2
        (promotion / mutation).
        """
        # --- Resolve assignment via lineage lookup -----------------------
        assigned_branch_id: str | None = None
        if parent_id is not None and parent_id in self._parent_to_branch:
            assigned_branch_id = self._parent_to_branch[parent_id]

        # --- Failure routing (no creation, no centroid fallback) --------
        if not success:
            if assigned_branch_id is None:
                self._orphan_failures += 1
                return BranchAssignment(branch_id=None, is_orphan=True)
            branch = self._branches[assigned_branch_id]
            branch.num_attempts += 1
            branch.consecutive_failures += 1
            # Retired branches still get diagnostic accounting but never revive.
            if branch.alive:
                self._maybe_retire(branch, generation)
            return BranchAssignment(branch_id=assigned_branch_id)

        # --- Success path -----------------------------------------------
        # Successful child with no live parent → no-lineage fallback.
        if assigned_branch_id is None or not self._branches[assigned_branch_id].alive:
            # nearest-by-centroid fallback (successful children only).
            nearest_id, nearest_dist = self._nearest_live_by_centroid(embedding)
            if (
                nearest_id is not None
                and nearest_dist is not None
                and nearest_dist <= MERGE_DISTANCE
            ):
                # Merge into nearest live branch as a continuation.
                self._extend_branch(
                    nearest_id,
                    child_id,
                    score,
                    generation,
                    embedding,
                    family=family,
                    aspect_signature=aspect_signature,
                )
                self._parent_to_branch[child_id] = nearest_id
                if assigned_branch_id is None:
                    self._orphan_successes += 1
                # Enforce stagnation / failure limits after the merge. Without
                # this, a branch that only ever grows through the nearest-
                # centroid fallback can accumulate stagnation but never be
                # retired, because every other exit path in register_attempt
                # calls _maybe_retire and this one used to bail out early.
                self._maybe_retire(self._branches[nearest_id], generation)
                return BranchAssignment(branch_id=nearest_id)

            new_id = self._create_branch(
                root_parent_id=child_id,
                family=family,
                aspect_signature=aspect_signature,
                score=score,
                generation=generation,
                created_via=("no_lineage" if assigned_branch_id is not None else "split"),
                centroid_embedding=list(embedding) if embedding else None,
            )
            self._parent_to_branch[child_id] = new_id
            if assigned_branch_id is None:
                self._orphan_successes += 1
            return BranchAssignment(branch_id=new_id, created_new=True)

        branch = self._branches[assigned_branch_id]

        # semantic split check (successful, parent in live branch).
        if self._should_split(branch, score, family, aspect_signature, embedding):
            # Charge an attempt to the parent branch first (we observed it).
            branch.num_attempts += 1
            branch.consecutive_failures = 0
            # Refresh stagnation on the parent. Normally _extend_branch does
            # this on the continuation path, but a split doesn't extend the
            # parent — without this refresh, a branch that mostly throws off
            # splits (never extending itself) never trips STAGNATION_LIMIT.
            branch.stagnation = generation - branch.last_improved_gen
            new_id = self._create_branch(
                root_parent_id=child_id,
                family=family,
                aspect_signature=aspect_signature,
                score=score,
                generation=generation,
                created_via="split",
                centroid_embedding=list(embedding) if embedding else None,
            )
            self._parent_to_branch[child_id] = new_id
            # Enforce retirement limits on the parent after charging the
            # attempt and refreshing stagnation. Matches the continuation
            # path at the bottom of register_attempt.
            self._maybe_retire(branch, generation)
            return BranchAssignment(branch_id=new_id, created_new=True)

        # Normal continuation: extend the existing branch
        self._extend_branch(
            assigned_branch_id,
            child_id,
            score,
            generation,
            embedding,
            family=family,
            aspect_signature=aspect_signature,
        )
        self._parent_to_branch[child_id] = assigned_branch_id
        self._maybe_retire(branch, generation)
        return BranchAssignment(branch_id=assigned_branch_id)

    def _extend_branch(
        self,
        branch_id: str,
        child_id: str,
        score: float,
        generation: int,
        embedding: list[float] | None,
        *,
        family: str,
        aspect_signature: str,
    ) -> None:
        branch = self._branches[branch_id]
        branch.num_attempts += 1
        branch.num_successes += 1
        branch.consecutive_failures = 0
        branch.depth += 1
        branch.latest_parent_id = child_id
        branch.latest_score = score
        if score > branch.best_score:
            branch.best_score = score
            branch.last_improved_gen = generation
            branch.stagnation = 0
            branch.num_improvements += 1
            # Issue #8 (Option E): refresh branch identity only when an
            # extending child improves the branch's best score, so stored
            # family/aspect_signature stay aligned with the branch's best
            # representative rather than the original founder. Truthy
            # guards mirror the defensive pattern in _should_split — don't
            # clobber a valid identity with empty strings.
            if family:
                branch.family = family
            if aspect_signature:
                branch.aspect_signature = aspect_signature
        else:
            branch.stagnation = generation - branch.last_improved_gen
        # EMA centroid update
        if embedding:
            if branch.centroid_embedding is None or len(branch.centroid_embedding) != len(
                embedding
            ):
                branch.centroid_embedding = list(embedding)
            else:
                branch.centroid_embedding = [
                    CENTROID_EMA * c + (1 - CENTROID_EMA) * e
                    for c, e in zip(branch.centroid_embedding, embedding)
                ]

    def _should_split(
        self,
        branch: BranchRecord,
        child_score: float,
        child_family: str,
        child_aspect_signature: str,
        child_embedding: list[float] | None,
    ) -> bool:
        if not child_embedding or branch.centroid_embedding is None:
            return False
        # Identity differs?
        family_diff = bool(child_family) and child_family != branch.family
        aspect_diff = (
            bool(child_aspect_signature) and child_aspect_signature != branch.aspect_signature
        )
        if not (family_diff or aspect_diff):
            return False
        # Score floor — only enforced for same-family forks. A genuinely
        # novel family (family_diff) almost always starts below the
        # incumbent's ceiling because it's an unoptimized first attempt;
        # gating splits on the incumbent's score floor silently absorbs
        # emerging strategies into the dominant branch and prevents the
        # diversity we're trying to preserve. Same-family aspect-only
        # splits still need the floor to reject weak forks.
        if not family_diff:
            floor = MIN_BRANCH_SCORE_FRAC * branch.best_score
            if branch.best_score > 0 and child_score < floor:
                return False
        # Embedding-distance gate — only enforced for same-family forks.
        # Parallel carve-out to the score floor above: a family-level
        # structural divergence (iterative-flat -> recursive-multi, etc.)
        # is a first-class split signal in its own right. Requiring the
        # embedding centroid to also move >SPLIT_DISTANCE silently absorbs
        # structurally-distinct strategies when the raw code still shares
        # surface text (stdin parsing, I/O boilerplate, shared variable
        # names), which the validation run empirically confirmed.
        # Same-family aspect-only splits still go through the distance
        # gate to avoid spurious forks on cosmetic cfhash drift.
        if family_diff:
            return True
        d = _cosine_distance(child_embedding, branch.centroid_embedding)
        if d is None:
            return False
        return d > SPLIT_DISTANCE

    def _nearest_live_by_centroid(
        self, embedding: list[float] | None
    ) -> tuple[str | None, float | None]:
        if not embedding:
            return None, None
        best_id: str | None = None
        best_d: float | None = None
        for b in self._branches.values():
            if not b.alive or b.centroid_embedding is None:
                continue
            d = _cosine_distance(embedding, b.centroid_embedding)
            if d is None:
                continue
            if best_d is None or d < best_d:
                best_d = d
                best_id = b.id
        return best_id, best_d

    def apply_dominance_retirement(self) -> list[str]:
        """Retire branches that are semantically dominated by another live branch.

        See design doc §1: dominance requires score margin AND newer
        improvement AND semantic overlap (aspect_signature match OR
        family match with centroid distance <= DOMINANCE_DISTANCE).
        Returns ids of newly retired branches.
        """
        live = self.live_branches()
        retired: list[str] = []
        for b in live:
            margin = DOMINANCE_MARGIN_FRAC * abs(b.best_score)
            for other in live:
                if other.id == b.id or not other.alive:
                    continue
                if other.best_score <= b.best_score + margin:
                    continue
                if other.last_improved_gen <= b.last_improved_gen:
                    continue
                # Semantic overlap check
                same_aspect = (
                    bool(other.aspect_signature) and other.aspect_signature == b.aspect_signature
                )
                same_family = bool(other.family) and other.family == b.family
                if not same_aspect:
                    if not same_family:
                        continue
                    d = _cosine_distance(b.centroid_embedding, other.centroid_embedding)
                    if d is None or d > DOMINANCE_DISTANCE:
                        continue
                b.alive = False
                b.retired_reason = "dominance"
                retired.append(b.id)
                break
        return retired

    def register_recombination_donor(
        self,
        *,
        donor_branch_id: str,
        success: bool,
    ) -> None:
        """Secondary credit for a recombination donor.

        Touches ONLY donor counters — never primary lineage fields.
        This method ships as a no-op-friendly stub so the
        recombination flow can wire it in without re-touching
        BranchRecord later.
        """
        branch = self._branches.get(donor_branch_id)
        if branch is None:
            return
        branch.num_donor_attempts += 1
        if success:
            branch.num_donor_successes += 1

    def evict_excess(self) -> list[str]:
        """Enforce MAX_LIVE_BRANCHES; return ids of newly retired branches.

        Eviction by composite score:
            0.5 * normalized_best + 0.3 * (1 / (stagnation+1)) + 0.2 * recency
        Lower composite → retired first.
        """
        live = self.live_branches()
        if len(live) <= MAX_LIVE_BRANCHES:
            return []

        if not live:
            return []
        max_best = max(b.best_score for b in live) or 1.0
        max_recent = max(b.last_improved_gen for b in live) or 1.0

        def composite(b: BranchRecord) -> float:
            norm_best = b.best_score / max_best if max_best > 0 else 0.0
            stag_term = 1.0 / (b.stagnation + 1)
            recency = b.last_improved_gen / max_recent if max_recent > 0 else 0.0
            return 0.5 * norm_best + 0.3 * stag_term + 0.2 * recency

        ranked = sorted(live, key=composite)
        evict_count = len(live) - MAX_LIVE_BRANCHES
        evicted: list[str] = []
        for b in ranked[:evict_count]:
            b.alive = False
            b.retired_reason = "max_live_cap"
            evicted.append(b.id)
        return evicted

    # --- branch-aware parent sampling ---------------------------

    def sample_branch_parents(
        self,
        roles: list[BranchRole],
        *,
        global_best: float = 0.0,
    ) -> list[tuple[BranchRole, BranchRecord]]:
        """Return (role, branch) pairs for the requested roles.

        The engine maps `branch.latest_parent_id` back to program code via
        its own index. Branches are deduplicated across roles so no role
        receives the same branch twice; if a role has no valid candidate
        it is simply dropped from the result.
        """
        live = self.live_branches()
        if not live:
            return []

        picks: list[tuple[BranchRole, BranchRecord]] = []
        used: set[str] = set()

        def _remaining() -> list[BranchRecord]:
            return [b for b in live if b.id not in used]

        anchor: BranchRecord | None = None

        for role in roles:
            pool = _remaining()
            if not pool:
                break
            choice: BranchRecord | None = None
            if role is BranchRole.ANCHOR:
                choice = max(pool, key=lambda b: (b.best_score, b.last_improved_gen))
                anchor = choice
            elif role is BranchRole.CONTINUATION:
                # Best improvement rate; require at least 2 successes and
                # at least one improvement; prefer non-stagnant lines.
                candidates = [b for b in pool if b.num_successes >= 2 and b.num_improvements >= 1]
                if candidates:
                    choice = max(
                        candidates,
                        key=lambda b: (
                            b.num_improvements / max(1, b.num_attempts),
                            b.best_score,
                        ),
                    )
            elif role is BranchRole.BREAKOUT:
                # Most stagnant viable branch (score >= 50% of global best).
                floor = 0.5 * global_best if global_best > 0 else 0.0
                candidates = [b for b in pool if b.best_score >= floor and b.stagnation > 0]
                if candidates:
                    choice = max(candidates, key=lambda b: (b.stagnation, b.best_score))
            elif role is BranchRole.DIVERSITY:
                # Farthest centroid from anchor's centroid (or any prior pick).
                reference = anchor.centroid_embedding if anchor else None
                if reference is None and picks:
                    reference = picks[0][1].centroid_embedding
                if reference is not None:
                    scored: list[tuple[float, BranchRecord]] = []
                    for b in pool:
                        d = _cosine_distance(reference, b.centroid_embedding)
                        if d is not None:
                            scored.append((d, b))
                    if scored:
                        scored.sort(key=lambda x: x[0], reverse=True)
                        choice = scored[0][1]
                if choice is None:
                    # Fallback: a different family branch if available
                    if anchor is not None:
                        diff_family = [b for b in pool if b.family != anchor.family]
                        if diff_family:
                            choice = max(diff_family, key=lambda b: b.best_score)

            if choice is not None:
                picks.append((role, choice))
                used.add(choice.id)

        return picks

    # --- recombination gates + pair finding ----------------------

    @property
    def last_recombination_gen(self) -> int:
        return self._last_recombination_gen

    @property
    def recombination_events(self) -> list[dict[str, Any]]:
        return list(self._recombination_events)

    def find_recombination_pair(
        self,
        *,
        global_best: float,
        global_stagnation: int,
        current_gen: int,
        spectral_undersampled: bool = False,
    ) -> tuple[BranchRecord, BranchRecord, dict[str, Any]] | None:
        """Select (anchor, donor) pair if all activation gates pass.

        Returns the pair plus a diagnostics dict (gate states + scores)
        when a suitable pair is found; returns None otherwise. The
        diagnostics are always populated by the first failing gate so
        callers can log *why* recombination did not fire.
        """
        diag: dict[str, Any] = {
            "live_count": 0,
            "global_stagnation": global_stagnation,
            "global_best": global_best,
            "cooldown_remaining": 0,
            "spectral_undersampled": spectral_undersampled,
            "reason": None,
        }

        # Gate 1 — branch population
        live = self.live_branches()
        diag["live_count"] = len(live)
        if len(live) < MIN_LIVE_FOR_RECOMBINE:
            diag["reason"] = "insufficient_live_branches"
            return None

        # Gate 2 — plateau
        if global_stagnation < MIN_PLATEAU_FOR_RECOMBINE:
            diag["reason"] = "global_stagnation_below_threshold"
            return None

        # Gate 3 — quality floor (need at least 2 branches at >= 0.7 * best)
        floor = RECOMBINE_QUALITY_FLOOR * global_best
        quality = [b for b in live if b.best_score >= floor]
        diag["quality_branches"] = len(quality)
        if len(quality) < 2:
            diag["reason"] = "quality_floor_two_needed"
            return None

        # Gate 5 — spectral sanity (cluster signal is trustworthy)
        if spectral_undersampled:
            diag["reason"] = "spectral_undersampled"
            return None

        # Gate 6 — cooldown
        cooldown_end = self._last_recombination_gen + RECOMBINE_COOLDOWN
        diag["cooldown_remaining"] = max(0, cooldown_end - current_gen)
        if current_gen < cooldown_end:
            diag["reason"] = "cooldown_active"
            return None

        # Pair selection: anchor = highest best_score in quality set;
        # donor = branch in `live` maximizing a pair score with the anchor.
        anchor = max(quality, key=lambda b: b.best_score)

        def _pair_score(b: BranchRecord) -> tuple[float, float]:
            """(diversity, score_sum). Sort key: diversity first, then sum."""
            if b.id == anchor.id:
                return (-1.0, -1.0)
            diversity = 0.0
            if anchor.centroid_embedding and b.centroid_embedding:
                d = _cosine_distance(anchor.centroid_embedding, b.centroid_embedding)
                if d is not None:
                    diversity = d
            else:
                diversity = 1.0 if b.family != anchor.family else 0.0
            return (diversity, anchor.best_score + b.best_score)

        ranked = sorted(live, key=_pair_score, reverse=True)
        donor = None
        best_diversity = 0.0
        for candidate in ranked:
            if candidate.id == anchor.id:
                continue
            if candidate.best_score < 0.5 * global_best:
                continue  # minimum quality floor for donor
            if anchor.centroid_embedding and candidate.centroid_embedding:
                d = _cosine_distance(anchor.centroid_embedding, candidate.centroid_embedding)
                if d is None:
                    continue
                if d < MIN_DIVERSITY:
                    continue
                best_diversity = d
            else:
                # No embeddings — require family difference as diversity proxy.
                if candidate.family == anchor.family:
                    continue
                best_diversity = 1.0
            donor = candidate
            break

        if donor is None:
            diag["reason"] = "no_diverse_donor"
            return None

        diag["anchor_id"] = anchor.id
        diag["donor_id"] = donor.id
        diag["diversity"] = best_diversity
        diag["reason"] = "gates_passed"
        return anchor, donor, diag

    def record_recombination_event(
        self,
        *,
        anchor_branch_id: str,
        donor_branch_id: str,
        child_id: str,
        success: bool,
        score: float,
        created_new_branch: bool,
        generation: int,
    ) -> None:
        """Log a recombination event and update bookkeeping.

        Callers must separately call `register_attempt` (for the child as
        a descendant of the anchor) and `register_recombination_donor` to
        credit the donor. This method handles only the event log and the
        cooldown timer.
        """
        self._last_recombination_gen = generation
        self._recombination_events.append(
            {
                "generation": generation,
                "anchor_branch_id": anchor_branch_id,
                "donor_branch_id": donor_branch_id,
                "child_id": child_id,
                "success": success,
                "score": score,
                "created_new_branch": created_new_branch,
            }
        )

    def render_report_section(self) -> list[str]:
        """Render a markdown fragment summarising branches.

        Kept on the manager so every runner can embed the same section
        without reimplementing the formatting.
        """
        lines: list[str] = []
        live = self.live_branches()
        retired = self.retired_branches()
        lines.append("## Branch Preservation")
        lines.append("")
        lines.append(f"- **Live branches**: {len(live)} (cap={MAX_LIVE_BRANCHES})")
        lines.append(f"- **Retired branches**: {len(retired)}")
        lines.append(f"- **Orphan successes**: {self._orphan_successes}")
        lines.append(f"- **Orphan failures**: {self._orphan_failures}")
        lines.append("")
        if live:
            lines.append("### Live")
            lines.append("")
            lines.append("| id | family | best | depth | att | succ | improv | stag | via |")
            lines.append("|----|--------|------|-------|-----|------|--------|------|-----|")
            for b in sorted(live, key=lambda x: x.best_score, reverse=True):
                lines.append(
                    f"| {b.id} | {b.family or '?'} | {b.best_score:.4f} "
                    f"| {b.depth} | {b.num_attempts} | {b.num_successes} "
                    f"| {b.num_improvements} | {b.stagnation} | {b.created_via} |"
                )
            lines.append("")
        if retired:
            lines.append("### Retired")
            lines.append("")
            lines.append("| id | family | best | reason | depth | improv |")
            lines.append("|----|--------|------|--------|-------|--------|")
            for b in sorted(retired, key=lambda x: x.best_score, reverse=True):
                lines.append(
                    f"| {b.id} | {b.family or '?'} | {b.best_score:.4f} "
                    f"| {b.retired_reason or '?'} | {b.depth} | {b.num_improvements} |"
                )
            lines.append("")
        if self._recombination_events:
            lines.append("### Recombination Events")
            lines.append("")
            lines.append("| gen | anchor | donor | child | success | score | new_branch |")
            lines.append("|-----|--------|-------|-------|---------|-------|------------|")
            for ev in self._recombination_events[-20:]:
                lines.append(
                    f"| {ev['generation']} | {ev['anchor_branch_id']} "
                    f"| {ev['donor_branch_id']} | {ev['child_id']} "
                    f"| {'yes' if ev['success'] else 'no'} "
                    f"| {ev['score']:.4f} "
                    f"| {'yes' if ev['created_new_branch'] else 'no'} |"
                )
            lines.append("")
        # Donor counters for live branches (only show branches ever used as donor)
        donor_lines = [b for b in self._branches.values() if b.num_donor_attempts > 0]
        if donor_lines:
            lines.append("### Donor Counters")
            lines.append("")
            lines.append("| id | family | donor_attempts | donor_successes | alive |")
            lines.append("|----|--------|----------------|-----------------|-------|")
            for b in sorted(donor_lines, key=lambda x: x.num_donor_attempts, reverse=True):
                lines.append(
                    f"| {b.id} | {b.family or '?'} | {b.num_donor_attempts} "
                    f"| {b.num_donor_successes} | {'yes' if b.alive else 'no'} |"
                )
            lines.append("")
        return lines

    # --- Persistence -----------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        return {
            "branches": [self._branch_to_dict(b) for b in self._branches.values()],
            "parent_to_branch": dict(self._parent_to_branch),
            "orphan_failures": self._orphan_failures,
            "orphan_successes": self._orphan_successes,
            "last_recombination_gen": self._last_recombination_gen,
            "recombination_events": list(self._recombination_events),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> BranchManager:
        mgr = cls()
        for raw in data.get("branches", []):
            b = cls._dict_to_branch(raw)
            mgr._branches[b.id] = b
        mgr._parent_to_branch = dict(data.get("parent_to_branch", {}))
        mgr._orphan_failures = int(data.get("orphan_failures", 0))
        mgr._orphan_successes = int(data.get("orphan_successes", 0))
        mgr._last_recombination_gen = int(
            data.get("last_recombination_gen", -1 - RECOMBINE_COOLDOWN)
        )
        mgr._recombination_events = list(data.get("recombination_events", []))
        return mgr

    def save_to(self, path: Path) -> None:
        path.write_text(json.dumps(self.serialize(), indent=2))

    @classmethod
    def load_from(cls, path: Path) -> BranchManager:
        if not path.exists():
            return cls()
        return cls.load(json.loads(path.read_text()))

    # --- Internal --------------------------------------------------------

    def _create_branch(
        self,
        *,
        root_parent_id: str,
        family: str,
        aspect_signature: str,
        score: float,
        generation: int,
        created_via: str = "split",
        centroid_embedding: list[float] | None = None,
    ) -> str:
        new_id = uuid.uuid4().hex[:8]
        b = BranchRecord(
            id=new_id,
            root_parent_id=root_parent_id,
            latest_parent_id=root_parent_id,
            family=family,
            aspect_signature=aspect_signature,
            centroid_embedding=centroid_embedding,
            created_gen=generation,
            last_improved_gen=generation,
            best_score=score,
            latest_score=score,
            depth=1,
            num_attempts=1,
            num_successes=1,
            num_improvements=1,
            consecutive_failures=0,
            stagnation=0,
            alive=True,
            created_via=created_via,
        )
        self._branches[new_id] = b
        return new_id

    def _maybe_retire(self, branch: BranchRecord, generation: int) -> None:
        if not branch.alive:
            return
        reason: str | None = None
        if branch.consecutive_failures >= FAILURE_LIMIT:
            reason = "consecutive_failures"
        elif branch.stagnation >= STAGNATION_LIMIT:
            reason = "stagnation"
        if reason is None:
            return
        # Fix 3: during a batch, queue the retirement so later slots in the
        # same batch still see this branch as live. Without this, a branch
        # can die mid-batch and subsequent slots hit the no-lineage fallback
        # or miss semantic-split opportunities against it.
        if self._batch_mode:
            self._pending_retirements.setdefault(branch.id, reason)
            return
        branch.alive = False
        branch.retired_reason = reason

    @staticmethod
    def _branch_to_dict(b: BranchRecord) -> dict[str, Any]:
        d = asdict(b)
        # Tuples don't survive JSON round-trip — convert.
        if d.get("recombination_parents") is not None:
            d["recombination_parents"] = list(d["recombination_parents"])
        return d

    @staticmethod
    def _dict_to_branch(d: dict[str, Any]) -> BranchRecord:
        rp = d.get("recombination_parents")
        if rp is not None:
            d["recombination_parents"] = tuple(rp)
        return BranchRecord(**d)
