# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""ESN engine: program-level search with core critic infrastructure."""

from __future__ import annotations

import ast
import hashlib
import json
import random  # noqa: S311 — deterministic seeding for reproducibility
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from esn.core.spectral_models import ESNConfig
from esn.core.archives import EliteArchive, FrontierArchive
from esn.core.enums import SearchMode
from esn.core.models import (
    CandidateRecord,
    EvaluationDiagnostics,
    EvaluationResult,
    SearchState,
)
from esn.core.operator_credit import OperatorCreditModel
from esn.core.search_mode_selector import SearchModeSelector

from esn.engine.ast_features import extract_ast_features
from esn.engine.batch_budget import BatchBudgetController, GenerationYield, RunStateSnapshot
from esn.engine.branch_manager import BranchManager
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.family_tracker import FamilyTracker
from esn.engine.models import MutationContext
from esn.engine.slot_scorer import BatchSlotScorer


def _require_eval_result(eval_result: Any) -> EvaluationResult:
    """Guard the domain-evaluator boundary.

    A newcomer wiring a custom ``DomainSpec`` can easily return a bare number
    (or a dict) from ``evaluator``. Without this guard that surfaces deep in the
    engine as ``'int' object has no attribute 'success'``. Fail loudly here, at
    the boundary, with a message that names the contract.
    """
    if not isinstance(eval_result, EvaluationResult):
        raise ValueError(
            "domain.evaluator must return an esn.EvaluationResult "
            f"(got {type(eval_result).__name__})"
        )
    return eval_result


@dataclass
class _CandidateOutcome:
    """Result of a single candidate's parallel mutation+compile+evaluate phase.

    Pure-data object with NO side effects — all state mutations
    happen in the sequential Phase 3 processing.
    """

    slot: int
    style: str
    mode: SearchMode
    parent_code: str

    success: bool = False
    score: float = 0.0
    raw_score: float = 0.0
    new_code: str = ""
    # Raw LLM output, preserved BEFORE the deterministic local improver
    # overwrites `new_code` with its polished artifact. Used so that branch
    # identity (aspect signature + centroid embedding) reflects the LLM's
    # actual strategy rather than the polish's numeric dump. Empty string
    # when no polish fired, in which case downstream call sites fall back
    # to `new_code`.
    raw_code: str = ""
    code_hash: str = ""
    family: str = ""
    family_confidence: str = ""
    solve_summary: str = ""

    eval_result: EvaluationResult | None = None
    compile_metadata: dict[str, Any] = field(default_factory=dict)

    failure_stage: str = ""
    errors: list[str] = field(default_factory=list)

    analysis: Any = None
    prediction: Any = None
    prediction_surprise: bool = False

    context: MutationContext | None = None


# Mutation style mapping per search mode
_MODE_STYLE_MAP: dict[SearchMode, list[str]] = {
    SearchMode.EXPLOIT: ["refine", "explore"],
    SearchMode.EXPLORE: ["explore", "radical"],
    SearchMode.REPAIR: ["repair", "refine"],
    SearchMode.BRIDGE: ["synthesize"],
    SearchMode.RECOVER: ["repair", "refine"],
    SearchMode.COMPRESS: ["refine"],
}

# Core styles tracked by UCB1 (synthesize is mode-triggered, not UCB1-sampled)
_CORE_STYLES = ["refine", "explore", "repair", "radical"]
_MIN_TRIES_PER_STYLE = 2  # Forced exploration before UCB1 kicks in

# Parent portfolio roles for batched generation
_PARENT_ROLES = ["best", "family_best", "breakthrough", "diversity"]


class _CodeWrapper:
    """Lightweight wrapper to pass code strings to Predictor/Mutator protocols."""

    def __init__(self, code: str) -> None:
        self._code = code

    @property
    def code(self) -> str:
        return self._code

    def summary(self) -> str:
        lines = self._code.strip().splitlines()
        return lines[0] if lines else ""

    def structural_hash(self) -> str:
        return hashlib.sha256(self._code.encode()).hexdigest()[:16]

    def serialize(self) -> str:
        return self._code

    @classmethod
    def deserialize(cls, data: str) -> _CodeWrapper:
        return cls(data)


class V3ObservationProvider:
    """Provides spectral observations from engine state."""

    def __init__(self, engine: ESNEngine):
        self._engine = engine

    def get_spectral_observations(self) -> list:
        from esn.core.spectral_compression import SpectralObservation

        observations: list[SpectralObservation] = []

        # Family summaries
        for line in self._engine._family_tracker.get_summary():
            if line.strip():
                observations.append(
                    SpectralObservation(
                        text=line.strip(),
                        embedding=None,
                        weight=0.7,
                        source="family",
                    )
                )

        # Recent failure summaries (deduplicated, capped at 3)
        seen_failures: set[str] = set()
        failure_count = 0
        for attempt in self._engine._recent_attempt_log:
            if not attempt.get("success", True) and attempt.get("error"):
                err_text = attempt["error"][:100]
                if err_text in seen_failures:
                    continue
                seen_failures.add(err_text)
                failure_count += 1
                if failure_count > 3:
                    break
                observations.append(
                    SpectralObservation(
                        text=f"Failed {attempt.get('style', 'unknown')}: {err_text}",
                        embedding=None,
                        weight=0.2,
                        source="failure",
                    )
                )

        # Top elite summaries with solve() description
        for candidate in self._engine.elite_archive.get_best(5):
            if candidate.object_summary:
                code = self._engine._program_store.get(candidate.id, "")
                solve_summary = self._engine._extract_solve_summary(code)
                observations.append(
                    SpectralObservation(
                        text=f"Elite {candidate.family}: score {candidate.score:.3f} — {solve_summary}",
                        embedding=None,
                        weight=0.6,
                        source="elite",
                    )
                )

        return observations


class ESNEngine:
    """Program-level evolutionary search engine with ESN critic infrastructure.

    Searches over programs (executable code) using LLM-driven mutation styles.
    Reuses core's archives, operator credit, knowledge bank, novelty computer,
    and persistence for observability and control.

    When batch_size > 1, each generation produces k candidates in parallel:
    - Phase 1 (sequential): select mode, build parent portfolio + style mix
    - Phase 2 (parallel): mutate + compile + evaluate k candidates
    - Phase 3 (sequential): archive, credit, knowledge updates for all results
    """

    def __init__(
        self,
        domain: DomainSpec,
        mutator: Any | None = None,  # Mutator protocol
        predictor: Any | None = None,  # Predictor protocol
        analyzer: Any | None = None,  # Analyzer protocol
        knowledge: Any | None = None,  # KnowledgeIntegration
        novelty_computer: Any | None = None,  # NoveltyComputer
        credit_model: OperatorCreditModel | None = None,
        config: ESNConfig | None = None,
        seed: int = 42,
        local_improver: Any | None = None,  # LocalImprover protocol
        batch_size: int = 1,
        slot_scorer: BatchSlotScorer | None = None,
        enable_recombination: bool = False,
        total_generations: int = 30,
    ) -> None:
        self.domain = domain
        self.mutator = mutator
        self.predictor = predictor
        self.analyzer = analyzer
        self.knowledge = knowledge
        self.novelty_computer = novelty_computer
        self.local_improver = local_improver
        self.config = config or ESNConfig()
        self._seed = seed
        self._rng = random.Random(seed)  # noqa: S311
        self._batch_size = batch_size

        # Control layer (from core)
        self.credit_model = credit_model or OperatorCreditModel()
        self.mode_selector = SearchModeSelector()
        self.elite_archive = EliteArchive(max_size=50)
        self.frontier_archive = FrontierArchive(max_size=100)

        # Family tracking
        self._family_tracker = FamilyTracker()

        # Slot scorer for batched generation
        self._slot_scorer = slot_scorer

        # State
        self.state = SearchState()
        self.generation = 0
        self._best_code: str = domain.initial_code
        self._best_score: float = 0.0
        self._seed_evaluated: bool = False
        self._program_store: dict[str, str] = {}  # id -> code
        self._last_diagnostics: EvaluationDiagnostics | None = None
        self._consecutive_failures: int = 0
        self._search_temperature: float = 0.0
        self._seen_hashes: set[str] = set()
        self._last_failed_code: str | None = None
        self._last_error_context: str = ""
        self._recent_attempt_log: list[dict[str, Any]] = []
        self._breakthrough_cooldown: int = 0

        # branch preservation
        self._branch_manager = BranchManager()
        self._code_to_program_id: dict[str, str] = {}  # code_hash → latest program id
        self._aspect_embedding_cache: dict[str, list[float] | None] = {}
        self.enable_recombination = enable_recombination
        # Slot-level recombination provenance (cleared per batch)
        self._recomb_slot_meta: dict[int, dict[str, str]] = {}

        # Adaptive batch budget controller (Phase 3: full adaptive enforcement)
        self._batch_budget = BatchBudgetController(
            total_generations=total_generations,
            initial_batch_size=batch_size,
        )
        self._last_batch_decision: dict[str, Any] | None = None

        # Wire observation provider into novelty computer
        if self.novelty_computer:
            self.novelty_computer.set_observation_providers([V3ObservationProvider(self)])

    def _evaluate_seed_if_needed(self) -> None:
        """Evaluate the seed program and seed archives before first mutation."""
        if self.elite_archive.size > 0 or self._seed_evaluated:
            return  # Already seeded

        compile_result = self.domain.compiler.compile(self._best_code)
        if not compile_result.success:
            return

        eval_result = _require_eval_result(self.domain.evaluator(compile_result.artifact))
        if not eval_result.success:
            return

        self._best_score = eval_result.score
        self.state.best_score = eval_result.score
        code_hash = hashlib.sha256(self._best_code.encode()).hexdigest()[:16]
        candidate = CandidateRecord(
            id=uuid.uuid4().hex[:8],
            generation=0,
            search_mode=SearchMode.EXPLOIT,
            operator_name="seed",
            object_hash=code_hash,
            object_summary=self._best_code[:200],
            score=eval_result.score,
            success=True,
        )
        family, confidence = extract_ast_features(self._best_code)["family"], "high"
        candidate.family = family
        candidate.family_confidence = confidence
        self.elite_archive.insert(candidate)
        self._program_store[candidate.id] = self._best_code
        self._seen_hashes.add(code_hash)
        self._code_to_program_id[code_hash] = candidate.id
        solve_summary = self._extract_solve_summary(self._best_code)
        self._family_tracker.record(family, eval_result.score, True, solve_summary)

        # register seed as the root of the first branch
        seed_aspect, seed_emb = self._branch_signals(self._best_code, family)
        seed_assignment = self._branch_manager.register_attempt(
            parent_id=None,
            child_id=candidate.id,
            success=True,
            score=eval_result.score,
            family=family,
            aspect_signature=seed_aspect,
            generation=0,
            embedding=seed_emb,
        )
        candidate.branch_id = seed_assignment.branch_id
        candidate.aspect_signature = seed_aspect
        self._seed_evaluated = True

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------

    def run_generation(self) -> CandidateRecord:
        """Execute one generation. Delegates to batch when batch_size > 1."""
        if self._batch_size <= 1:
            return self._run_single_generation()

        records = self.run_batch_generation()
        # Return the best successful candidate, or the first if all failed
        successful = [r for r in records if r.success]
        if successful:
            return max(successful, key=lambda r: r.score)
        return records[0]

    def run_batch_generation(self) -> list[CandidateRecord]:
        """Execute one generation with k parallel candidates.

        Returns list of CandidateRecords (one per candidate in the batch).
        """
        self._evaluate_seed_if_needed()

        # --- Phase 1: Sequential planning ---
        self.generation += 1
        self.state.generation = self.generation

        mode = self._select_mode()
        self.state.current_mode = mode

        # Capture pre-batch state for yield accounting (before any outcomes are processed)
        self._pre_batch_frontier_size = self.frontier_archive.size
        self._pre_batch_best_score = self._best_score

        # Batch budget approval (Phase 3: full adaptive enforcement)
        snapshot = self._build_batch_snapshot()
        heuristic_request = self._batch_budget.compute_heuristic_request(snapshot)
        decision = self._batch_budget.compute_approval(
            requested=heuristic_request,
            actual_batch=self._batch_size,
            snapshot=snapshot,
        )

        # Phase 3: full adaptive — controller can both shrink and expand.
        # Temporarily swap _batch_size so all downstream code (plan_batch,
        # slot_scorer, etc.) sees the effective value, then restore nominal
        # in the finally block.
        #
        # Warmup: don't enforce during the first lookback_window generations.
        # The controller needs history to make informed decisions.
        nominal_batch = self._batch_size
        warmup = self._batch_budget.config.lookback_window
        if self.generation <= warmup:
            effective_batch = nominal_batch
        elif decision.approved <= nominal_batch:
            # Shrinking: always allowed (Phase 2 behavior)
            effective_batch = decision.approved
        else:
            # Expansion: only when the run has earned it.
            effective_batch = self._gate_expansion(
                nominal_batch,
                decision.approved,
                snapshot,
            )
        self._batch_size = effective_batch

        # Update the BatchDecision object directly so both the controller's
        # internal _decisions list and batch_budget.json reflect the effective
        # batch (authoritative audit artifact).
        decision.actual = effective_batch
        self._last_batch_decision = decision.to_dict()

        try:
            # Consecutive failure recovery: after 2+ failures, snap back
            if self._consecutive_failures >= 2:
                mode = SearchMode.EXPLOIT
                self.state.current_mode = mode
                assignments = [([self._best_code], "refine")]
                self._recomb_slot_meta.clear()
            else:
                assignments = self._plan_batch(mode)

            # Build contexts for each assignment (sequential, reads shared state)
            planned: list[tuple[list[str], str, MutationContext]] = []
            for parents_list, style in assignments:
                context = self._build_context(mode, style, parents_list)
                planned.append((parents_list, style, context))

            # --- Parallel mutation + compile + evaluate ---
            # PythonSandboxCompiler uses signal.SIGALRM which only works in main
            # thread — fall back to sequential execution in that case.
            use_threads = len(planned) > 1 and not isinstance(
                self.domain.compiler, PythonSandboxCompiler
            )

            outcomes: list[_CandidateOutcome] = []
            if not use_threads:
                for i, (parents, style, ctx) in enumerate(planned):
                    outcomes.append(self._run_candidate(i, parents, style, mode, ctx))
            else:
                with ThreadPoolExecutor(max_workers=len(planned)) as pool:
                    futures = {}
                    for i, (parents, style, ctx) in enumerate(planned):
                        fut = pool.submit(self._run_candidate, i, parents, style, mode, ctx)
                        futures[fut] = i
                    for fut in as_completed(futures):
                        outcomes.append(fut.result())

            # --- Sequential state updates ---
            # Fix 3: hold branch retirements until every slot in the batch has
            # been processed. Otherwise a mid-batch stagnation retirement can
            # knock a branch out of existence before later slots in the same
            # batch get a chance to lineage-attach to it, forcing those slots
            # into the no-lineage fallback or missing split opportunities.
            records: list[CandidateRecord] = []
            any_success = False
            best_outcome: _CandidateOutcome | None = None

            self._branch_manager.begin_batch()
            try:
                for outcome in outcomes:
                    record = self._process_outcome(outcome)
                    records.append(record)
                    if outcome.success:
                        any_success = True
                        if best_outcome is None or outcome.score > best_outcome.score:
                            best_outcome = outcome
            finally:
                self._branch_manager.end_batch(self.generation)

            # Batch-level state updates
            self._finalize_batch(outcomes, any_success, best_outcome)

            return records
        finally:
            self._batch_size = nominal_batch

    def _select_mode(self) -> SearchMode:
        """Select search mode with engine overrides."""
        spectral_summary = None
        if self.novelty_computer and self.novelty_computer.spectral_state:
            ss = self.novelty_computer.spectral_state
            spectral_summary = {"S1": ss.S1, "S2": ss.S2, "erank": ss.erank}
        mode = self.mode_selector.select_mode(self.state, spectral_summary)

        # Engine override: force EXPLOIT during breakthrough cooldown
        if self._breakthrough_cooldown > 0:
            mode = SearchMode.EXPLOIT
            self._breakthrough_cooldown -= 1

        # Engine override: bias toward exploitation when spikes are active.
        # Phase 1 follow-up: honor BBP actionable spikes too, but only when the
        # spectral report is NOT flagged undersampled — otherwise we'd let a
        # noisy small-sample spectrum steer mode selection.
        has_active_structure = False
        if self.novelty_computer and self.novelty_computer.spectral_state:
            legacy_spikes = self.novelty_computer.spectral_state.num_spikes
            report = getattr(self.novelty_computer, "spectral_report", None)
            bbp_actionable = 0
            bbp_trusted = False
            if report is not None:
                bbp_actionable = sum(1 for s in report.spikes if s.above_gate)
                bbp_trusted = not report.undersampled
            has_active_structure = legacy_spikes > 0 or (bbp_actionable > 0 and bbp_trusted)
        if (
            has_active_structure
            and mode == SearchMode.EXPLORE
            and self.state.stagnation_counter < 6
        ):
            mode = SearchMode.EXPLOIT

        # Engine override: force EXPLORE if no improvement for 4+ gens
        if (
            self.state.stagnation_counter >= 4
            and mode == SearchMode.EXPLOIT
            and (not spectral_summary or spectral_summary.get("S1", 0) == 0)
        ):
            mode = SearchMode.EXPLORE

        return mode

    def _build_batch_snapshot(self) -> RunStateSnapshot:
        """Build a run-state snapshot for batch budget decisions."""
        # Meaningful branches: depth >= 2 or recent improvement
        meaningful = 0
        for branch in self._branch_manager.live_branches():
            if branch.depth >= 2 or branch.last_improved_gen >= self.generation - 5:
                meaningful += 1
        meaningful = max(1, meaningful)

        # Recent family diversity from attempt log
        recent_families = set()
        for attempt in self._recent_attempt_log[-10:]:
            fam = attempt.get("family", "")
            if fam and fam != "unknown" and attempt.get("success"):
                recent_families.add(fam)

        # Recent stats from batch budget history
        stats = self._batch_budget._compute_recent_stats()

        return RunStateSnapshot(
            generation=self.generation,
            total_generations=self._batch_budget._total_generations,
            meaningful_branch_count=meaningful,
            recent_family_diversity=max(1, len(recent_families)),
            recent_extra_slot_yield=stats["extra_slot_yield"],
            recent_duplicate_rate=stats["duplicate_rate"],
            recent_collapse_rate=stats["collapse_rate"],
            stagnation_counter=self.state.stagnation_counter,
            recent_frontier_improvements=stats["frontier_improvements"],
            recent_best_improved=stats["any_best_improved"],
        )

    def _gate_expansion(
        self,
        nominal: int,
        approved: int,
        snapshot: RunStateSnapshot,
    ) -> int:
        """Gate expansion above nominal batch size.

        Expansion is higher-risk than shrinking because it can burn budget
        quickly on weak evidence.  Only allow it when the run shows real
        justification: multiple productive branches, evidence that extra
        slots help, and low waste.

        Policy: expansion is for "productive diversity," not merely
        "non-wasteful diversity" — Gate 6 requires actual upward momentum
        (``recent_best_improved=True``), not just diversity in a flat regime.

        Returns the effective batch size (nominal <= effective <= approved).
        """
        # Gate 1: multiple meaningful branches (not converged onto one line)
        if snapshot.meaningful_branch_count < 2:
            return nominal

        # Gate 2: multiple successful families recently (real diversity)
        if snapshot.recent_family_diversity < 2:
            return nominal

        # Gate 3: evidence that extra slots actually helped
        if snapshot.recent_extra_slot_yield < 0.3:
            return nominal

        # Gate 4: not producing too many duplicates or collapses
        if snapshot.recent_duplicate_rate > 0.3:
            return nominal
        if snapshot.recent_collapse_rate > 0.3:
            return nominal

        # Gate 5: not already overspent relative to budget pace
        if self._batch_budget.pace_ratio > 1.2:
            return nominal

        # Gate 6: actual upward momentum — diversity alone is not enough
        if not snapshot.recent_best_improved:
            return nominal

        # All gates pass — allow expansion up to approved
        return approved

    def _plan_batch(self, mode: SearchMode) -> list[tuple[list[str], str]]:
        """Plan parent+style assignments for the batch.

        Returns list of (parents_list, style) tuples. parents_list has one
        entry for normal styles and two for multi-parent styles (synthesize,
        recombine). For batch_size=1, returns a single assignment using
        existing logic. For batch_size>1, uses the slot scorer or legacy
        portfolio selection.
        """
        k = self._batch_size
        self._recomb_slot_meta.clear()

        if k <= 1:
            parents = self._select_parents(mode)
            style = self._select_style(mode)
            return [([parents[0]], style)]

        if self._slot_scorer is None:
            return self._plan_batch_legacy(mode)

        # Slot scorer still returns (parent_code, style); wrap into lists.
        scorer_plan = self._slot_scorer.plan_batch(self, mode)
        return [([p], s) for p, s in scorer_plan]

    def _plan_batch_legacy(self, mode: SearchMode) -> list[tuple[list[str], str]]:
        """Legacy batch planning with hand-coded portfolio selection.

        Builds a style + parent portfolio, then optionally
        replaces one explore slot with a recombine slot if all activation
        gates pass.
        """
        k = self._batch_size

        # Build parent portfolio
        parent_portfolio = self._select_parent_portfolio(mode)
        # Build style portfolio
        style_portfolio = self._select_style_portfolio(mode, k)

        # Pair parents and styles: zip with cycling
        assignments: list[tuple[list[str], str]] = []
        for i in range(k):
            parent = parent_portfolio[i % len(parent_portfolio)]
            style = style_portfolio[i % len(style_portfolio)]
            assignments.append(([parent], style))

        # attempt to allocate one recombine slot.
        if self.enable_recombination:
            self._maybe_allocate_recombine_slot(assignments)

        return assignments

    def _maybe_allocate_recombine_slot(self, assignments: list[tuple[list[str], str]]) -> None:
        """Replace one explore slot with a recombine slot if gates pass.

        Updates `assignments` in-place and records donor metadata in
        `self._recomb_slot_meta[slot_index]` for later provenance logging.
        Never replaces the exploit anchor (slot 0 by convention) or an
        exploit/refine slot.
        """
        # Find the first replaceable slot: an explore style that is NOT
        # the anchor slot (slot 0 is typically refine or the global best).
        replace_idx: int | None = None
        for i, (_parents, style) in enumerate(assignments):
            if i == 0:
                continue
            if style == "explore":
                replace_idx = i
                break
        if replace_idx is None:
            return

        # Check gates via branch manager
        undersampled = False
        if self.novelty_computer is not None:
            report = getattr(self.novelty_computer, "spectral_report", None)
            if report is not None:
                undersampled = bool(getattr(report, "undersampled", False))

        pair = self._branch_manager.find_recombination_pair(
            global_best=self._best_score,
            global_stagnation=self.state.stagnation_counter,
            current_gen=self.generation,
            spectral_undersampled=undersampled,
        )
        if pair is None:
            return

        anchor_branch, donor_branch, _diag = pair
        anchor_code = self._program_store.get(anchor_branch.latest_parent_id)
        donor_code = self._program_store.get(donor_branch.latest_parent_id)
        if not anchor_code or not donor_code or anchor_code == donor_code:
            return

        assignments[replace_idx] = ([anchor_code, donor_code], "recombine")
        self._recomb_slot_meta[replace_idx] = {
            "anchor_branch_id": anchor_branch.id,
            "donor_branch_id": donor_branch.id,
        }

    def _select_parent_portfolio(self, mode: SearchMode) -> list[str]:
        """Select diverse parents for batched generation.

        Branch-aware sampling. When >= 2 live branches exist we
        pick parents by role (anchor/continuation/breakout/diversity) from
        the branch manager and resolve `latest_parent_id` to code via the
        program store. Falls back to the legacy archive-driven portfolio
        when branches are insufficient.
        """
        from esn.engine.branch_manager import BranchRole

        parents: list[str] = []
        seen_hashes: set[str] = set()

        def _add(code: str) -> None:
            if not code:
                return
            h = hashlib.sha256(code.encode()).hexdigest()[:16]
            if h not in seen_hashes:
                seen_hashes.add(h)
                parents.append(code)

        live_branches = self._branch_manager.live_branches()
        if len(live_branches) >= 2:
            picks = self._branch_manager.sample_branch_parents(
                [
                    BranchRole.ANCHOR,
                    BranchRole.CONTINUATION,
                    BranchRole.BREAKOUT,
                    BranchRole.DIVERSITY,
                ],
                global_best=self._best_score,
            )
            for _role, branch in picks:
                code = self._program_store.get(branch.latest_parent_id)
                if code:
                    _add(code)

        # Always include the current global best as a safety anchor.
        _add(self._best_code)

        # Legacy fallback roles (applied when branches didn't fill the slate).
        if len(parents) < 2:
            promising = self._find_underdeveloped_family()
            if promising:
                code = self._get_best_code_for_family(promising)
                if code:
                    _add(code)

            elites = self.elite_archive.get_best(3)
            for elite in elites[1:]:
                code = self._program_store.get(elite.id)
                if code:
                    _add(code)
                    break

            novel = self.frontier_archive.get_novel_candidates(2)
            for n in novel:
                code = self._program_store.get(n.id)
                if code:
                    _add(code)
                    break

        if not parents:
            parents.append(self._best_code)

        return parents

    def _select_style_portfolio(self, mode: SearchMode, k: int) -> list[str]:
        """Select diverse styles for batched generation.

        Guarantees: at least one refine, at least one explore.
        Remaining slots filled by UCB1 sampling.
        """
        styles: list[str] = []

        # Forced exploration check (early gens)
        for style in _CORE_STYLES:
            stats = self.credit_model.get_stats(style)
            if stats.attempts < _MIN_TRIES_PER_STYLE:
                styles.append(style)
                if len(styles) >= k:
                    return styles[:k]

        if not styles:
            # Guaranteed diversity: one refine + one explore
            styles.append("refine")
            if k >= 2:
                styles.append("explore")

        # Fill remaining with UCB1 samples
        eligible = _MODE_STYLE_MAP.get(mode, ["refine"])
        while len(styles) < k:
            sampled = self.credit_model.sample_operator(eligible, mode)
            styles.append(sampled)

        return styles[:k]

    def _run_candidate(
        self,
        slot: int,
        parents: list[str],
        style: str,
        mode: SearchMode,
        context: MutationContext,
    ) -> _CandidateOutcome:
        """Run a single candidate through mutate → compile → evaluate.

        This method is PURE — no shared state mutations. Safe for threads.
        """
        outcome = _CandidateOutcome(
            slot=slot,
            style=style,
            mode=mode,
            parent_code=parents[0],
            context=context,
        )

        # Predict (optional)
        if self.predictor and self.knowledge:
            hypotheses = self.knowledge.get_active_hypotheses_for_prompt(
                limit=15, novelty_computer=self.novelty_computer
            )
            score_history = context.score_history
            outcome.prediction = self.predictor.predict(
                _CodeWrapper(parents[0]),
                style,
                hypotheses,
                score_history,
            )

        # Mutate
        if self.mutator:
            parent_wrappers = [_CodeWrapper(c) for c in parents]
            mutation = self.mutator.mutate(parent_wrappers, style, context)
            if not mutation.success:
                outcome.failure_stage = "mutation"
                outcome.errors = list(mutation.errors[:3])
                if mutation.code:
                    outcome.new_code = mutation.code
                    outcome.family, outcome.family_confidence = (
                        extract_ast_features(mutation.code)["family"],
                        "high",
                    )
                return outcome
            outcome.new_code = mutation.code
        else:
            outcome.new_code = parents[0]

        # Validate
        validation_errors = (
            self.domain.compiler.validate(outcome.new_code)
            if hasattr(self.domain.compiler, "validate")
            else []
        )
        if validation_errors:
            outcome.failure_stage = "validation"
            outcome.errors = list(validation_errors[:3])
            if outcome.new_code:
                outcome.family, outcome.family_confidence = (
                    extract_ast_features(outcome.new_code)["family"],
                    "high",
                )
            return outcome

        # Compile
        compile_result = self.domain.compiler.compile(outcome.new_code)
        if not compile_result.success:
            outcome.failure_stage = "compile"
            outcome.errors = list(compile_result.errors[:3])
            if outcome.new_code:
                outcome.family, outcome.family_confidence = (
                    extract_ast_features(outcome.new_code)["family"],
                    "high",
                )
            return outcome

        # Evaluate
        eval_result = _require_eval_result(self.domain.evaluator(compile_result.artifact))
        outcome.eval_result = eval_result
        outcome.score = eval_result.score
        outcome.raw_score = eval_result.score
        outcome.success = eval_result.success
        outcome.family, outcome.family_confidence = (
            extract_ast_features(outcome.new_code)["family"],
            "high",
        )
        outcome.solve_summary = self._extract_solve_summary(outcome.new_code)

        if not eval_result.success:
            outcome.failure_stage = "eval"
            violations = []
            if eval_result.diagnostics and eval_result.diagnostics.violations:
                violations = eval_result.diagnostics.violations[:3]
            outcome.errors = [f"eval failure: score={eval_result.score}"]
            if violations:
                outcome.errors.extend(violations)

        # Local improvement (deterministic, no LLM)
        if outcome.success and self.local_improver:
            try:
                li_result = self.local_improver.improve(
                    code=outcome.new_code,
                    artifact=compile_result.artifact,
                    score=outcome.score,
                    evaluator=self.domain.evaluator,
                )
                if li_result.improved and li_result.score > outcome.score:
                    # Stash the raw LLM output BEFORE the polish overwrites
                    # new_code. Branch identity (aspect signature + centroid)
                    # must reflect the LLM's actual strategy, not the polish's
                    # numeric dump artifact.
                    outcome.raw_code = outcome.new_code
                    outcome.new_code = li_result.code
                    outcome.score = li_result.score
                    outcome.eval_result = self.domain.evaluator(li_result.artifact)
                    # Polish-shear fix: reclassify family after the polish
                    # rewrites new_code. The pre-polish classification (set
                    # above at the eval step) may no longer match the polished
                    # code's AST structure. Downstream consumers (family
                    # tracker, slot scorer, CandidateRecord) need the family
                    # of the code that actually gets stored and scored.
                    outcome.family = extract_ast_features(outcome.new_code)["family"]
            except Exception:  # noqa: S110
                pass

        # Analyze (optional)
        if outcome.success and self.analyzer and self.knowledge:
            outcome.analysis = self.analyzer.analyze(
                solution_summary=outcome.new_code[:500],
                score=outcome.score,
                diagnostics=eval_result.diagnostics,
                active_hypotheses=self.knowledge.get_active_hypotheses_for_prompt(
                    limit=10, novelty_computer=self.novelty_computer
                ),
                strategy=(
                    f"{style}: {context.intended_effect}" if context.intended_effect else style
                ),
            )

        # Prediction surprise
        if outcome.prediction is not None:
            lo, hi = outcome.prediction.score_range
            outcome.prediction_surprise = outcome.score < lo or outcome.score > hi

        outcome.code_hash = hashlib.sha256(outcome.new_code.encode()).hexdigest()[:16]

        return outcome

    def _process_outcome(self, outcome: _CandidateOutcome) -> CandidateRecord:
        """Process a single candidate outcome: update state + build record.

        This is called SEQUENTIALLY in Phase 3.
        """
        style = outcome.style
        mode = outcome.mode

        # Handle failures
        if outcome.failure_stage and outcome.failure_stage != "eval":
            if outcome.new_code:
                self._last_failed_code = outcome.new_code
            if outcome.family:
                self._family_tracker.record(outcome.family, 0.0, False, "")
            error_msg = f"{outcome.failure_stage} failure: {'; '.join(outcome.errors[:2])}"
            self._log_attempt(style, 0.0, success=False, error=error_msg, family=outcome.family)
            self._update_search_state(score=0.0, style=style, success=False)
            self.credit_model.record(
                operator_name=style,
                compile_success=(
                    outcome.failure_stage not in ("mutation", "validation", "compile")
                ),
                eval_success=False,
                score_delta=-1.0,
                generation=self.generation,
            )
            fail_compile_meta: dict[str, Any] = {
                "stage": outcome.failure_stage,
                "errors": outcome.errors,
            }
            # stamp recombination provenance on failed records too.
            fail_recomb_meta = (
                self._recomb_slot_meta.get(outcome.slot) if outcome.slot is not None else None
            )
            if fail_recomb_meta is not None:
                fail_compile_meta["recombined_from"] = [
                    fail_recomb_meta["anchor_branch_id"],
                    fail_recomb_meta["donor_branch_id"],
                ]

            fail_parent_id = self._lookup_parent_id(outcome.parent_code)
            failed_record = CandidateRecord(
                id=str(uuid.uuid4())[:8],
                generation=self.generation,
                parent_id=fail_parent_id,
                search_mode=mode,
                operator_name=style,
                object_hash="",
                score=0.0,
                success=False,
                family=outcome.family,
                family_confidence=outcome.family_confidence,
                compile_metadata=fail_compile_meta,
            )
            # register failed attempt against parent's branch (or orphan).
            # Use raw_code if present so signature reflects LLM strategy, not polish.
            # Issue #9: failed attempts never feed embedding-based split /
            # centroid logic, so skip embedding inference entirely.
            fail_aspect, _fail_emb = self._branch_signals(
                outcome.raw_code or outcome.new_code or "",
                outcome.family or "",
                compute_embedding=False,
            )
            assignment = self._branch_manager.register_attempt(
                parent_id=fail_parent_id,
                child_id=failed_record.id,
                success=False,
                score=0.0,
                family=outcome.family,
                aspect_signature=fail_aspect,
                generation=self.generation,
            )
            failed_record.branch_id = assignment.branch_id
            failed_record.aspect_signature = fail_aspect
            # Persist the failed candidate's source so post-hoc analysis
            # (programs.json, audit_log embed) can inspect what the mutator
            # actually produced. Applies to every non-eval failure stage —
            # compile, validation, mutation, AND timeout (stage="timeout" is
            # set by the compilers on subprocess.TimeoutExpired). Before
            # this fix, program_store was success-only which meant timed-out
            # candidates were unrecoverable after the run ended.
            if outcome.new_code:
                self._program_store[failed_record.id] = outcome.new_code
            # failed recombinations still credit donor attempts
            # and get logged as events for observability.
            if fail_recomb_meta is not None:
                self._branch_manager.register_recombination_donor(
                    donor_branch_id=fail_recomb_meta["donor_branch_id"],
                    success=False,
                )
                self._branch_manager.record_recombination_event(
                    anchor_branch_id=fail_recomb_meta["anchor_branch_id"],
                    donor_branch_id=fail_recomb_meta["donor_branch_id"],
                    child_id=failed_record.id,
                    success=False,
                    score=0.0,
                    created_new_branch=False,
                    generation=self.generation,
                )
            return failed_record

        # Successful compile+eval (may still have eval failure)
        score = outcome.score
        raw_score = outcome.raw_score
        success = outcome.success

        if not success:
            error_ctx = f"eval failure: score={score}"
            if outcome.errors:
                error_ctx += f"; {'; '.join(outcome.errors[:3])}"
            self._log_attempt(style, score, success=False, error=error_ctx, family=outcome.family)
        else:
            self._log_attempt(style, score, success=True, family=outcome.family)

        # Knowledge + novelty (sequential — not thread-safe)
        analysis_data = None
        if outcome.analysis and self.knowledge:
            from esn.core.observation_enrichment import build_observation_enrichment

            enrichment = build_observation_enrichment(
                code=outcome.new_code,
                score=outcome.score,
                style=outcome.style,
                intended_effect=(outcome.context.intended_effect if outcome.context else None),
                family=outcome.family or None,
                success=outcome.success,
                diagnostics=(outcome.eval_result.diagnostics if outcome.eval_result else None),
                errors=outcome.errors or None,
            )
            analysis_data = self.knowledge.preview_analysis(
                outcome.analysis, self.generation, enrichment=enrichment
            )

        ep_novelty, sp_novelty, unified_novelty = 0.0, 0.0, 0.0
        if self.novelty_computer and analysis_data:
            ep_novelty, sp_novelty, unified_novelty = self.novelty_computer.compute(
                relevant_data=analysis_data.get("relevant_data", []),
                new_count=analysis_data.get("new_count", 0),
                engaged_hypotheses=analysis_data.get("engaged", []),
                actual_score=score,
                prediction_surprise=outcome.prediction_surprise,
            )

        if self.knowledge and analysis_data:
            self.knowledge.apply_prepared_analysis(analysis_data, self.generation)

        # Credit (use raw score delta)
        raw_score_delta = raw_score - self._best_score if self._best_score > 0 else 0.0
        self.credit_model.record(
            operator_name=style,
            compile_success=True,
            eval_success=success,
            score_delta=raw_score_delta,
            epistemic_novelty=ep_novelty,
            spectral_novelty=sp_novelty,
            generation=self.generation,
        )

        # Build candidate record
        prediction_meta: dict[str, Any] = {
            "raw_score": raw_score,
            "local_improvement": score - raw_score,
        }
        if outcome.prediction is not None:
            prediction_meta["prediction_range"] = list(outcome.prediction.score_range)
            prediction_meta["prediction_surprise"] = outcome.prediction_surprise

        self._family_tracker.record(outcome.family, score, success, outcome.solve_summary)

        # Record combo outcome for slot scorer
        if outcome.family and self._slot_scorer is not None:
            self._slot_scorer.record_outcome(outcome.family, style, success, self.generation)

        # recombination provenance (if this slot was allocated
        # as a recombine, stamp it on compile_metadata BEFORE we create
        # the candidate so it round-trips through persistence).
        recomb_meta = self._recomb_slot_meta.get(outcome.slot) if outcome.slot is not None else None
        if recomb_meta is not None:
            prediction_meta["recombined_from"] = [
                recomb_meta["anchor_branch_id"],
                recomb_meta["donor_branch_id"],
            ]

        parent_id = self._lookup_parent_id(outcome.parent_code)
        candidate = CandidateRecord(
            id=str(uuid.uuid4())[:8],
            generation=self.generation,
            parent_id=parent_id,
            search_mode=mode,
            operator_name=style,
            object_hash=outcome.code_hash,
            object_summary=outcome.new_code[:200],
            score=score,
            success=success,
            diagnostics=outcome.eval_result.diagnostics if outcome.eval_result else None,
            epistemic_novelty=ep_novelty,
            spectral_novelty=sp_novelty,
            plan_rationale=outcome.context.intended_effect if outcome.context else "",
            family=outcome.family,
            family_confidence=outcome.family_confidence,
            slot=outcome.slot,
            compile_metadata=prediction_meta,
        )

        # Store program
        self._program_store[candidate.id] = outcome.new_code
        self._seen_hashes.add(outcome.code_hash)
        if outcome.code_hash:
            self._code_to_program_id[outcome.code_hash] = candidate.id

        # register attempt against parent's branch (success or eval-fail).
        # For recombine slots the anchor parent drives lineage attribution —
        # use outcome.parent_code (set to parents[0] = anchor in _run_candidate).
        # Use raw_code when polish fired, so branch identity tracks the LLM's
        # strategy and not the polished coordinate dump.
        aspect_sig, child_emb = self._branch_signals(
            outcome.raw_code or outcome.new_code or "", outcome.family or ""
        )
        assignment = self._branch_manager.register_attempt(
            parent_id=parent_id,
            child_id=candidate.id,
            success=success,
            score=score,
            family=outcome.family,
            aspect_signature=aspect_sig,
            generation=self.generation,
            embedding=child_emb if success else None,
        )
        candidate.branch_id = assignment.branch_id
        candidate.aspect_signature = aspect_sig

        # donor credit + event log (only for recombine slots).
        if recomb_meta is not None:
            self._branch_manager.register_recombination_donor(
                donor_branch_id=recomb_meta["donor_branch_id"],
                success=success,
            )
            self._branch_manager.record_recombination_event(
                anchor_branch_id=recomb_meta["anchor_branch_id"],
                donor_branch_id=recomb_meta["donor_branch_id"],
                child_id=candidate.id,
                success=success,
                score=score,
                created_new_branch=bool(assignment.created_new),
                generation=self.generation,
            )

        # Archive
        if success:
            elite_band = max(0, self._best_score * 0.005)
            if score >= self._best_score - elite_band:
                self.elite_archive.insert(candidate)
            else:
                self.frontier_archive.insert(candidate, novelty=unified_novelty)

        # Search state
        self._update_search_state(score=score, style=style, success=success)

        return candidate

    def _lookup_parent_id(self, parent_code: str) -> str | None:
        """Resolve parent code → program id via the code-hash index."""
        if not parent_code:
            return None
        h = hashlib.sha256(parent_code.encode()).hexdigest()[:16]
        return self._code_to_program_id.get(h)

    def _branch_signals(
        self, code: str, family: str, *, compute_embedding: bool = True
    ) -> tuple[str, list[float] | None]:
        """Build aspect_signature + deterministic feature vector for branch identity.

        The feature vector is built directly from AST features (one-hot
        family, binary feature flags, numeric scalars) — no neural embedder.
        This gives meaningful cosine distances by construction.

        Issue #9: pass ``compute_embedding=False`` on the failure path so
        the signature is still returned (branch_manager uses it for
        aspect-signature accounting on failed attempts) but no vector is
        computed. Failed attempts never feed the split / centroid logic.
        """
        from esn.engine.ast_features import extract_ast_features, features_to_vector
        from esn.engine.branch_manager import build_aspect_signature

        sig = build_aspect_signature(code=code or "", family=family or "")
        if not compute_embedding:
            return sig, None
        cache = self._aspect_embedding_cache
        if sig in cache:
            return sig, cache[sig]
        try:
            result = extract_ast_features(code or "")
            emb = features_to_vector(result)
        except Exception:
            emb = None
        cache[sig] = emb
        # Bound cache to avoid unbounded growth in long runs.
        if len(cache) > 512:
            # drop oldest insertion (dict preserves order)
            for k in list(cache.keys())[:128]:
                cache.pop(k, None)
        return sig, emb

    def _finalize_batch(
        self,
        outcomes: list[_CandidateOutcome],
        any_success: bool,
        best_outcome: _CandidateOutcome | None,
    ) -> None:
        """Batch-level updates after all candidates are processed."""
        # Consecutive failures: reset if any succeeded, increment if all failed
        if any_success:
            self._consecutive_failures = 0
            self._last_error_context = ""
        else:
            self._consecutive_failures += 1
            # Use the last failure's error context
            for outcome in outcomes:
                if outcome.errors:
                    self._last_error_context = (
                        f"{outcome.failure_stage} failure: {'; '.join(outcome.errors[:3])}"
                    )
                    if outcome.new_code:
                        self._last_failed_code = outcome.new_code

        # Best score update with stagnation deadband
        # 0.5% additive deadband (sign-agnostic): 0.5% of |baseline|, floored at
        # 1.0 so the threshold remains meaningful near score=0.
        improve_threshold = self._best_score + max(abs(self._best_score), 1.0) * 0.005
        if best_outcome and best_outcome.score > improve_threshold:
            self._best_score = best_outcome.score
            self._best_code = best_outcome.new_code
            self.state.stagnation_counter = 0
            self._search_temperature = 0.0
            self._breakthrough_cooldown = 3
        else:
            self.state.stagnation_counter += 1
            thresh = 3
            increment = 0.15
            if self.state.stagnation_counter > thresh:
                self._search_temperature = min(
                    1.0, (self.state.stagnation_counter - thresh) * increment
                )

        # Update best diagnostics from best outcome
        if best_outcome and best_outcome.eval_result:
            self._last_diagnostics = best_outcome.eval_result.diagnostics

        if best_outcome:
            self.state.best_score = max(self.state.best_score, best_outcome.score)
        self.state.best_score = max(self.state.best_score, self._best_score)
        self.state.elite_size = self.elite_archive.size
        self.state.frontier_size = self.frontier_archive.size
        self.state.frontier_distinct_count = self.frontier_archive.distinct_object_hashes

        # Knowledge maintenance (once per generation, not per candidate)
        if self.knowledge:
            self.knowledge.run_maintenance(self.generation)

        # Spectral update (once per generation)
        if self.novelty_computer:
            self.novelty_computer.end_of_generation()

        # dominance retirement, then enforce live-branch cap
        self._branch_manager.apply_dominance_retirement()
        self._branch_manager.evict_excess()

        # Prune program store
        if len(self._program_store) > 200:
            keep_ids = {c.id for c in self.elite_archive.get_all()}
            keep_ids |= {c.id for c in self.frontier_archive.get_all()}
            for pid in list(self._program_store):
                if pid not in keep_ids and len(self._program_store) > 100:
                    del self._program_store[pid]

        # Record batch yield for adaptive budget controller.
        # Pre-batch state was captured at the top of run_batch_generation().
        successes = sum(1 for o in outcomes if o.success)
        families_seen = {o.family for o in outcomes if o.success and o.family}
        code_hashes = [
            hashlib.sha256(o.new_code.encode()).hexdigest()[:16] for o in outcomes if o.new_code
        ]
        duplicates = len(code_hashes) - len(set(code_hashes))

        # frontier_improvements: compare current size to pre-batch snapshot
        frontier_before = getattr(self, "_pre_batch_frontier_size", self.frontier_archive.size)
        frontier_improvements = max(0, self.frontier_archive.size - frontier_before)

        # best_improved: compare best outcome to pre-batch best score
        pre_best = getattr(self, "_pre_batch_best_score", self._best_score)
        best_improved = best_outcome is not None and best_outcome.score > pre_best * 1.005

        # collapsed_count: domain-specific, not generically available.
        # Domains that expose basin_collapsed in eval_result.raw_outputs
        # can be detected here; for now count outcomes where eval succeeded
        # but produced a near-zero score relative to the pre-batch best.
        collapsed = 0
        for o in outcomes:
            if o.eval_result and o.eval_result.raw_outputs.get("basin_collapsed"):
                collapsed += 1

        gen_yield = GenerationYield(
            generation=self.generation,
            batch_size=len(outcomes),
            successes=successes,
            frontier_improvements=frontier_improvements,
            best_improved=best_improved,
            unique_families=len(families_seen),
            collapsed_count=collapsed,
            duplicate_count=duplicates,
        )
        self._batch_budget.record_yield(gen_yield)

    # ------------------------------------------------------------------
    # Single-candidate generation (original path, batch_size=1)
    # ------------------------------------------------------------------

    def _run_single_generation(self) -> CandidateRecord:
        """Execute one generation of the engine search loop (single candidate)."""

        # Evaluate seed before first mutation
        self._evaluate_seed_if_needed()

        # Step 1: Update state
        self.generation += 1
        self.state.generation = self.generation

        # Step 2: Select mode
        mode = self._select_mode()
        self.state.current_mode = mode

        # Step 3: Select parent(s)
        parents = self._select_parents(mode)

        # Consecutive failure recovery: after 2+ failures, snap back to elite + refine
        if self._consecutive_failures >= 2:
            mode = SearchMode.EXPLOIT
            self.state.current_mode = mode
            style = "refine"
            parents = [self._best_code]
        else:
            # Step 4: Select mutation style
            style = self._select_style(mode)

        # Step 5: Build mutation context
        context = self._build_context(mode, style, parents)

        # Step 6: Predict (Task 1, optional)
        prediction = None
        if self.predictor and self.knowledge:
            hypotheses = self.knowledge.get_active_hypotheses_for_prompt(
                limit=15, novelty_computer=self.novelty_computer
            )
            score_history = self._collect_score_history()
            context.score_history = score_history
            prediction = self.predictor.predict(
                _CodeWrapper(parents[0]),
                style,
                hypotheses,
                score_history,
            )

        # Step 7: Mutate
        if self.mutator:
            parent_wrappers = [_CodeWrapper(c) for c in parents]
            mutation = self.mutator.mutate(parent_wrappers, style, context)
            if not mutation.success:
                fail_family = ""
                if mutation.code:
                    self._last_failed_code = mutation.code
                    fail_family, _conf = extract_ast_features(mutation.code)["family"], "high"
                    self._family_tracker.record(fail_family, 0.0, False, "")
                self._update_search_state(score=0.0, style=style, success=False)
                self._log_attempt(
                    style,
                    0.0,
                    success=False,
                    error=f"mutation failure: {'; '.join(mutation.errors[:2])}",
                    family=fail_family,
                )
                return self._record_failure(parents[0], style, mode, mutation.errors, "mutation")
            new_code = mutation.code
        else:
            # No mutator -- identity (for testing)
            new_code = parents[0]

        # Validate
        validation_errors = (
            self.domain.compiler.validate(new_code)
            if hasattr(self.domain.compiler, "validate")
            else []
        )
        if validation_errors:
            self._last_failed_code = new_code
            val_family = ""
            if new_code:
                val_family, _conf = extract_ast_features(new_code)["family"], "high"
                self._family_tracker.record(val_family, 0.0, False, "")
            self._update_search_state(score=0.0, style=style, success=False)
            self._log_attempt(
                style,
                0.0,
                success=False,
                error=f"validation failure: {'; '.join(validation_errors[:2])}",
                family=val_family,
            )
            return self._record_failure(parents[0], style, mode, validation_errors, "validation")

        # Step 8: Compile (execute program in sandbox)
        compile_result = self.domain.compiler.compile(new_code)
        if not compile_result.success:
            self._last_failed_code = new_code
            comp_family = ""
            if new_code:
                comp_family, _conf = extract_ast_features(new_code)["family"], "high"
                self._family_tracker.record(comp_family, 0.0, False, "")
            self._update_search_state(score=0.0, style=style, success=False)
            self._log_attempt(
                style,
                0.0,
                success=False,
                error=f"compile failure: {'; '.join(compile_result.errors[:2])}",
                family=comp_family,
            )
            return self._record_failure(parents[0], style, mode, compile_result.errors, "compile")

        # Step 9: Evaluate
        eval_result = _require_eval_result(self.domain.evaluator(compile_result.artifact))
        score = eval_result.score
        raw_score = score  # Capture before local improvement
        success = eval_result.success

        # Classify family for logging (full classification happens in Step 14)
        eval_family, _ = extract_ast_features(new_code)["family"], "high"

        if not success:
            self._consecutive_failures += 1
            self._last_error_context = f"eval failure: score={score}"
            if eval_result.diagnostics and eval_result.diagnostics.violations:
                self._last_error_context += f"; {'; '.join(eval_result.diagnostics.violations[:3])}"
            self._log_attempt(
                style, score, success=False, error=self._last_error_context, family=eval_family
            )
        else:
            self._consecutive_failures = 0
            self._last_error_context = ""
            self._log_attempt(style, score, success=True, family=eval_family)

        # Step 9b: Local improvement (deterministic, no LLM)
        # Preserve the raw LLM output so branch identity reflects the LLM's
        # actual strategy rather than the polish's numeric dump. Empty string
        # means "no polish fired" and downstream call sites should fall back
        # to new_code.
        raw_new_code = ""
        if success and self.local_improver:
            try:
                li_result = self.local_improver.improve(
                    code=new_code,
                    artifact=compile_result.artifact,
                    score=score,
                    evaluator=self.domain.evaluator,
                )
                if li_result.improved and li_result.score > score:
                    raw_new_code = new_code
                    new_code = li_result.code
                    score = li_result.score
                    eval_result = self.domain.evaluator(li_result.artifact)
                    compile_result = self.domain.compiler.compile(new_code)
            except Exception:  # noqa: S110
                pass  # local improvement is best-effort

        # Step 10: Analyze (Task 2, optional)
        analysis_data = None
        if self.analyzer and self.knowledge:
            analysis = self.analyzer.analyze(
                solution_summary=new_code[:500],
                score=score,
                diagnostics=eval_result.diagnostics,
                active_hypotheses=self.knowledge.get_active_hypotheses_for_prompt(
                    limit=10, novelty_computer=self.novelty_computer
                ),
                strategy=(
                    f"{style}: {context.intended_effect}" if context.intended_effect else style
                ),
            )
            # Step 11a: Preview analysis (before novelty)
            from esn.core.observation_enrichment import build_observation_enrichment

            enrichment = build_observation_enrichment(
                code=new_code,
                score=score,
                style=style,
                intended_effect=context.intended_effect if context else None,
                family=eval_family or None,
                success=success,
                diagnostics=eval_result.diagnostics if eval_result else None,
                errors=None,
            )
            analysis_data = self.knowledge.preview_analysis(
                analysis, self.generation, enrichment=enrichment
            )

        # Step 11b: Compute novelty
        prediction_surprise = False
        if prediction is not None:
            lo, hi = prediction.score_range
            prediction_surprise = (score < lo) or (score > hi)

        ep_novelty, sp_novelty, unified_novelty = 0.0, 0.0, 0.0
        if self.novelty_computer and analysis_data:
            ep_novelty, sp_novelty, unified_novelty = self.novelty_computer.compute(
                relevant_data=analysis_data.get("relevant_data", []),
                new_count=analysis_data.get("new_count", 0),
                engaged_hypotheses=analysis_data.get("engaged", []),
                actual_score=score,
                prediction_surprise=prediction_surprise,
            )

        # Step 11c: Apply knowledge updates (after novelty)
        if self.knowledge and analysis_data:
            self.knowledge.apply_prepared_analysis(analysis_data, self.generation)

        # Step 11d: Maintenance
        if self.knowledge:
            self.knowledge.run_maintenance(self.generation)

        # Step 12: Update spectral
        if self.novelty_computer:
            self.novelty_computer.end_of_generation()

        # Step 13: Update operator credit (use raw score delta so credit
        # reflects LLM contribution, not local optimizer contribution)
        raw_score_delta = raw_score - self._best_score if self._best_score > 0 else 0.0
        self.credit_model.record(
            operator_name=style,
            compile_success=True,
            eval_success=success,
            score_delta=raw_score_delta,
            epistemic_novelty=ep_novelty,
            spectral_novelty=sp_novelty,
            generation=self.generation,
        )

        # Step 14: Build candidate record
        code_hash = hashlib.sha256(new_code.encode()).hexdigest()[:16]
        prediction_meta: dict[str, Any] = {
            "raw_score": raw_score,
            "local_improvement": score - raw_score,
        }
        if prediction is not None:
            prediction_meta["prediction_range"] = list(prediction.score_range)
            prediction_meta["prediction_surprise"] = prediction_surprise
        # Classify family
        family, family_confidence = extract_ast_features(new_code)["family"], "high"
        solve_summary = self._extract_solve_summary(new_code)
        self._family_tracker.record(family, score, success, solve_summary)

        parent_id = self._lookup_parent_id(parents[0] if parents else "")
        candidate = CandidateRecord(
            id=str(uuid.uuid4())[:8],
            generation=self.generation,
            parent_id=parent_id,
            search_mode=mode,
            operator_name=style,
            object_hash=code_hash,
            object_summary=new_code[:200],
            score=score,
            success=success,
            diagnostics=eval_result.diagnostics,
            epistemic_novelty=ep_novelty,
            spectral_novelty=sp_novelty,
            plan_rationale=context.intended_effect,
            family=family,
            family_confidence=family_confidence,
            compile_metadata=prediction_meta,
        )

        # Store program for parent selection
        self._program_store[candidate.id] = new_code
        self._seen_hashes.add(code_hash)
        if code_hash:
            self._code_to_program_id[code_hash] = candidate.id

        # register attempt against parent's branch.
        # Prefer raw LLM output so branch identity tracks the strategy, not
        # the polished coordinate dump.
        aspect_sig, child_emb = self._branch_signals(raw_new_code or new_code or "", family or "")
        assignment = self._branch_manager.register_attempt(
            parent_id=parent_id,
            child_id=candidate.id,
            success=success,
            score=score,
            family=family,
            aspect_signature=aspect_sig,
            generation=self.generation,
            embedding=child_emb if success else None,
        )
        candidate.branch_id = assignment.branch_id
        candidate.aspect_signature = aspect_sig

        # Step 15: Update archives
        if success:
            elite_band = max(0, self._best_score * 0.005)
            if score >= self._best_score - elite_band:
                self.elite_archive.insert(candidate)
            else:
                self.frontier_archive.insert(candidate, novelty=unified_novelty)

        # Step 16: Update search state
        self._update_search_state(score=score, style=style, success=success)
        self.state.elite_size = self.elite_archive.size
        self.state.frontier_size = self.frontier_archive.size
        self.state.frontier_distinct_count = self.frontier_archive.distinct_object_hashes

        # Update best score with stagnation deadband
        # 0.5% additive deadband (sign-agnostic): 0.5% of |baseline|, floored at
        # 1.0 so the threshold remains meaningful near score=0.
        improve_threshold = self._best_score + max(abs(self._best_score), 1.0) * 0.005
        if score > improve_threshold:
            self._best_score = score
            self._best_code = new_code
            self.state.stagnation_counter = 0
            self._search_temperature = 0.0
            self._breakthrough_cooldown = 3  # exploit for 3 gens after breakthrough
        else:
            self.state.stagnation_counter += 1
            # v1-style search temperature
            thresh = 3
            increment = 0.15
            if self.state.stagnation_counter > thresh:
                self._search_temperature = min(
                    1.0, (self.state.stagnation_counter - thresh) * increment
                )

        self.state.best_score = max(self.state.best_score, score)
        self._last_diagnostics = eval_result.diagnostics

        # Step 17: Prune program store
        if len(self._program_store) > 200:
            keep_ids = {c.id for c in self.elite_archive.get_all()}
            keep_ids |= {c.id for c in self.frontier_archive.get_all()}
            for pid in list(self._program_store):
                if pid not in keep_ids and len(self._program_store) > 100:
                    del self._program_store[pid]

        # dominance retirement, then enforce live-branch cap
        self._branch_manager.apply_dominance_retirement()
        self._branch_manager.evict_excess()

        return candidate

    # --- Private helpers ---

    def _select_parents(self, mode: SearchMode) -> list[str]:
        """Select parent program code(s) based on search mode."""
        if mode == SearchMode.BRIDGE:
            elites = self.elite_archive.get_best(3)
            if len(elites) >= 2:
                return [self._program_store.get(e.id, self._best_code) for e in elites[:3]]

        if mode == SearchMode.EXPLOIT:
            if self._search_temperature >= 0.6 and self.elite_archive.size >= 3:
                elites = self.elite_archive.get_best(3)
                return [self._program_store.get(elites[2].id, self._best_code)]
            return [self._best_code]

        if mode == SearchMode.EXPLORE:
            # prefer branch-aware breakout/diversity when we have
            # multiple live branches to choose from.
            from esn.engine.branch_manager import BranchRole

            if len(self._branch_manager.live_branches()) >= 2:
                picks = self._branch_manager.sample_branch_parents(
                    [BranchRole.BREAKOUT, BranchRole.DIVERSITY],
                    global_best=self._best_score,
                )
                for _role, branch in picks:
                    code = self._program_store.get(branch.latest_parent_id)
                    if code:
                        return [code]
            # Family-aware: find underdeveloped promising family
            promising = self._find_underdeveloped_family()
            if promising:
                code = self._get_best_code_for_family(promising)
                if code:
                    return [code]
            # Fallback: frontier novel candidate
            novel = self.frontier_archive.get_novel_candidates(1)
            if novel:
                return [self._program_store.get(novel[0].id, self._best_code)]
            return [self._best_code]

        if mode in (SearchMode.REPAIR, SearchMode.RECOVER):
            if self._last_failed_code is not None:
                return [self._last_failed_code]
            return [self._best_code]

        return [self._best_code]

    def _find_underdeveloped_family(self) -> str | None:
        """Find a family with decent scores but few attempts."""
        for name, stats in sorted(
            self._family_tracker._families.items(),
            key=lambda x: x[1].best_score,
            reverse=True,
        ):
            if stats.attempt_count <= 4 and stats.best_score > self._best_score * 0.8:
                return name
        return None

    def _get_best_code_for_family(self, family: str) -> str | None:
        """Get the best program code from a given family."""
        # Elite archive is sorted by score descending
        for candidate in self.elite_archive.get_all():
            if candidate.family == family and candidate.id in self._program_store:
                return self._program_store[candidate.id]
        for candidate in self.frontier_archive.get_all():
            if candidate.family == family and candidate.id in self._program_store:
                return self._program_store[candidate.id]
        return None

    def _select_style(self, mode: SearchMode) -> str:
        """Select mutation style: forced exploration, then UCB1."""
        if mode == SearchMode.BRIDGE:
            return "synthesize"

        # Forced exploration: each core style must be tried at least N times
        # This overrides mode filtering to ensure all styles get tested
        for style in _CORE_STYLES:
            stats = self.credit_model.get_stats(style)
            if stats.attempts < _MIN_TRIES_PER_STYLE:
                return style

        # After forced exploration complete, filter by mode and sample via UCB1
        eligible = _MODE_STYLE_MAP.get(mode, ["refine"])
        return self.credit_model.sample_operator(eligible, mode)

    def _build_context(
        self,
        mode: SearchMode,
        style: str,
        parents: list[str],
    ) -> MutationContext:
        """Build mutation context with all available signals."""
        top_hypotheses: list[dict[str, Any]] = []
        if self.knowledge:
            top_hypotheses = self.knowledge.get_active_hypotheses_for_prompt(
                limit=10, novelty_computer=self.novelty_computer
            )

        spectral_guidance: dict[str, Any] = {}
        if self.novelty_computer:
            spectral_guidance = self.novelty_computer.spectral_guidance

        diagnostics: dict[str, Any] = {}
        if self._last_diagnostics:
            diagnostics = self._last_diagnostics.model_dump()

        # Build recent attempts from rolling log (includes failure reasons)
        recent_attempts: list[dict[str, Any]] = list(self._recent_attempt_log)

        # Build intended_effect from mode + style + stagnation
        if style == "explore":
            intended_effect = (
                f"Find a qualitatively different algorithm family. "
                f"Current best ({self.state.best_score:.4f}) has been stagnant "
                f"for {self.state.stagnation_counter} gens."
            )
        elif style == "radical":
            intended_effect = (
                f"Invent a completely new solver approach. "
                f"Nothing in the archive has exceeded {self.state.best_score:.4f}."
            )
        elif style == "repair":
            intended_effect = f"Fix the specific failure: {self._last_error_context}"
        else:
            intended_effect = (
                f"Improve the current best ({self.state.best_score:.4f}) "
                f"through targeted refinement."
            )

        # Build targeted_hypothesis_ids from uncertain hypotheses
        targeted_hypothesis_ids: list[str] = []
        if self.knowledge and hasattr(self.knowledge, "get_active_hypothesis_records"):
            records = self.knowledge.get_active_hypothesis_records()
            uncertain = [r for r in records if 0.3 <= r.confidence <= 0.7]
            targeted_hypothesis_ids = [r.id for r in uncertain[:3]]

        # Build archive family summaries from top elites (use solve() docstring)
        archive_families: list[str] = []
        for elite in self.elite_archive.get_best(5):
            code = self._program_store.get(elite.id, "")
            summary = self._extract_solve_summary(code)
            archive_families.append(f"- Score {elite.score:.4f}: {summary}")

        # Family reasoning
        family_summaries = self._family_tracker.get_summary()
        parent_family, _ = (
            (extract_ast_features(parents[0])["family"], "high") if parents else ("unknown", "low")
        )

        # Collect family failure reasons from recent attempt log
        family_failure_reasons: dict[str, list[str]] = {}
        for attempt in self._recent_attempt_log:
            if not attempt.get("success") and attempt.get("error"):
                # Use family from attempt if stored, otherwise skip
                attempt_family = attempt.get("family", "")
                if attempt_family and attempt_family != "unknown":
                    family_failure_reasons.setdefault(attempt_family, []).append(attempt["error"])

        return MutationContext(
            search_mode=mode.value,
            mutation_style=style,
            top_hypotheses=top_hypotheses,
            spectral_guidance=spectral_guidance,
            search_temperature=self._search_temperature,
            diagnostics=diagnostics,
            score_history=self._collect_score_history(),
            error_context=self._last_error_context,
            best_code=self._best_code,
            best_score=self.state.best_score,
            recent_attempts=recent_attempts,
            archive_families=archive_families,
            stagnation_gens=self.state.stagnation_counter,
            intended_effect=intended_effect,
            targeted_hypothesis_ids=targeted_hypothesis_ids,
            family_summaries=family_summaries,
            parent_family=parent_family,
            family_failure_reasons=family_failure_reasons,
        )

    def _collect_score_history(self) -> dict[str, Any]:
        """Collect score history for prediction calibration."""
        scores = self.state.recent_scores
        if not scores:
            return {}
        return {
            "recent": scores[-5:],
            "best": self.state.best_score,
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "generation": self.generation,
        }

    def _update_search_state(self, score: float, style: str, success: bool) -> None:
        """Update search-state bookkeeping (scores, operators, stagnation)."""
        self.state.recent_scores.append(score)
        if len(self.state.recent_scores) > 20:
            self.state.recent_scores = self.state.recent_scores[-20:]
        self.state.recent_operators.append(style)
        if len(self.state.recent_operators) > 20:
            self.state.recent_operators = self.state.recent_operators[-20:]
        # Note: stagnation_counter is updated once per generation in
        # _finalize_batch() (batched path) or _run_single_generation()
        # (single path), NOT per-candidate here.

    def _log_attempt(
        self,
        style: str,
        score: float,
        *,
        success: bool,
        error: str = "",
        family: str = "",
    ) -> None:
        """Append to rolling attempt log."""
        entry: dict[str, Any] = {"style": style, "score": score, "success": success}
        if error:
            entry["error"] = error
        if family:
            entry["family"] = family
        self._recent_attempt_log.append(entry)
        max_log = max(8, 2 * self._batch_size)
        if len(self._recent_attempt_log) > max_log:
            self._recent_attempt_log = self._recent_attempt_log[-max_log:]

    @staticmethod
    def _extract_solve_summary(code: str) -> str:
        """Extract solve() docstring as a summary, falling back gracefully."""
        if not code.strip():
            return "empty program"
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "solve":
                    docstring = ast.get_docstring(node)
                    if docstring:
                        return docstring.strip().split("\n")[0][:120]
                    return "no description"
            return "no solve() function"
        except Exception:
            return "unparseable"

    def _record_failure(
        self,
        parent_code: str,
        style: str,
        mode: SearchMode,
        errors: list[str],
        stage: str,
    ) -> CandidateRecord:
        """Record a failed generation attempt."""
        self._consecutive_failures += 1
        self._last_error_context = f"{stage} failure: {'; '.join(errors[:3])}"
        self.credit_model.record(
            operator_name=style,
            compile_success=(stage not in ("mutation", "validation", "compile")),
            eval_success=False,
            score_delta=-1.0,
            generation=self.generation,
        )
        parent_id = self._lookup_parent_id(parent_code)
        record = CandidateRecord(
            id=str(uuid.uuid4())[:8],
            generation=self.generation,
            parent_id=parent_id,
            search_mode=mode,
            operator_name=style,
            object_hash="",
            score=0.0,
            success=False,
            compile_metadata={"stage": stage, "errors": errors},
        )
        # register failure against parent's branch
        assignment = self._branch_manager.register_attempt(
            parent_id=parent_id,
            child_id=record.id,
            success=False,
            score=0.0,
            family="",
            aspect_signature="",
            generation=self.generation,
        )
        record.branch_id = assignment.branch_id
        return record

    # --- Persistence ---

    def save_state(self, directory: Path) -> None:
        """Save full engine state for resume."""
        directory.mkdir(parents=True, exist_ok=True)

        from esn.core.persistence import (
            ArchiveStore,
            KnowledgeStore,
            NoveltyStore,
            OperatorCreditStore,
            SearchStateStore,
        )

        SearchStateStore.save(self.state, directory / "search_state.json")
        ArchiveStore.save_elite(self.elite_archive, directory / "elite.json")
        ArchiveStore.save_frontier(self.frontier_archive, directory / "frontier.json")
        OperatorCreditStore.save(self.credit_model, directory / "credit.json")

        if self.knowledge:
            KnowledgeStore.save(self.knowledge, directory / "knowledge.json")
        if self.novelty_computer:
            NoveltyStore.save(self.novelty_computer, directory / "novelty.json")

        # engine-specific state
        v3_state = {
            "generation": self.generation,
            "best_score": self._best_score,
            "best_code": self._best_code,
            "seed_evaluated": self._seed_evaluated,
            "consecutive_failures": self._consecutive_failures,
            "search_temperature": self._search_temperature,
            "seen_hashes": list(self._seen_hashes),
            "seed": self._seed,
            "last_failed_code": self._last_failed_code,
            "last_error_context": self._last_error_context,
            "recent_attempt_log": self._recent_attempt_log,
            "breakthrough_cooldown": self._breakthrough_cooldown,
            "family_tracker": self._family_tracker.to_dict(),
            "batch_size": self._batch_size,
            "batch_budget": self._batch_budget.to_dict(),
            "last_batch_decision": self._last_batch_decision,
        }
        if self._slot_scorer is not None:
            v3_state["slot_scorer"] = self._slot_scorer.to_dict()
        (directory / "v3_state.json").write_text(json.dumps(v3_state, indent=2))

        # Batch budget state (standalone file for easy inspection)
        (directory / "batch_budget.json").write_text(
            json.dumps(self._batch_budget.to_dict(), indent=2)
        )

        # Program store
        (directory / "programs.json").write_text(json.dumps(self._program_store, indent=2))

        # branch state
        self._branch_manager.save_to(directory / "branches.json")
        (directory / "code_index.json").write_text(json.dumps(self._code_to_program_id, indent=2))

    def load_state(self, directory: Path) -> None:
        """Load engine state from checkpoint."""
        from esn.core.persistence import (
            ArchiveStore,
            KnowledgeStore,
            NoveltyStore,
            OperatorCreditStore,
            SearchStateStore,
        )

        state_path = directory / "search_state.json"
        if state_path.exists():
            self.state = SearchStateStore.load(state_path)

        elite_path = directory / "elite.json"
        if elite_path.exists():
            self.elite_archive = ArchiveStore.load_elite(elite_path)

        frontier_path = directory / "frontier.json"
        if frontier_path.exists():
            self.frontier_archive = ArchiveStore.load_frontier(frontier_path)

        credit_path = directory / "credit.json"
        if credit_path.exists():
            self.credit_model = OperatorCreditStore.load(credit_path)

        knowledge_path = directory / "knowledge.json"
        if knowledge_path.exists() and self.knowledge:
            self.knowledge = KnowledgeStore.load(
                knowledge_path,
                self.config,
                getattr(self.knowledge, "_embedder", None),
                getattr(self.knowledge, "_embedding_dim", 1024),
            )

        novelty_path = directory / "novelty.json"
        if novelty_path.exists() and self.knowledge:
            self.novelty_computer = NoveltyStore.load(
                novelty_path,
                self.knowledge,
                self.config,
            )
            # Re-wire observation providers lost during deserialization
            self.novelty_computer.set_observation_providers([V3ObservationProvider(self)])

        v3_path = directory / "v3_state.json"
        if v3_path.exists():
            v3 = json.loads(v3_path.read_text())
            self.generation = v3["generation"]
            self._best_score = v3["best_score"]
            self._best_code = v3["best_code"]
            self._seed_evaluated = v3.get("seed_evaluated", False)
            self._consecutive_failures = v3["consecutive_failures"]
            self._search_temperature = v3["search_temperature"]
            self._seen_hashes = set(v3.get("seen_hashes", []))
            self._last_failed_code = v3.get("last_failed_code")
            self._last_error_context = v3.get("last_error_context", "")
            self._recent_attempt_log = v3.get("recent_attempt_log", [])
            self._breakthrough_cooldown = v3.get("breakthrough_cooldown", 0)
            if "family_tracker" in v3:
                self._family_tracker = FamilyTracker.from_dict(v3["family_tracker"])
            if self._slot_scorer is not None and "slot_scorer" in v3:
                self._slot_scorer = BatchSlotScorer.from_dict(v3["slot_scorer"])
            if "batch_size" in v3:
                self._batch_size = v3["batch_size"]
            if "batch_budget" in v3:
                self._batch_budget = BatchBudgetController.from_dict(v3["batch_budget"])
            self._last_batch_decision = v3.get("last_batch_decision")

        programs_path = directory / "programs.json"
        if programs_path.exists():
            self._program_store = json.loads(programs_path.read_text())

        # branch state
        branches_path = directory / "branches.json"
        if branches_path.exists():
            self._branch_manager = BranchManager.load_from(branches_path)
        code_index_path = directory / "code_index.json"
        if code_index_path.exists():
            self._code_to_program_id = json.loads(code_index_path.read_text())
