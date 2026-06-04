# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""RMT-based hypothesis deduplication for ESN knowledge banks.

Uses the spike subspace V_k (already computed by the spectral pipeline) to
identify near-duplicate hypotheses.  Only hypotheses with nontrivial certainty
(w_i = 2|c_i - 0.5| > threshold) are considered — unresolved hypotheses
(c ≈ 0.5) have near-zero weight and their spike-space coordinates are
dominated by centering artifacts.

Algorithm:
  1. Run spectral pipeline → V_k, mean_row
  2. Build certainty-weighted rows k_i = w_i * e_i, center: k_tilde_i = k_i - mean_row
  3. Project into spike space: z_i = V_k^T @ k_tilde_i
  4. Compute residual norms r_i = ||k_tilde_i - V_k V_k^T k_tilde_i||
  5. For each pair (i, j) passing hard gates (polarity, tag overlap, cosine
     in spike space, residual below threshold), add a union-find edge
  6. Within each cluster, merge via Bayesian pseudo-count pooling
  7. Keep the representative with highest n_obs, then highest |c - 0.5|
"""

import logging
from dataclasses import dataclass

import numpy as np

from .spectral_models import HypothesisRecord, HypothesisStatus
from .spectral import run_spectral_pipeline
from .utils import cosine_similarity

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DedupConfig:
    """Parameters for RMT-based dedup."""

    # Minimum certainty weight to be eligible for dedup.
    # w_i = 2|c_i - 0.5|; threshold 0.1 means c < 0.45 or c > 0.55.
    certainty_threshold: float = 0.1

    # Cosine similarity threshold in spike space.
    spike_cosine_threshold: float = 0.97

    # Concept-tag Jaccard overlap threshold.
    tag_jaccard_threshold: float = 0.7

    # Residual percentile gate — only merge if both hypotheses' residuals
    # are below this percentile of the eligible set.
    residual_percentile: float = 75.0

    # Raw embedding cosine threshold — used as fallback when spike space
    # has fewer than min_spikes_for_projection dimensions.
    embedding_cosine_threshold: float = 0.92

    # Minimum spike count for spike-space projection to be meaningful.
    # Below this, fall back to raw embedding cosine.
    min_spikes_for_projection: int = 3


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polarity(c: float) -> str:
    """Classify hypothesis polarity from confidence."""
    if c > 0.5:
        return "confirmed"
    elif c < 0.5:
        return "refuted"
    return "unresolved"


def _normalize_tag(tag: str) -> str:
    """Normalize a concept tag: lowercase, underscores→spaces, strip."""
    return tag.lower().replace("_", " ").strip()


def _tag_jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two concept-tag lists (normalized)."""
    sa = {_normalize_tag(t) for t in a}
    sb = {_normalize_tag(t) for t in b}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _spike_cosine(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """Cosine similarity between two spike-space projections."""
    ni = np.linalg.norm(z_i)
    nj = np.linalg.norm(z_j)
    if ni < 1e-12 or nj < 1e-12:
        return 0.0
    return float(np.dot(z_i, z_j) / (ni * nj))


# ---------------------------------------------------------------------------
# Merge result
# ---------------------------------------------------------------------------


@dataclass
class MergeCluster:
    """A cluster of near-duplicate hypotheses to be merged."""

    representative_id: str
    representative_text: str
    merged_ids: list[str]
    merged_texts: list[str]
    merged_confidence: float
    merged_n_obs: int
    cluster_size: int


@dataclass
class DedupResult:
    """Full result of a dedup pass."""

    total_hypotheses: int
    eligible_hypotheses: int
    skipped_unresolved: int
    num_spikes: int
    clusters: list[MergeCluster]
    total_merged: int  # hypotheses absorbed into representatives
    surviving: int  # eligible - total_merged

    def summary(self) -> str:
        lines = [
            f"Hypotheses: {self.total_hypotheses} total, {self.eligible_hypotheses} eligible "
            f"({self.skipped_unresolved} skipped — unresolved/low certainty)",
            f"Spike subspace: {self.num_spikes} spikes",
            f"Duplicate clusters: {len(self.clusters)}",
            f"Merged away: {self.total_merged} hypotheses",
            f"Surviving eligible: {self.surviving}",
        ]
        if self.clusters:
            lines.append("")
            lines.append("Top clusters (by size):")
            for cl in sorted(self.clusters, key=lambda c: c.cluster_size, reverse=True)[:10]:
                lines.append(
                    f"  [{cl.cluster_size}] rep='{cl.representative_text[:80]}...' "
                    f"(c={cl.merged_confidence:.3f}, n={cl.merged_n_obs})"
                )
                for mid, mtxt in zip(cl.merged_ids[:3], cl.merged_texts[:3]):
                    lines.append(f"       <- '{mtxt[:70]}...'")
                if cl.cluster_size > 4:
                    lines.append(f"       ... and {cl.cluster_size - 4} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core dedup
# ---------------------------------------------------------------------------


def run_dedup(
    hypotheses: list[HypothesisRecord],
    config: DedupConfig | None = None,
) -> DedupResult:
    """Run RMT spike-space deduplication on a list of hypotheses.

    Only active hypotheses with certainty > config.certainty_threshold are
    considered.  Returns a DedupResult describing the proposed merges.
    """
    if config is None:
        config = DedupConfig()

    total = len(hypotheses)
    active = [h for h in hypotheses if h.status == HypothesisStatus.ACTIVE]

    # Filter to hypotheses with nontrivial certainty
    eligible = []
    for h in active:
        w = 2.0 * abs(h.confidence - 0.5)
        if w > config.certainty_threshold:
            eligible.append(h)

    skipped = len(active) - len(eligible)
    n = len(eligible)

    if n < 2:
        return DedupResult(
            total_hypotheses=total,
            eligible_hypotheses=n,
            skipped_unresolved=skipped,
            num_spikes=0,
            clusters=[],
            total_merged=0,
            surviving=n,
        )

    # --- Step 1: Spectral pipeline on ELIGIBLE hypotheses only ---
    # Running on the full active set would drown the signal in zero-weight
    # rows from untested hypotheses (c=0.5 → w=0).
    spectral_state = run_spectral_pipeline(eligible)

    num_spikes = 0
    use_spike_space = False

    if spectral_state is not None and spectral_state.V_k is not None:
        num_spikes = spectral_state.num_spikes

    if num_spikes >= config.min_spikes_for_projection:
        use_spike_space = True
        log.info("Using spike-space projection (%d spikes)", num_spikes)
    else:
        log.info(
            "Spike count (%d) < min_spikes_for_projection (%d) — "
            "falling back to raw embedding cosine (threshold=%.2f)",
            num_spikes,
            config.min_spikes_for_projection,
            config.embedding_cosine_threshold,
        )

    V_k = spectral_state.V_k if spectral_state else None
    mean_row = spectral_state.mean_row if spectral_state else None

    # --- Step 2: Project eligible hypotheses into spike space (if enough spikes) ---
    z: np.ndarray | None = None
    residual_norms: np.ndarray | None = None
    residual_threshold = float("inf")

    if use_spike_space:
        z = np.zeros((n, num_spikes))
        residual_norms = np.zeros(n)

        for i, h in enumerate(eligible):
            w_i = 2.0 * abs(h.confidence - 0.5)
            k_i = w_i * h.embedding
            k_tilde_i = k_i - mean_row

            z[i] = V_k.T @ k_tilde_i

            proj = V_k @ z[i]
            residual = k_tilde_i - proj
            residual_norms[i] = np.linalg.norm(residual)

        residual_threshold = float(np.percentile(residual_norms, config.residual_percentile))

    # --- Step 3: Pairwise hard gates + union-find ---
    uf = UnionFind(n)

    # Pre-compute polarities
    polarities = [_polarity(h.confidence) for h in eligible]

    edges_checked = 0
    edges_passed = 0

    for i in range(n):
        # Residual gate (only in spike-space mode)
        if use_spike_space and residual_norms[i] > residual_threshold:
            continue
        for j in range(i + 1, n):
            if use_spike_space and residual_norms[j] > residual_threshold:
                continue

            edges_checked += 1

            # Gate 1: same polarity
            if polarities[i] != polarities[j]:
                continue

            # Gate 2: tag Jaccard overlap
            jac = _tag_jaccard(eligible[i].concepts, eligible[j].concepts)
            if jac < config.tag_jaccard_threshold:
                continue

            # Gate 3: similarity — spike-space cosine or raw embedding cosine
            if use_spike_space:
                cos = _spike_cosine(z[i], z[j])
                if cos < config.spike_cosine_threshold:
                    continue
            else:
                cos = cosine_similarity(eligible[i].embedding, eligible[j].embedding)
                if cos < config.embedding_cosine_threshold:
                    continue

            # All gates passed — union
            uf.union(i, j)
            edges_passed += 1

    log.info("Edges checked: %d, passed: %d", edges_checked, edges_passed)

    # --- Step 4: Extract clusters ---
    clusters_map: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters_map.setdefault(root, []).append(i)

    # Only keep clusters with 2+ members
    merge_clusters: list[MergeCluster] = []
    total_merged = 0

    for members in clusters_map.values():
        if len(members) < 2:
            continue

        # Pick representative: highest n_obs, then highest certainty
        members_sorted = sorted(
            members,
            key=lambda idx: (eligible[idx].n_obs, abs(eligible[idx].confidence - 0.5)),
            reverse=True,
        )
        rep_idx = members_sorted[0]
        absorbed = members_sorted[1:]

        # Bayesian pseudo-count pooling
        alpha_sum = sum(eligible[idx].confidence * eligible[idx].n_obs for idx in members)
        beta_sum = sum((1.0 - eligible[idx].confidence) * eligible[idx].n_obs for idx in members)
        n_merged = alpha_sum + beta_sum
        c_merged = alpha_sum / n_merged if n_merged > 0 else 0.5

        merge_clusters.append(
            MergeCluster(
                representative_id=eligible[rep_idx].id,
                representative_text=eligible[rep_idx].text,
                merged_ids=[eligible[idx].id for idx in absorbed],
                merged_texts=[eligible[idx].text for idx in absorbed],
                merged_confidence=c_merged,
                merged_n_obs=int(round(n_merged)),
                cluster_size=len(members),
            )
        )
        total_merged += len(absorbed)

    return DedupResult(
        total_hypotheses=total,
        eligible_hypotheses=n,
        skipped_unresolved=skipped,
        num_spikes=num_spikes,
        clusters=merge_clusters,
        total_merged=total_merged,
        surviving=n - total_merged,
    )


# ---------------------------------------------------------------------------
# Apply merges to a knowledge bank (mutates in place)
# ---------------------------------------------------------------------------


def apply_dedup(
    hypotheses: list[HypothesisRecord],
    result: DedupResult,
) -> int:
    """Apply dedup merges to a hypothesis list in place.

    For each cluster:
    - Updates the representative with pooled confidence/n_obs and merged concepts
    - Archives absorbed hypotheses

    Returns number of hypotheses archived.
    """
    by_id = {h.id: h for h in hypotheses}
    archived = 0

    for cluster in result.clusters:
        rep = by_id.get(cluster.representative_id)
        if rep is None:
            continue

        # Update representative with pooled stats
        rep.confidence = cluster.merged_confidence
        rep.n_obs = cluster.merged_n_obs

        # Merge concepts from all absorbed
        all_concepts = set(rep.concepts)
        for mid in cluster.merged_ids:
            absorbed = by_id.get(mid)
            if absorbed is None:
                continue
            all_concepts.update(absorbed.concepts)
            absorbed.status = HypothesisStatus.ARCHIVED
            archived += 1

        rep.concepts = sorted(all_concepts)

    return archived
