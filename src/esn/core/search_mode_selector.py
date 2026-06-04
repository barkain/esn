# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Search mode selector for ESN core."""

from __future__ import annotations

from esn.core.enums import SearchMode
from esn.core.models import SearchState


class SearchModeSelector:
    """Selects search mode based on stagnation, failure rate, and spectral signals."""

    def select_mode(
        self,
        state: SearchState,
        spectral_summary: dict[str, float] | None = None,
    ) -> SearchMode:
        scores = state.recent_scores

        # Failure rate from recent scores (score <= 0 or empty means failure)
        if scores:
            failures = sum(1 for s in scores if s <= 0)
            failure_rate = failures / len(scores)
        else:
            failure_rate = 0.0

        recent_ops = state.recent_operators[-8:]
        op_diversity = len(set(recent_ops)) / max(1, len(recent_ops)) if recent_ops else 1.0

        # High failure rate
        if failure_rate > 0.7:
            return SearchMode.REPAIR

        # Moderate failure with some stagnation
        if failure_rate > 0.5 and state.stagnation_counter > 1:
            return SearchMode.RECOVER

        # Improving scores (last 3 trending up)
        if len(scores) >= 3:
            last3 = scores[-3:]
            if last3[0] < last3[1] < last3[2]:
                return SearchMode.EXPLOIT

        # Keep exploit longer; only explore under prolonged stagnation,
        # low operator diversity, and a genuinely diverse frontier.
        if (
            state.stagnation_counter > 8
            and op_diversity < 0.5
            and state.frontier_distinct_count >= 3
        ):
            return SearchMode.EXPLORE

        # Spectral complexity growing
        if spectral_summary and spectral_summary.get("complexity_trend", 0) > 0:
            return SearchMode.COMPRESS

        return SearchMode.EXPLOIT
