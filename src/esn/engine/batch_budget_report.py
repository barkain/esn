# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Pure markdown formatter for the Adaptive Batching report section.

This module has no side effects and does not mutate controller state; it only
reads from a ``BatchBudgetController`` to produce a list of markdown lines.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esn.engine.batch_budget import BatchBudgetController


def render_adaptive_batching_section(
    controller: "BatchBudgetController",
    nominal_batch_size: int,
    resume_restored_batch_size: bool,
) -> list[str]:
    """Return markdown lines for an '## Adaptive Batching' report section.

    The output starts and ends with empty strings so that it can be directly
    extended into a list of report lines without manual padding.
    """
    budgeted = controller._total_slot_budget
    spent = controller._slots_spent
    saved = controller.slots_remaining

    shrink = sum(1 for d in controller._decisions if d.actual < nominal_batch_size)
    expand = sum(1 for d in controller._decisions if d.actual > nominal_batch_size)

    lines: list[str] = []
    lines.append("")
    lines.append("## Adaptive Batching")
    lines.append("")
    lines.append(f"- Nominal batch size: **{nominal_batch_size}**")
    lines.append(f"- Total slots: budgeted **{budgeted}** / spent **{spent}** / saved **{saved}**")
    lines.append(f"- Shrink events: **{shrink}** (effective < nominal)")
    lines.append(f"- Expand events: **{expand}** (effective > nominal)")
    if resume_restored_batch_size:
        lines.append("- Resume: restored batch size from checkpoint (CLI override ignored)")
        lines.append(
            "- Note: counters and histogram above cover the current resumed "
            "segment only (pre-resume history is not preserved across "
            "checkpoints)."
        )

    history = controller._history
    if not history:
        lines.append("- No generations recorded yet.")
        lines.append("")
        return lines

    counts: Counter[int] = Counter(y.batch_size for y in history)

    lines.append("")
    lines.append("### Effective-batch histogram")
    lines.append("")
    lines.append("| Size | Count |")
    lines.append("| --- | --- |")
    for size in sorted(counts):
        lines.append(f"| {size} | {counts[size]} |")
    lines.append("")

    return lines
