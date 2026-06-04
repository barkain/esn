# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Family-level statistics tracker for solver programs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FamilyStats:
    name: str
    best_score: float = 0.0
    attempt_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    recent_scores: list[float] = field(default_factory=list)  # last 5
    plateau_gens: int = 0  # gens since last improvement in this family
    representative_summary: str = ""  # docstring of best program in family
    last_score: float = 0.0  # most recent score


class FamilyTracker:
    def __init__(self) -> None:
        self._families: dict[str, FamilyStats] = {}

    def record(self, family: str, score: float, success: bool, summary: str = "") -> None:
        """Record a generation result for a family."""
        if family not in self._families:
            self._families[family] = FamilyStats(name=family)
        fs = self._families[family]
        fs.attempt_count += 1
        if success:
            fs.success_count += 1
        else:
            fs.failure_count += 1
        fs.last_score = score
        fs.recent_scores.append(score)
        if len(fs.recent_scores) > 5:
            fs.recent_scores = fs.recent_scores[-5:]
        if score > fs.best_score:
            fs.best_score = score
            fs.plateau_gens = 0
            if summary:
                fs.representative_summary = summary
        else:
            fs.plateau_gens += 1

    def get_summary(self) -> list[str]:
        """Return one-line summaries for each known family, sorted by best score."""
        result: list[str] = []
        for fs in sorted(self._families.values(), key=lambda f: f.best_score, reverse=True):
            status = f"plateau {fs.plateau_gens} gens" if fs.plateau_gens > 0 else "improving"
            line = (
                f"{fs.name}: best={fs.best_score:.4f}, "
                f"{fs.attempt_count} attempts ({fs.success_count} ok, {fs.failure_count} fail), "
                f"{status}, last: {fs.last_score:.2f}"
            )
            if fs.representative_summary:
                line += f" | {fs.representative_summary[:80]}"
            result.append(line)
        return result

    def get_stats(self, family: str) -> FamilyStats | None:
        """Get stats for a specific family."""
        return self._families.get(family)

    def to_dict(self) -> dict:
        out: dict = {}
        for name, fs in self._families.items():
            out[name] = {
                "best_score": fs.best_score,
                "attempt_count": fs.attempt_count,
                "success_count": fs.success_count,
                "failure_count": fs.failure_count,
                "recent_scores": fs.recent_scores,
                "plateau_gens": fs.plateau_gens,
                "representative_summary": fs.representative_summary,
                "last_score": fs.last_score,
            }
        return out

    @classmethod
    def from_dict(cls, data: dict) -> FamilyTracker:
        tracker = cls()
        for name, vals in data.items():
            tracker._families[name] = FamilyStats(
                name=name,
                best_score=vals.get("best_score", 0.0),
                attempt_count=vals.get("attempt_count", 0),
                success_count=vals.get("success_count", 0),
                failure_count=vals.get("failure_count", 0),
                recent_scores=vals.get("recent_scores", []),
                plateau_gens=vals.get("plateau_gens", 0),
                representative_summary=vals.get("representative_summary", ""),
                last_score=vals.get("last_score", 0.0),
            )
        return tracker
