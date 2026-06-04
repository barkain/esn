"""Phase 1 follow-up: engine _select_mode honors BBP actionable spikes.

Drives the _select_mode override logic with lightweight stubs so we can
assert it flips EXPLORE→EXPLOIT on BBP-only structure AND respects the
undersampled guardrail.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

from esn.core.enums import SearchMode
from esn.engine.engine import ESNEngine


@dataclass
class _Spike:
    above_gate: bool


def _make_engine(
    legacy_spikes: int,
    bbp_actionable_above_gate: int,
    undersampled: bool,
    stagnation: int,
    forced_mode: SearchMode,
) -> ESNEngine:
    eng = ESNEngine.__new__(ESNEngine)  # bypass heavy __init__
    eng._breakthrough_cooldown = 0
    eng.state = SimpleNamespace(stagnation_counter=stagnation, best_score=0.0)

    eng.mode_selector = MagicMock()
    eng.mode_selector.select_mode.return_value = forced_mode

    report = SimpleNamespace(
        spikes=[_Spike(above_gate=True) for _ in range(bbp_actionable_above_gate)],
        undersampled=undersampled,
    )
    spectral_state = SimpleNamespace(num_spikes=legacy_spikes, S1=0.0, S2=0.0, erank=5.0)
    eng.novelty_computer = SimpleNamespace(spectral_state=spectral_state, spectral_report=report)
    return eng


class TestModeSelectionBBPGuardrails:
    def test_bbp_actionable_flips_explore_to_exploit_when_well_sampled(self):
        eng = _make_engine(
            legacy_spikes=0,
            bbp_actionable_above_gate=2,
            undersampled=False,
            stagnation=2,
            forced_mode=SearchMode.EXPLORE,
        )
        assert eng._select_mode() == SearchMode.EXPLOIT

    def test_bbp_actionable_does_not_flip_when_undersampled(self):
        eng = _make_engine(
            legacy_spikes=0,
            bbp_actionable_above_gate=2,
            undersampled=True,
            stagnation=2,
            forced_mode=SearchMode.EXPLORE,
        )
        assert eng._select_mode() == SearchMode.EXPLORE

    def test_legacy_spikes_still_flip_even_when_undersampled(self):
        eng = _make_engine(
            legacy_spikes=1,
            bbp_actionable_above_gate=0,
            undersampled=True,
            stagnation=2,
            forced_mode=SearchMode.EXPLORE,
        )
        assert eng._select_mode() == SearchMode.EXPLOIT

    def test_no_structure_stays_explore(self):
        eng = _make_engine(
            legacy_spikes=0,
            bbp_actionable_above_gate=0,
            undersampled=False,
            stagnation=2,
            forced_mode=SearchMode.EXPLORE,
        )
        assert eng._select_mode() == SearchMode.EXPLORE

    def test_high_stagnation_does_not_flip(self):
        eng = _make_engine(
            legacy_spikes=0,
            bbp_actionable_above_gate=2,
            undersampled=False,
            stagnation=10,  # >= 6 → no flip
            forced_mode=SearchMode.EXPLORE,
        )
        # At stagnation>=4 and no legacy S1, also forces EXPLORE in second override
        assert eng._select_mode() == SearchMode.EXPLORE
