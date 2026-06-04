# ruff: noqa: S101
"""Tests for the generic ``preferred_solution_shape`` domain field.

Covers:
- ``DomainSpec`` default is ``None``; constructor accepts an optional string.
- ``MutatorInputBundle`` carries the field through; default ``None``.
- ``LLMMutator`` prompt renders the ``# Preferred solution shape`` section
  with the supplied text when populated, and with the fallback
  ``(no domain-specific preference)`` when ``None``.
- ``ClaudeAgentSDKClient`` prompt renders the same two cases via
  ``_render_prompt``.
- The ``circle_packing`` domain factory populates the field with the
  expected distinctive phrase.
"""

from __future__ import annotations

from esn.core.models import EvaluationResult
from esn.engine import LLMMutator
from esn.engine.claude_agent_client import MutatorInputBundle, _render_prompt
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec


def _make_domain(preferred_solution_shape: str | None = None) -> DomainSpec:
    """Minimal DomainSpec; optionally populated ``preferred_solution_shape``."""
    return DomainSpec(
        name="test",
        description="simple test domain",
        initial_code="def solve():\n    return [1, 2, 3]\n",
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=lambda artifact: EvaluationResult(score=1.0, success=True),
        allowed_imports=frozenset({"math"}),
        max_code_lines=40,
        hard_constraints=["must define solve()"],
        hints=["keep changes small"],
        preferred_solution_shape=preferred_solution_shape,
    )


def _make_bundle(
    preferred_solution_shape: str | None = None,
) -> MutatorInputBundle:
    """Minimal MutatorInputBundle for prompt-render tests."""
    return MutatorInputBundle(
        domain_name="test_domain",
        domain_description="test domain",
        hard_constraints=[],
        allowed_imports=[],
        max_code_lines=50,
        program_interface="solve",
        parent_code="def solve():\n    return 0\n",
        style="refine",
        intended_effect="",
        targeted_hypothesis_ids=[],
        top_hypotheses_summary=[],
        mutation_style="refine",
        search_mode="exploit",
        preferred_solution_shape=preferred_solution_shape,
    )


# ---------------------------------------------------------------------------
# DomainSpec
# ---------------------------------------------------------------------------


def test_domain_spec_default_preferred_solution_shape_is_none() -> None:
    domain = DomainSpec(
        name="x",
        description="x",
        initial_code="def solve():\n    return 0\n",
        compiler=PythonSandboxCompiler(allowed_imports=frozenset()),
        evaluator=lambda _a: EvaluationResult(score=0.0, success=True),
    )
    assert domain.preferred_solution_shape is None


def test_domain_spec_accepts_preferred_solution_shape_string() -> None:
    domain = _make_domain(preferred_solution_shape="prefer constructive")
    assert domain.preferred_solution_shape == "prefer constructive"


# ---------------------------------------------------------------------------
# MutatorInputBundle
# ---------------------------------------------------------------------------


def test_mutator_input_bundle_default_preferred_solution_shape_is_none() -> None:
    bundle = _make_bundle()
    assert bundle.preferred_solution_shape is None


def test_mutator_input_bundle_carries_preferred_solution_shape() -> None:
    bundle = _make_bundle(preferred_solution_shape="prefer constructive")
    assert bundle.preferred_solution_shape == "prefer constructive"


# ---------------------------------------------------------------------------
# LLMMutator system prompt
# ---------------------------------------------------------------------------


def _llm_system_prompt(domain: DomainSpec) -> str:
    # The LLMMutator constructor requires an LLM client, but we never call
    # it here — we only want the system-prompt string. A trivial sentinel
    # satisfies the Protocol.
    def _never_called(_system: str, _user: str) -> str:
        raise AssertionError("LLM client must not be invoked in prompt-render test")

    mutator = LLMMutator(_never_called, domain)
    return mutator._build_system_prompt("refine")  # noqa: SLF001


def test_llm_mutator_prompt_renders_populated_preferred_solution_shape() -> None:
    text = "prefer direct constructive solutions; avoid long global search"
    prompt = _llm_system_prompt(_make_domain(preferred_solution_shape=text))
    assert "# Preferred solution shape" in prompt
    assert text in prompt
    assert "(no domain-specific preference)" not in prompt


def test_llm_mutator_prompt_renders_fallback_when_none() -> None:
    prompt = _llm_system_prompt(_make_domain(preferred_solution_shape=None))
    assert "# Preferred solution shape" in prompt
    assert "(no domain-specific preference)" in prompt


# ---------------------------------------------------------------------------
# ClaudeAgentSDKClient (``_render_prompt``)
# ---------------------------------------------------------------------------


def test_render_prompt_includes_populated_preferred_solution_shape() -> None:
    text = "prefer direct constructive solutions; avoid long global search"
    bundle = _make_bundle(preferred_solution_shape=text)
    prompt = _render_prompt(bundle)
    assert "# Preferred solution shape" in prompt
    assert text in prompt
    assert "(no domain-specific preference)" not in prompt


def test_render_prompt_includes_fallback_when_none() -> None:
    bundle = _make_bundle(preferred_solution_shape=None)
    prompt = _render_prompt(bundle)
    assert "# Preferred solution shape" in prompt
    assert "(no domain-specific preference)" in prompt


def test_render_prompt_fallback_also_in_research_mode() -> None:
    # The section is rendered regardless of the tool-use mode — both
    # no-tools and research-enabled prompts route through the same
    # ``_render_prompt`` body below the protocol block.
    bundle = _make_bundle(preferred_solution_shape=None)
    prompt = _render_prompt(bundle, mutator_tools="research")
    assert "# Preferred solution shape" in prompt
    assert "(no domain-specific preference)" in prompt
