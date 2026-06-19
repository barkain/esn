# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""User-facing one-call API for the ESN novelty generator.

This module is the thin, friendly surface that turns the research engine
(``esn.engine.engine.ESNEngine``) into a ready-to-use general-purpose novelty
generator. The three entry points are:

- :class:`MockMutator` -- a zero-dependency mutator (no API key, no network)
  that always returns a fixed candidate. Use it for CI / smoke tests and as
  the default mutator in :func:`run`, so the whole loop runs offline.
- :func:`run` -- construct an :class:`~esn.engine.engine.ESNEngine`, drive it
  for ``generations`` generations, and return a small :class:`RunResult`.
- :func:`make_llm_mutator` -- build a real LLM-backed
  :class:`~esn.engine.mutator.LLMMutator` for OpenAI / Anthropic models, with the
  provider chosen from the model-name prefix.

Heavy / optional dependencies (sentence-transformers for ``[novelty]``, the
OpenAI / Anthropic SDKs for ``[llm]``) are imported lazily inside the
functions that need them, so ``import esn.api`` stays cheap and the offline
path never touches them.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from esn.engine.models import MutationResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from esn.engine.domain import DomainSpec
    from esn.engine.models import MutationContext
    from esn.engine.protocols import Analyzer, Mutator, Predictor, ProgramObject

# Library logger. ESN follows the standard library convention: it emits records
# on the ``esn`` logger and never configures handlers itself — the application
# decides whether/how to display them (e.g. ``logging.basicConfig`` or
# ``logging.getLogger("esn").setLevel(logging.INFO)``).
logger = logging.getLogger("esn")


# ---------------------------------------------------------------------------
# MockMutator: no-API mutator for CI / smoke tests
# ---------------------------------------------------------------------------

# A fixed, trivially-valid solver. It is intentionally NOT a good solution --
# the point of MockMutator is to exercise the full mutate -> compile -> eval
# loop deterministically and offline, not to win the benchmark. The body must
# satisfy whatever the domain evaluator expects from ``solve()``; this default
# returns an empty packing, which the circle-packing evaluator rejects (score
# 0.0) but which still drives the engine end-to-end without an API key.
_MOCK_CANDIDATE_CODE = (
    "def solve():\n"
    "    # Fixed offline candidate emitted by esn.api.MockMutator.\n"
    "    return [], []\n"
)


class MockMutator:
    """Offline mutator implementing the ``Mutator`` protocol.

    ``mutate(parents, style, context) -> MutationResult`` always returns the
    same fixed candidate with ``success=True``. No LLM, no network, no API key.
    """

    def __init__(self, code: str = _MOCK_CANDIDATE_CODE) -> None:
        self._code = code

    def mutate(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> MutationResult:
        return MutationResult(
            code=self._code,
            success=True,
            metadata={"style": style, "mutator": "mock"},
        )


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Summary of an :func:`run` invocation.

    Attributes:
        best_code: Source of the best program found (``engine._best_code``).
        best_score: Score of the best program (``engine._best_score``).
        generations: Number of generations actually executed.
        history: Per-generation summaries (gen index, best score so far, and
            the best candidate's score / success for that generation).
    """

    best_code: str
    best_score: float
    generations: int
    history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# run(): the one-call driver
# ---------------------------------------------------------------------------


def run(
    domain: DomainSpec,
    *,
    generations: int = 10,
    batch_size: int = 4,
    mutator: Mutator | None = None,
    analyzer: Analyzer | None = None,
    predictor: Predictor | None = None,
    tuner: Any | None = None,
    enable_divergence: bool = False,
    seed: int = 42,
    enable_recombination: bool = False,
    spectral_threshold_mode: str = "empirical",
) -> RunResult:
    """Run the ESN program-search engine on ``domain`` and return a summary.

    ESN's defining behavior is novelty-guided selection (the epistemic +
    spectral ``N_sp`` signal). That machinery is driven by *hypotheses*, which
    are produced by an **analyzer** — so novelty is not a separate on/off flag:
    pass an ``analyzer`` (e.g. from :func:`make_analyzer`) and the full novelty
    stack is wired automatically. Without an analyzer the engine has no
    hypothesis source, the novelty signal cannot form, and the search runs
    without novelty (it still uses score, archive, branch, family, and
    search-mode heuristics); ``run`` **warns loudly** when that happens so the
    change is never silent.

    Args:
        domain: The :class:`~esn.engine.domain.DomainSpec` to search over.
        generations: How many generations to run (each generation produces
            ``batch_size`` candidates).
        batch_size: Candidates per generation. When > 1, the engine's
            ``run_batch_generation`` path is used; when 1, ``run_generation``.
        mutator: A ``Mutator``-protocol object that proposes candidates.
            Defaults to :class:`MockMutator`, so the loop runs offline with no
            API key.
        analyzer: An ``Analyzer``-protocol object (e.g. :func:`make_analyzer`)
            that turns evaluated candidates into hypotheses. Supplying it
            **activates** the epistemic-spectral novelty machinery. ``None``
            (the default) means fitness-only search, with a loud warning.
        predictor: Optional ``Predictor``-protocol object (e.g.
            :func:`make_predictor`) adding a prediction-surprise term to the
            epistemic novelty. Inert unless an ``analyzer`` is also supplied.
        tuner: Optional ``Tuner`` (e.g.
            :class:`~esn.engine.tuner.ParameterTuner`) — evaluator-guided
            continuous-parameter polish of candidates. Helps float-literal-driven
            problems; a safe no-op on combinatorial/structural ones. ``None`` =
            off.
        enable_divergence: Experimental, off by default. When True, force a
            parentless structural-escape ("diverge") slot on stagnation. A
            controlled study showed no escape benefit on a weak model; kept
            opt-in.
        seed: Seed for the engine's RNG (reproducibility).
        enable_recombination: When True, let the engine recombine
            high-performing branches (an extra exploration operator). Off by
            default.
        spectral_threshold_mode: How the spectral pipeline picks its
            spike-detection threshold — ``"empirical"`` (shuffle-null, default),
            ``"mp"`` (Marchenko–Pastur edge), or ``"hybrid"``. Only relevant
            when an ``analyzer`` activates novelty.

    Returns:
        A :class:`RunResult` with ``best_code``, ``best_score``,
        ``generations`` (actually executed), and a per-generation ``history``.
    """
    from esn.engine.engine import ESNEngine

    if mutator is None:
        mutator = MockMutator()

    # Novelty is intrinsic to ESN, not an opt-in flag — but it can only form
    # when an analyzer supplies hypotheses. With an analyzer we build the full
    # stack; without one the machinery would be inert, so we skip it (avoiding a
    # heavy embedder load) and warn loudly rather than silently no-op.
    knowledge = None
    novelty_computer = None
    config = None
    if analyzer is not None:
        knowledge, novelty_computer, config = _build_novelty_stack(seed, spectral_threshold_mode)
    else:
        warnings.warn(
            "esn.run() was called without an `analyzer`, so ESN's "
            "epistemic-spectral novelty machinery is INACTIVE: no hypotheses "
            "are generated and N_sp / N_ep stay 0, so the search runs without "
            "novelty (it still uses score, archive, branch, family, and "
            "search-mode heuristics; this is not the full ESN algorithm). Pass "
            "analyzer=esn.make_analyzer(model=...) (and optionally "
            "predictor=esn.make_predictor(model=...)) to enable novelty-guided "
            "search.",
            RuntimeWarning,
            stacklevel=2,
        )

    engine = ESNEngine(
        domain=domain,
        mutator=mutator,
        predictor=predictor,
        analyzer=analyzer,
        knowledge=knowledge,
        novelty_computer=novelty_computer,
        config=config,
        seed=seed,
        batch_size=batch_size,
        total_generations=generations,
        enable_recombination=enable_recombination,
        tuner=tuner,
        enable_divergence=enable_divergence,
    )

    logger.info(
        "ESN run starting: domain=%s | generations=%d | batch_size=%d | seed=%d | novelty=%s",
        domain.name,
        generations,
        batch_size,
        seed,
        "on" if analyzer is not None else "off (no analyzer)",
    )

    history: list[dict[str, Any]] = []
    completed = 0
    for gen in range(1, generations + 1):
        if batch_size > 1:
            records = engine.run_batch_generation()
            successful = [r for r in records if r.success]
            best_record = (
                max(successful, key=lambda r: r.score)
                if successful
                else (records[0] if records else None)
            )
            n_candidates = len(records)
            n_ok = len(successful)
        else:
            best_record = engine.run_generation()
            n_candidates = 1
            n_ok = 1 if (best_record is not None and best_record.success) else 0

        completed = gen
        gen_score = 0.0
        gen_success = False
        if best_record is not None:
            gen_score = best_record.score if best_record.score is not None else 0.0
            gen_success = best_record.success
        history.append(
            {
                "generation": gen,
                "best_score": engine._best_score,
                "gen_score": gen_score,
                "gen_success": gen_success,
            }
        )
        run_best = engine._best_score if engine._best_score is not None else float("nan")
        logger.info(
            "generation %d/%d | candidates=%d ok=%d | gen_best=%.6g | run_best=%.6g",
            gen,
            generations,
            n_candidates,
            n_ok,
            gen_score,
            run_best,
        )

    final_best = engine._best_score if engine._best_score is not None else float("nan")
    logger.info("ESN run complete: %d generation(s) | best_score=%.6g", completed, final_best)
    return RunResult(
        best_code=engine._best_code,
        best_score=engine._best_score,
        generations=completed,
        history=history,
    )


def _build_novelty_stack(seed: int, spectral_threshold_mode: str = "empirical"):
    """Lazily construct (KnowledgeIntegration, NoveltyComputer, ESNConfig).

    Returns ``(None, None, None)`` and warns if the ``[novelty]`` extra or its
    embedder cannot be loaded, so callers can fall back to a no-novelty run.
    """
    try:
        from esn.core.spectral_models import ESNConfig
        from esn.core.knowledge import KnowledgeIntegration
        from esn.core.novelty import NoveltyComputer
    except Exception as exc:  # pragma: no cover - defensive import guard
        warnings.warn(
            "an analyzer was supplied (novelty requested) but the novelty stack "
            f"could not be imported ({exc!r}); proceeding without the novelty "
            "signal. Install the '[novelty]' extra to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, None

    config = ESNConfig()
    config.spectral_threshold_mode = spectral_threshold_mode

    # The embedder is the part that pulls in the heavy optional dependency
    # (sentence-transformers / torch). If it is unavailable, KnowledgeIntegration
    # still works with zero/random embeddings, but we surface a warning so the
    # caller knows novelty quality is degraded rather than silently weak.
    embedder = None
    try:
        from esn.core.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(config.embedding_model)
    except Exception as exc:
        warnings.warn(
            "novelty is enabled but the sentence-transformers embedder is "
            f"unavailable ({exc!r}); proceeding with the novelty stack but "
            "without learned embeddings (weak signal). Install the '[novelty]' "
            "extra for full novelty signals.",
            RuntimeWarning,
            stacklevel=2,
        )

    knowledge = KnowledgeIntegration(config=config, embedder=embedder)
    novelty_computer = NoveltyComputer(knowledge, config=config, seed=seed)
    return knowledge, novelty_computer, config


# ---------------------------------------------------------------------------
# make_llm_mutator(): real LLM-backed mutator
# ---------------------------------------------------------------------------

# Provider dispatch by model-name prefix. Kept small and explicit so the
# mapping is obvious from the public API rather than buried in a factory.
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")
_ANTHROPIC_PREFIXES = ("claude-",)


def make_llm_mutator(
    domain: DomainSpec,
    *,
    model: str,
    mutator_policy: str = "single_shot",
    max_tokens: int | None = None,
) -> Any:
    """Build an :class:`~esn.engine.mutator.LLMMutator` backed by a real provider.

    The provider is chosen from ``model``'s prefix: ``gpt-*`` / ``o*`` ->
    OpenAI, ``claude-*`` -> Anthropic. The provider SDK is imported lazily; a
    clear error is raised if the ``[llm]`` extra is not installed.

    Args:
        domain: The :class:`~esn.engine.domain.DomainSpec` the mutator targets.
        model: Provider model name (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-..."``).
        mutator_policy: ``"single_shot"`` (full-rewrite, default), ``"diff"``
            (SEARCH/REPLACE edit blocks applied to the parent), or ``"agentic_v1"``.
        max_tokens: Max completion tokens per call. Default (None) keeps the
            adapter default (1024), which truncates large programs — raise it for
            domains whose programs are long.

    Returns:
        An :class:`~esn.engine.mutator.LLMMutator` ready to pass to :func:`run`.

    Raises:
        ValueError: If the model prefix matches no known provider.
        RuntimeError: If the provider SDK (the ``[llm]`` extra) is not installed.
    """
    from esn.engine.mutator import LLMMutator

    llm_client = _build_llm_client(model, max_tokens=max_tokens)
    return LLMMutator(llm_client, domain, mutator_policy=mutator_policy)


def _build_llm_client(model: str, max_tokens: int | None = None):
    """Build a clean ``(system_prompt, user_prompt) -> str`` LLM callable.

    Reuses the adapters in :mod:`esn.core.llm_adapters` so the callable contract
    matches what :class:`~esn.engine.mutator.LLMMutator` expects.
    """
    if model.startswith(_OPENAI_PREFIXES):
        return _build_openai_client(model, max_tokens=max_tokens)
    if model.startswith(_ANTHROPIC_PREFIXES):
        return _build_anthropic_client(model, max_tokens=max_tokens)
    raise ValueError(
        f"Cannot infer a provider for model {model!r}. Expected a name "
        f"starting with one of {_OPENAI_PREFIXES} (OpenAI) or "
        f"{_ANTHROPIC_PREFIXES} (Anthropic)."
    )


def _build_openai_client(model: str, max_tokens: int | None = None):
    import os

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI model requested but the 'openai' package is not installed. "
            "Install the '[llm]' extra (e.g. `uv pip install 'esn[llm]'`)."
        ) from exc

    from esn.core.llm_adapters import OpenAIAdapter

    api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    return OpenAIAdapter(client, model, **kwargs)


def _build_anthropic_client(model: str, max_tokens: int | None = None):
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic model requested but the 'anthropic' package is not "
            "installed. Install the '[llm]' extra "
            "(e.g. `uv pip install 'esn[llm]'`)."
        ) from exc

    from esn.core.llm_adapters import AnthropicAdapter

    client = anthropic.Anthropic()
    kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
    return AnthropicAdapter(client, model, **kwargs)


# ---------------------------------------------------------------------------
# make_analyzer() / make_predictor(): LLM-backed novelty drivers
# ---------------------------------------------------------------------------


def make_analyzer(*, model: str, max_new_hypotheses: int = 3) -> Any:
    """Build an LLM-backed analyzer that activates ESN's novelty machinery.

    The analyzer turns each evaluated candidate into evidence + new hypotheses;
    those hypotheses are what the epistemic-spectral ``N_sp`` signal is computed
    over. Pass the result to :func:`run` as ``analyzer=`` to enable novelty.

    The provider is inferred from ``model``'s prefix (``gpt-*`` / ``o*`` ->
    OpenAI, ``claude-*`` -> Anthropic), exactly like :func:`make_llm_mutator`;
    the provider SDK (the ``[llm]`` extra) is imported lazily.

    Args:
        model: Provider model name (e.g. ``"gpt-4o-mini"``).
        max_new_hypotheses: Cap on hypotheses admitted per analysis.

    Returns:
        An :class:`~esn.engine.analyzer.LLMAnalyzer` to pass to :func:`run`.
    """
    from esn.engine.analyzer import LLMAnalyzer

    return LLMAnalyzer(_build_llm_client(model), max_new_hypotheses=max_new_hypotheses)


def make_predictor(*, model: str) -> Any:
    """Build an LLM-backed predictor (adds a prediction-surprise novelty term).

    Optional companion to :func:`make_analyzer`; pass to :func:`run` as
    ``predictor=``. Provider inferred from ``model``'s prefix, like
    :func:`make_llm_mutator`. Inert unless an ``analyzer`` is also supplied.

    Returns:
        An :class:`~esn.engine.predictor.LLMPredictor` to pass to :func:`run`.
    """
    from esn.engine.predictor import LLMPredictor

    return LLMPredictor(_build_llm_client(model))


# ---------------------------------------------------------------------------
# make_agent_mutator(): agentic (Claude Agent SDK) mutator
# ---------------------------------------------------------------------------


def make_agent_mutator(
    domain: DomainSpec,
    *,
    model: str = "claude-haiku-4-5-20251001",
    mutator_tools: str = "none",
    call_timeout_seconds: float = 300,
) -> Any:
    """Build an agentic (Claude Agent SDK) mutator for ``domain``.

    Unlike :func:`make_llm_mutator` (a single LLM completion per mutation),
    this returns a :class:`~esn.engine.claude_agent_mutator.ClaudeAgentMutator`
    that drives a multi-turn Claude *agent* loop and can optionally consult
    research tools (``mutator_tools="research"`` exposes WebSearch / WebFetch
    behind a strict isolation boundary). It authenticates via your Claude
    subscription / keychain credentials or the ``ANTHROPIC_API_KEY`` env var.

    The Claude Agent SDK is imported lazily; a clear error is raised if the
    ``[agent]`` extra is not installed.

    Args:
        domain: The :class:`~esn.engine.domain.DomainSpec` the mutator targets.
        model: Claude model name (default ``"claude-haiku-4-5-20251001"``).
        mutator_tools: Tool policy — ``"none"`` (default; no tools) or
            ``"research"`` (WebSearch / WebFetch only).
        call_timeout_seconds: Per-call wall-clock timeout for the full SDK
            stream consumption (default ``300``).

    Returns:
        A :class:`~esn.engine.claude_agent_mutator.ClaudeAgentMutator` ready to
        pass to :func:`run`.

    Raises:
        RuntimeError: If the Claude Agent SDK (the ``[agent]`` extra) is not
            installed.
    """
    try:
        from esn.engine.claude_agent_client import ClaudeAgentSDKClient
        from esn.engine.claude_agent_mutator import ClaudeAgentMutator
    except ImportError as exc:
        raise RuntimeError(
            "Agentic mutator requested but the Claude Agent SDK is not "
            "installed. Install the '[agent]' extra "
            '(e.g. `uv pip install "esn[agent]"`).'
        ) from exc

    client = ClaudeAgentSDKClient(
        model=model,
        mutator_tools=mutator_tools,
        call_timeout_seconds=call_timeout_seconds,
    )
    return ClaudeAgentMutator(client, domain)


# ---------------------------------------------------------------------------
# Subscription (Claude Agent SDK) analyzer / predictor: key-free novelty drivers
# ---------------------------------------------------------------------------


class _AgentLLMClient:
    """A ``(system_prompt, user_prompt) -> str`` client backed by the Claude
    subscription via ``claude_agent_sdk`` (no API key).

    Mirrors the one-shot SDK usage in
    :meth:`esn.engine.claude_agent_client.ClaudeAgentSDKClient.run_mutation`:
    a single ``query`` turn with no tools, no workspace, and no inherited
    settings, run on a dedicated thread so it never collides with a caller's
    own event loop. The concatenated assistant text is returned verbatim — the
    analyzer / predictor own the JSON parsing.
    """

    def __init__(self, model: str) -> None:
        self._model = model

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        # Lazy import with the same clear [agent]-extra guard as
        # make_agent_mutator, so importing esn.api stays SDK-optional.
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                TextBlock,
                query,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Agentic analyzer/predictor requested but the Claude Agent SDK "
                "is not installed. Install the '[agent]' extra "
                '(e.g. `uv pip install "esn[agent]"`).'
            ) from exc

        import asyncio
        import tempfile

        prompt = f"{system_prompt}\n\n{user_prompt}"
        result: dict[str, Any] = {"text": "", "error": None}

        async def _run() -> str:
            tempdir = tempfile.mkdtemp(prefix="esn_agent_analyzer_")
            options = ClaudeAgentOptions(
                model=self._model,
                max_turns=1,
                allowed_tools=[],
                disallowed_tools=[],
                mcp_servers={},
                cwd=tempdir,
                setting_sources=None,
            )
            chunks: list[str] = []
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
            return "\n".join(chunks)

        def _thread_main() -> None:
            try:
                result["text"] = asyncio.run(_run())
            except Exception as exc:  # noqa: BLE001 — surfaced to caller below
                result["error"] = exc

        # Dedicated thread: asyncio.run() needs a fresh loop, and the caller
        # (mutator / engine) may already own a running loop on this thread.
        import threading

        thread = threading.Thread(target=_thread_main, daemon=True)
        thread.start()
        thread.join()
        if result["error"] is not None:
            raise result["error"]
        return result["text"]


def make_agent_analyzer(*, model: str = "claude-haiku-4-5-20251001") -> Any:
    """Build a key-FREE analyzer backed by the Claude subscription.

    Same role as :func:`make_analyzer` (turns evaluated candidates into
    hypotheses, activating ESN's novelty machinery), but authenticated through
    your local Claude install / macOS keychain instead of an API key — like
    :func:`make_agent_mutator`. Pass the result to :func:`run` as ``analyzer=``.

    The Claude Agent SDK is imported lazily; a clear error is raised on first
    use if the ``[agent]`` extra is not installed.

    Args:
        model: Claude model name (default ``"claude-haiku-4-5-20251001"``).

    Returns:
        An :class:`~esn.engine.analyzer.LLMAnalyzer` to pass to :func:`run`.
    """
    from esn.engine.analyzer import LLMAnalyzer

    return LLMAnalyzer(_AgentLLMClient(model))


def make_agent_predictor(*, model: str = "claude-haiku-4-5-20251001") -> Any:
    """Build a key-FREE predictor backed by the Claude subscription.

    Subscription-backed companion to :func:`make_predictor` (adds a
    prediction-surprise novelty term); pass to :func:`run` as ``predictor=``.
    Like :func:`make_agent_mutator`, it needs no API key — only the ``[agent]``
    extra. Inert unless an ``analyzer`` is also supplied.

    Returns:
        An :class:`~esn.engine.predictor.LLMPredictor` to pass to :func:`run`.
    """
    from esn.engine.predictor import LLMPredictor

    return LLMPredictor(_AgentLLMClient(model))


__all__ = [
    "MockMutator",
    "RunResult",
    "make_agent_analyzer",
    "make_agent_mutator",
    "make_agent_predictor",
    "make_analyzer",
    "make_llm_mutator",
    "make_predictor",
    "run",
]
