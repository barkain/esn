#!/usr/bin/env python3
"""One-command circle-packing evolution-vs-sampling reproduction.

This is a thin, user-facing wrapper around the research harness pieces in
``runs/``. It deliberately wires the experiment setup in-process so users do
not have to remember the old pile of environment variables.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable


EVOLUTION_BATCH_SIZE = 4
NZ_TIMEOUT_SECONDS = 90
NOVELTY_SPECTRAL_DIM = 8


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prepend_import_paths(root: Path) -> None:
    """Put this worktree's code and helper dirs ahead of any editable install."""
    paths = [
        root / "src",
        root / "examples",
        root / "runs" / "h2h_bf",
        root / "runs" / "novelty_exp",
    ]
    path_strings = [str(path) for path in paths]

    # Keep child processes, including candidate sandboxes, on the same code.
    existing_env = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        path_strings + [p for p in existing_env if p not in path_strings]
    )

    # The venv has an editable esn pointing at the main checkout; this process
    # must import from the current worktree instead.
    for path in reversed(path_strings):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _bootstrap_environment(root: Path) -> None:
    _prepend_import_paths(root)
    os.environ["NZ_TIMEOUT"] = str(NZ_TIMEOUT_SECONDS)
    os.environ["NZ_SEED"] = str(root / "runs" / "h2h_bf" / "scipy_seed.py")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_ESN")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this experiment.")
    os.environ["OPENAI_API_KEY"] = api_key


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the circle-packing evolution-vs-sampling experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=("sampling", "evolution"), required=True)
    parser.add_argument("--gens", type=int, default=40, help="Evolution generations.")
    parser.add_argument("--n", type=int, default=80, help="Sampling batch size.")
    parser.add_argument(
        "--novelty", action="store_true", help="Enable ESN novelty with spectral_dim=8."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args(argv)

    if args.method == "sampling" and args.n <= 0:
        parser.error("--n must be positive for sampling")
    if args.method == "evolution" and args.gens <= 0:
        parser.error("--gens must be positive for evolution")
    if args.method == "sampling" and args.novelty:
        parser.error("--novelty is only meaningful with --method evolution")
    return args


def _install_gate_neutralization() -> None:
    """Match the clean comparison by removing the non-upstream parent floor."""
    from esn.engine import engine as engine_mod

    engine_mod.PARENT_QUALITY_FLOOR_RATIO = 0.0


def _install_open_evolve_prompt() -> None:
    """Use the OpenEvolve-spirit prompt from the experiment helpers."""
    import oe_prompt

    oe_prompt.install()


def _install_spectral_dim_patch(spectral_dim: int) -> None:
    """Force the novelty stack to the working spectral dimension used here."""
    import esn.api as api

    def patched_stack(
        seed: int, spectral_threshold_mode: str = "empirical"
    ) -> tuple[Any, Any, Any]:
        from esn.core.knowledge import KnowledgeIntegration
        from esn.core.novelty import NoveltyComputer
        from esn.core.spectral_models import ESNConfig

        config = ESNConfig()
        config.spectral_threshold_mode = spectral_threshold_mode
        config.spectral_dim = spectral_dim

        embedder = None
        try:
            import contextlib
            import io

            from esn.core.embeddings import SentenceTransformerEmbedder

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*get_sentence_embedding_dimension.*",
                    category=FutureWarning,
                )
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    embedder = SentenceTransformerEmbedder(config.embedding_model)
        except Exception as exc:
            warnings.warn(
                "novelty is enabled but the sentence-transformers embedder is unavailable "
                f"({exc!r}); proceeding with a weaker novelty signal.",
                RuntimeWarning,
                stacklevel=2,
            )

        knowledge = KnowledgeIntegration(config=config, embedder=embedder)
        novelty_computer = NoveltyComputer(knowledge, config=config, seed=seed)
        return knowledge, novelty_computer, config

    api._build_novelty_stack = patched_stack


def _install_eval_counter() -> Callable[[], int]:
    """Count evaluated candidates the same way the research harness does."""
    from esn.engine.engine import ESNEngine

    original = ESNEngine._process_outcome
    count = {"n": 0}

    def counted_process_outcome(self: Any, outcome: Any) -> Any:
        count["n"] += 1
        return original(self, outcome)

    ESNEngine._process_outcome = counted_process_outcome
    return lambda: count["n"]


def _run(args: argparse.Namespace) -> tuple[float, int]:
    import esn
    from biasfree_nz import biasfree_nz_domain

    _install_gate_neutralization()
    _install_open_evolve_prompt()
    if args.novelty:
        _install_spectral_dim_patch(NOVELTY_SPECTRAL_DIM)

    n_evals = _install_eval_counter()
    domain = biasfree_nz_domain()
    mutator = esn.make_llm_mutator(domain, model=args.model)

    analyzer = None
    predictor = None
    if args.novelty:
        analyzer = esn.make_analyzer(model=args.model)
        predictor = esn.make_predictor(model=args.model)

    generations = 1 if args.method == "sampling" else args.gens
    batch_size = args.n if args.method == "sampling" else EVOLUTION_BATCH_SIZE
    enable_recombination = args.method == "evolution"

    if analyzer is None:
        warnings.filterwarnings(
            "ignore",
            message=r"esn\.run\(\) was called without an `analyzer`.*",
            category=RuntimeWarning,
        )

    result = esn.run(
        domain,
        mutator=mutator,
        analyzer=analyzer,
        predictor=predictor,
        generations=generations,
        batch_size=batch_size,
        seed=args.seed,
        enable_recombination=enable_recombination,
    )
    return float(result.best_score), n_evals()


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    args = _parse_args(argv)
    _bootstrap_environment(root)
    best_score, n_evals = _run(args)

    print(f"RESULT method={args.method} best_score={best_score:.6f} n_evals={n_evals}")
    print("(sampling ceiling ~2.61, AlphaEvolve SOTA 2.635)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
