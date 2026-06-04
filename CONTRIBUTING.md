# Contributing

Thanks for your interest in improving this project. This guide is intentionally
short.

> This project is **uv-only**. Use [`uv`](https://docs.astral.sh/uv/) for every
> command. Do not use `pip`, `poetry`, or `conda` directly.

## Dev setup

```bash
uv venv
uv pip install -e ".[dev,all]"
```

This creates a virtual environment and installs the package in editable mode
with the development and all optional dependencies.

## Run the tests

```bash
uv run pytest
```

## Lint and format

```bash
uv run ruff check        # lint
uv run ruff check --fix  # lint + autofix
uv run ruff format       # format
```

Please make sure `uv run ruff check` and `uv run pytest` both pass before
opening a pull request.

## Pull requests

- **Branch first.** Never commit directly to `main`; open a PR from a feature
  branch.
- Keep PRs focused — one logical change per PR.
- Write a clear title and a description of *what* changed and *why*.
- Add or update tests for any behavior change.
- Make sure CI (lint + tests) is green.
- If you change a public API surface (engine, `DomainSpec`, mutator/compiler
  protocols), note it in the PR description.

## Reporting bugs

Open an issue with a minimal reproduction — ideally a tiny `DomainSpec` and the
command you ran. For sensitive or security-related reports, please contact the
maintainers privately rather than filing a public issue.
