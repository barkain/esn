# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Pre-built LLM client adapters for common providers."""

from __future__ import annotations

import json
from typing import Any


class LLMAPIError(RuntimeError):
    """Fatal LLM API error (authentication failure, quota exhaustion).

    Raised when the API returns an error that cannot be resolved by retrying
    (e.g. invalid key, expired key, billing quota exceeded).  The engine should
    let this propagate and stop the run immediately.
    """


def _is_fatal_openai_error(exc: Exception) -> bool:
    """Return True if *exc* is an openai error that should kill the run."""
    try:
        import openai  # type: ignore[import-not-found]  # noqa: F811
    except ImportError:
        return False
    if isinstance(exc, openai.AuthenticationError):
        return True
    if isinstance(exc, openai.RateLimitError):
        msg = str(exc).lower()
        # Transient overload (e.g. Kimi "engine is currently overloaded")
        # is NOT fatal — let the caller retry.
        if "overloaded" in msg:
            return False
        # Quota exhaustion messages contain "exceeded" or "quota";
        # transient per-minute rate limits do not.
        if "quota" in msg or "exceeded" in msg or "billing" in msg:
            return True
    return False


def _is_fatal_anthropic_error(exc: Exception) -> bool:
    """Return True if *exc* is an anthropic error that should kill the run."""
    try:
        import anthropic  # type: ignore[import-not-found]  # noqa: F811
    except ImportError:
        return False
    if isinstance(exc, anthropic.AuthenticationError):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        msg = str(exc).lower()
        if "quota" in msg or "exceeded" in msg or "billing" in msg:
            return True
    return False


class AnthropicAdapter:
    """Adapter for Anthropic's Claude API."""

    def __init__(self, client: Any, model: str = "claude-sonnet-4-20250514", **kwargs: Any) -> None:
        self.client = client
        self.model = model
        self._extra_kwargs = kwargs

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._extra_kwargs.get("max_tokens", 1024),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        # Merge extra kwargs (excluding max_tokens already handled)
        for k, v in self._extra_kwargs.items():
            if k != "max_tokens":
                call_kwargs[k] = v
        try:
            response = self.client.messages.create(**call_kwargs)
        except Exception as exc:
            if _is_fatal_anthropic_error(exc):
                raise LLMAPIError(f"Anthropic API fatal error: {exc}") from exc
            raise
        return response.content[0].text


class OpenAIAdapter:
    """Adapter for OpenAI's API (GPT-4, GPT-5, etc.)."""

    def __init__(self, client: Any, model: str = "gpt-4o", **kwargs: Any) -> None:
        self.client = client
        self.model = model
        self._extra_kwargs = kwargs

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        is_o_series = self.model.startswith("o")
        # o-series reasoning models split max_completion_tokens between
        # invisible reasoning tokens and visible output. With the prior 16384
        # default, o3 frequently used the entire budget on reasoning and
        # returned empty / truncated code (Phase 0.2 root cause). Bumped to
        # 32768 so reasoning + a full ~600-line program both fit.
        default_tok = 32768 if is_o_series else 1024
        max_tok = self._extra_kwargs.get("max_tokens", default_tok)
        token_key = "max_completion_tokens" if is_o_series else "max_tokens"
        sys_role = "developer" if is_o_series else "system"
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            token_key: max_tok,
            "messages": [
                {"role": sys_role, "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        for k, v in self._extra_kwargs.items():
            if k != "max_tokens":
                call_kwargs[k] = v
        try:
            response = self.client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            if _is_fatal_openai_error(exc):
                raise LLMAPIError(f"OpenAI API fatal error: {exc}") from exc
            raise
        return response.choices[0].message.content or ""


class MockLLMClient:
    """Mock client for testing. Returns a fixed JSON response."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response
        self.last_system_prompt: str = ""
        self.last_user_prompt: str = ""
        self.call_count: int = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.call_count += 1
        if self.response is not None:
            return self.response
        # Default: return a valid MutationPlan JSON
        return json.dumps(
            {
                "search_mode": "exploit",
                "operator_name": "perturb_position",
                "target": "circles",
                "parameters": {},
                "rationale": "refine positions",
                "expected_effect": "small improvement",
                "risk": "low",
            }
        )
