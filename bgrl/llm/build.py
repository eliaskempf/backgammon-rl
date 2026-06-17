"""Assemble an LLM agent + client stack from CLI-style string options.

Shared by ``scripts/eval_llm.py`` (win-rate vs pubeval), ``scripts/eval_vs_gnubg.py``
(equity loss vs gnubg), and — later — the web opponent, so the string→object resolution
(renderer/template/format/reasoning names) and the ``Caching(BudgetGuard(real))`` client
wiring live in one place and the scripts stay thin.
"""

from __future__ import annotations

import numpy as np

from bgrl.agents.llm_agent import AgentStats, Fallback, LLMAgent
from bgrl.llm.client import (
    BudgetGuardClient,
    CachingChatClient,
    ChatClient,
    FakeChatClient,
    ResponseCache,
)
from bgrl.llm.parse import OutputFormat
from bgrl.llm.prompt import ALL_TEMPLATES
from bgrl.llm.render import ALL_RENDERERS


def reasoning_option(name: str) -> dict | None:
    """Map a CLI reasoning name to an OpenRouter ``reasoning`` block (``off`` -> ``None``)."""
    return None if name == "off" else {"effort": name}


def build_chat_client(
    *,
    live: bool,
    cache_path: str | None = None,
    budget_usd: float = float("inf"),
    max_calls: int | None = None,
) -> tuple[ChatClient, BudgetGuardClient | None]:
    """Assemble the client stack and return ``(client, budget_guard)``.

    Live: ``CachingChatClient(BudgetGuardClient(OpenRouterClient))`` — the guard wraps the
    real client *inside* the cache so cache hits never re-spend, and a cost/call cap aborts
    cleanly via :class:`~bgrl.llm.client.BudgetExceededError`. Offline: a fake client that
    always answers ``"0"`` (the first legal move) for a network-free wiring check; the cache
    wraps it only when a ``cache_path`` is given. ``budget_guard`` is ``None`` offline.
    """
    guard: BudgetGuardClient | None = None
    inner: ChatClient
    if live:
        from dotenv import load_dotenv

        from bgrl.llm.client import OpenRouterClient

        load_dotenv()  # pull OPENROUTER_API_KEY from a local dotenv file if present
        guard = BudgetGuardClient(OpenRouterClient(), cap_usd=budget_usd, max_calls=max_calls)
        inner = guard
    else:
        inner = FakeChatClient(responder=lambda messages, params: "0")

    cache = ResponseCache(cache_path) if cache_path else None
    if cache is not None or live:
        return CachingChatClient(inner, cache), guard
    return inner, guard


def build_llm_agent(
    client: ChatClient,
    *,
    model: str,
    renderer: str = "pip_list",
    template: str = "coach",
    output_format: str = "index_text",
    reasoning: str = "off",
    max_reprompts: int = 2,
    fallback: str = "first_legal",
    rng: np.random.Generator | None = None,
    seed: int | None = 0,
    stats: AgentStats | None = None,
) -> LLMAgent:
    """Build an :class:`~bgrl.agents.llm_agent.LLMAgent` from CLI-style string names.

    Names are resolved through the registries (``ALL_RENDERERS``/``ALL_TEMPLATES``/
    ``OutputFormat``/``Fallback``); an unknown name raises ``KeyError``/``ValueError`` rather
    than silently picking a default.
    """
    return LLMAgent(
        client,
        model=model,
        renderer=ALL_RENDERERS[renderer],
        template=ALL_TEMPLATES[template],
        output_format=OutputFormat(output_format),
        reasoning=reasoning_option(reasoning),
        max_reprompts=max_reprompts,
        fallback=Fallback(fallback),
        rng=rng,
        seed=seed,
        stats=stats,
    )
