"""LLM agent: prompt a frontier model (via OpenRouter) to choose a move.

Implements the WP0 :class:`~bgrl.agents.base.Agent` contract like every other agent —
``act(state, dice, legal) -> Move`` — so it drops into :func:`bgrl.game.play_game` and
the web UI unchanged. The model never scores afterstates; it picks one move from the
enumerated ``legal`` list by integer index (afterstate-first, CLAUDE.md §4), and the
agent maps that index straight back to the original ``Move`` object.

Robustness is part of the contract, not an afterthought: a model can return an
out-of-range index, prose, or malformed JSON. The pipeline is *constrained output ->
parse -> re-prompt up to N times -> deterministic fallback*, and every failure is
counted in :class:`AgentStats`. The reported strength is therefore "model + harness":
a weak parser would understate a strong model, so the invalid/fallback rates are
surfaced alongside the score rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from bgrl.env import Dice, EnvState, Move
from bgrl.llm.client import (
    ChatClient,
    ChatMessage,
    ChatParams,
    ChatResponse,
    StructuredOutputUnsupported,
)
from bgrl.llm.parse import OutputFormat, parse_choice
from bgrl.llm.prompt import PromptTemplate
from bgrl.llm.render import BoardRenderer


class Fallback(Enum):
    """What to play when every prompt attempt fails to yield a valid index."""

    FIRST_LEGAL = "first_legal"
    RANDOM_LEGAL = "random_legal"


@dataclass
class AgentStats:
    """Mutable accounting accumulated across :meth:`LLMAgent.act` calls.

    ``decisions`` counts turns; ``api_calls`` counts model completions (>= decisions
    because of re-prompts). ``invalid_responses`` are completions that did not parse
    to a valid index; ``fallbacks`` are turns that exhausted re-prompts and used the
    fallback move. ``structured_unsupported`` flips if the model rejected
    ``response_format`` and the agent downgraded to text parsing.
    """

    decisions: int = 0
    api_calls: int = 0
    invalid_responses: int = 0
    fallbacks: int = 0
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    structured_unsupported: bool = False

    @property
    def reprompts(self) -> int:
        return self.api_calls - self.decisions

    @property
    def invalid_rate(self) -> float:
        return self.invalid_responses / self.api_calls if self.api_calls else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.decisions if self.decisions else 0.0


class LLMAgent:
    """An :class:`~bgrl.agents.base.Agent` that selects moves via an LLM.

    ``client`` is any :class:`~bgrl.llm.client.ChatClient` (real, fake, or caching), so
    tests never hit the network. ``renderer``, ``template``, ``output_format``, and
    ``reasoning`` are the swept knobs. With ``fallback=RANDOM_LEGAL`` an ``rng`` is
    required (so fallbacks stay reproducible).
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        model: str,
        renderer: BoardRenderer,
        template: PromptTemplate,
        output_format: OutputFormat = OutputFormat.INDEX_TEXT,
        reasoning: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        max_reprompts: int = 2,
        fallback: Fallback = Fallback.FIRST_LEGAL,
        rng: np.random.Generator | None = None,
        stats: AgentStats | None = None,
    ) -> None:
        if fallback is Fallback.RANDOM_LEGAL and rng is None:
            raise ValueError("fallback=RANDOM_LEGAL requires an rng for reproducibility")
        self._client = client
        self._model = model
        self._renderer = renderer
        self._template = template
        self._output_format = output_format
        self._reasoning = reasoning
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._max_reprompts = max_reprompts
        self._fallback = fallback
        self._rng = rng
        self.stats = stats if stats is not None else AgentStats()

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move:
        self.stats.decisions += 1
        n = len(legal)
        fmt = self._output_format
        messages = self._template.build(state, dice, legal, self._renderer, fmt)

        attempts = 0
        while attempts <= self._max_reprompts:
            params = ChatParams(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=fmt.response_format(n),
                reasoning=self._reasoning,
                seed=self._seed,
            )
            try:
                response = self._client.complete(messages, params)
            except StructuredOutputUnsupported:
                if fmt is OutputFormat.INDEX_TEXT:
                    raise  # text mode sends no response_format; a rejection here is unexpected
                # Downgrade once to plain-text indices and retry without spending an attempt.
                self.stats.structured_unsupported = True
                fmt = OutputFormat.INDEX_TEXT
                messages = self._template.build(state, dice, legal, self._renderer, fmt)
                continue

            attempts += 1
            self.stats.api_calls += 1
            self._record_usage(response)
            idx = parse_choice(response.text, n=n, fmt=fmt)
            if idx is not None:
                return legal[idx][0]
            self.stats.invalid_responses += 1
            messages = [
                *messages,
                ChatMessage("assistant", response.text),
                ChatMessage("user", _corrective(n, fmt)),
            ]

        self.stats.fallbacks += 1
        return self._fallback_move(legal)

    def _record_usage(self, response: ChatResponse) -> None:
        self.stats.total_cost += response.usage.cost
        self.stats.total_prompt_tokens += response.usage.prompt_tokens
        self.stats.total_completion_tokens += response.usage.completion_tokens

    def _fallback_move(self, legal: list[tuple[Move, EnvState]]) -> Move:
        if self._fallback is Fallback.RANDOM_LEGAL:
            assert self._rng is not None  # guaranteed by __init__
            return legal[int(self._rng.integers(len(legal)))][0]
        return legal[0][0]


def _corrective(n: int, fmt: OutputFormat) -> str:
    return f"Your previous response did not contain a valid move index. {fmt.instruction(n)}"
