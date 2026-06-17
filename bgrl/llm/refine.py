"""The refinement harness: sweep prompt/format/model candidates and rank them.

A candidate is one point in the cross-product of the swept axes — ``{model, board
renderer, prompt template, output format, reasoning on/off}``. Every candidate is run
as an :class:`~bgrl.agents.llm_agent.LLMAgent` over the *same* frozen position set
(from a :class:`~bgrl.llm.scorer.PositionScorer`) and graded, so the comparison has no
dice variance and differs only in the thing under test.

Cost and latency are first-class: pin ``temperature=0`` + a fixed ``seed`` and wrap the
client in :class:`~bgrl.llm.client.CachingChatClient` so re-runs and overlapping
candidates reuse paid responses, and enforce a hard USD ``budget_usd`` (and optional
``max_calls``) that raises :class:`BudgetExceeded` carrying the partial report rather
than silently overspending.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

from bgrl.agents.llm_agent import AgentStats, Fallback, LLMAgent
from bgrl.llm.client import ChatClient
from bgrl.llm.parse import OutputFormat
from bgrl.llm.prompt import PromptTemplate
from bgrl.llm.render import BoardRenderer
from bgrl.llm.scorer import PositionScorer


@dataclass(frozen=True, slots=True)
class Candidate:
    """One configuration to evaluate."""

    model: str
    renderer: BoardRenderer
    template: PromptTemplate
    output_format: OutputFormat
    reasoning: dict | None

    def label(self) -> str:
        """Stable, human-readable id used in the report and as a sort tiebreak input."""
        effort = (self.reasoning or {}).get("effort", "on") if self.reasoning else "off"
        return (
            f"{self.model} | {self.renderer.name} | {self.template.name} | "
            f"{self.output_format.value} | reason:{effort}"
        )


@dataclass(frozen=True, slots=True)
class SweepAxes:
    """The axes whose cross-product forms the candidate set."""

    models: tuple[str, ...]
    renderers: tuple[BoardRenderer, ...]
    templates: tuple[PromptTemplate, ...]
    output_formats: tuple[OutputFormat, ...]
    reasoning_options: tuple[dict | None, ...] = (None,)

    def candidates(self) -> Iterator[Candidate]:
        for model, renderer, template, fmt, reasoning in itertools.product(
            self.models, self.renderers, self.templates, self.output_formats, self.reasoning_options
        ):
            yield Candidate(model, renderer, template, fmt, reasoning)

    def count(self) -> int:
        return (
            len(self.models)
            * len(self.renderers)
            * len(self.templates)
            * len(self.output_formats)
            * len(self.reasoning_options)
        )


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Everything :func:`run_sweep` needs except the chat client."""

    axes: SweepAxes
    scorer: PositionScorer
    n_positions: int
    seed: int = 0
    budget_usd: float = float("inf")
    max_calls: int | None = None
    max_reprompts: int = 2


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """One candidate's results — quality plus the honest "model + harness" overheads."""

    label: str
    mean_score: float
    n_positions: int
    invalid_rate: float
    fallback_rate: float
    total_cost_usd: float
    mean_completion_tokens: float
    structured_unsupported: bool


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Candidates ranked best-first, with the run's totals."""

    candidates: tuple[CandidateReport, ...]
    scorer_name: str
    n_positions: int
    total_cost_usd: float


class BudgetExceeded(RuntimeError):
    """The cost (or call) cap was hit mid-sweep. Carries the partial :class:`SweepReport`."""

    def __init__(self, report: SweepReport, message: str) -> None:
        self.report = report
        super().__init__(message)


def run_sweep(config: SweepConfig, client: ChatClient) -> SweepReport:
    """Evaluate every candidate over the frozen position set and return a ranked report.

    Raises :class:`BudgetExceeded` (with the partial report attached) the moment the
    cumulative cost exceeds ``budget_usd`` or the cumulative model calls exceed
    ``max_calls``.
    """
    positions = config.scorer.positions(n=config.n_positions, seed=config.seed)
    reports: list[CandidateReport] = []
    spent = 0.0
    calls = 0

    for candidate in config.axes.candidates():
        stats = AgentStats()
        agent = LLMAgent(
            client,
            model=candidate.model,
            renderer=candidate.renderer,
            template=candidate.template,
            output_format=candidate.output_format,
            reasoning=candidate.reasoning,
            seed=config.seed,
            max_reprompts=config.max_reprompts,
            fallback=Fallback.FIRST_LEGAL,
            stats=stats,
        )
        scores: list[float] = []
        for pos in positions:
            chosen = agent.act(pos.state, pos.dice, list(pos.legal))
            scores.append(config.scorer.score(pos, chosen))
            cost_so_far = spent + stats.total_cost
            calls_so_far = calls + stats.api_calls
            if cost_so_far > config.budget_usd or _over_calls(calls_so_far, config.max_calls):
                reports.append(_candidate_report(candidate, scores, stats))
                partial = _rank(reports, config.scorer.name, config.n_positions, cost_so_far)
                raise BudgetExceeded(
                    partial,
                    f"budget exceeded: ${cost_so_far:.4f} cost / {calls_so_far} calls "
                    f"(limits: ${config.budget_usd} / {config.max_calls})",
                )
        reports.append(_candidate_report(candidate, scores, stats))
        spent += stats.total_cost
        calls += stats.api_calls

    return _rank(reports, config.scorer.name, config.n_positions, spent)


def _over_calls(calls: int, max_calls: int | None) -> bool:
    return max_calls is not None and calls > max_calls


def _candidate_report(
    candidate: Candidate, scores: list[float], stats: AgentStats
) -> CandidateReport:
    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_completion = stats.total_completion_tokens / stats.api_calls if stats.api_calls else 0.0
    return CandidateReport(
        label=candidate.label(),
        mean_score=mean_score,
        n_positions=len(scores),
        invalid_rate=stats.invalid_rate,
        fallback_rate=stats.fallback_rate,
        total_cost_usd=stats.total_cost,
        mean_completion_tokens=mean_completion,
        structured_unsupported=stats.structured_unsupported,
    )


def _rank(
    reports: list[CandidateReport], scorer_name: str, n_positions: int, total_cost: float
) -> SweepReport:
    ranked = sorted(reports, key=lambda r: (-r.mean_score, r.total_cost_usd, r.label))
    return SweepReport(
        candidates=tuple(ranked),
        scorer_name=scorer_name,
        n_positions=n_positions,
        total_cost_usd=total_cost,
    )
