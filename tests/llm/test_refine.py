"""Tests for the refinement sweep — all offline via FakeChatClient."""

from __future__ import annotations

import pytest

from bgrl.llm.client import CachingChatClient, ChatResponse, FakeChatClient, Usage
from bgrl.llm.parse import OutputFormat
from bgrl.llm.prompt import TERSE
from bgrl.llm.refine import BudgetExceeded, SweepAxes, SweepConfig, run_sweep
from bgrl.llm.render import PipListRenderer
from bgrl.llm.scorer import ReferenceAgentScorer


def _first_legal_responder(messages, params):
    return "0"  # always pick the first enumerated candidate


def _axes(models=("a", "b"), formats=(OutputFormat.INDEX_TEXT, OutputFormat.STRUCTURED)):
    return SweepAxes(
        models=models,
        renderers=(PipListRenderer(),),
        templates=(TERSE,),
        output_formats=formats,
    )


def test_sweep_ranks_all_candidates():
    config = SweepConfig(axes=_axes(), scorer=ReferenceAgentScorer.pubeval(), n_positions=3, seed=1)
    report = run_sweep(config, FakeChatClient(responder=_first_legal_responder))
    assert len(report.candidates) == 4  # 2 models x 2 formats
    assert report.n_positions == 3
    # ranked best-first, scores are agreement rates in [0, 1]
    scores = [c.mean_score for c in report.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # "0" is always a valid index -> no invalid responses, no fallbacks
    assert all(c.invalid_rate == 0.0 and c.fallback_rate == 0.0 for c in report.candidates)


def test_sweep_is_reproducible():
    config = SweepConfig(axes=_axes(), scorer=ReferenceAgentScorer.pubeval(), n_positions=3, seed=5)
    a = run_sweep(config, FakeChatClient(responder=_first_legal_responder))
    b = run_sweep(config, FakeChatClient(responder=_first_legal_responder))
    assert a == b


def test_budget_cap_raises_with_partial_report():
    def costly(messages, params):
        return ChatResponse("0", Usage(0, 0, 0, 0.01), "m")

    config = SweepConfig(
        axes=_axes(),
        scorer=ReferenceAgentScorer.pubeval(),
        n_positions=3,
        seed=1,
        budget_usd=0.005,  # the very first call (0.01) blows the budget
    )
    with pytest.raises(BudgetExceeded) as excinfo:
        run_sweep(config, FakeChatClient(responder=costly))
    assert excinfo.value.report.candidates  # partial report is attached and non-empty
    assert excinfo.value.report.total_cost_usd >= 0.01


def test_caching_makes_rerun_free():
    axes = _axes(models=("solo",), formats=(OutputFormat.INDEX_TEXT,))
    config = SweepConfig(axes=axes, scorer=ReferenceAgentScorer.pubeval(), n_positions=3, seed=2)
    inner = FakeChatClient(responder=_first_legal_responder)
    client = CachingChatClient(inner)
    run_sweep(config, client)
    calls_after_first = len(inner.calls)
    assert calls_after_first == 3  # one unique request per position
    run_sweep(config, client)  # identical requests -> all served from cache
    assert len(inner.calls) == calls_after_first
