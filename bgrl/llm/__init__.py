"""LLM agent (WP4): prompt a frontier model via OpenRouter to choose moves.

The package is deliberately layered so each axis the refinement harness sweeps is
an independent, swappable piece:

* :mod:`bgrl.llm.client` — the OpenRouter HTTP client behind a :class:`ChatClient`
  protocol, with a network-free fake and a caching decorator for the sweep.
* :mod:`bgrl.llm.render` — pluggable board serialisations (the model's "view").
* :mod:`bgrl.llm.parse` — output-format strategies and the pure choice parser.
* :mod:`bgrl.llm.prompt` — composes a renderer + format into chat messages.
* :mod:`bgrl.llm.refine` — the candidate sweep, scorers, and ranked report.

The agent itself lives at :class:`bgrl.agents.llm_agent.LLMAgent` (it implements
the WP0 :class:`~bgrl.agents.base.Agent` contract like every other agent).
"""

from __future__ import annotations
