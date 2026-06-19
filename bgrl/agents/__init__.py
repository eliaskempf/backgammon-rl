"""Agents: the Agent interface and implementations (random, value, td, llm, ...)."""

from .base import Agent, CubeCapable, LearningAgent
from .cube_policy import (
    NetEvaluator,
    PositionEvaluator,
    evaluator_for,
    onroll_outcome,
    wants_to_double,
    wants_to_take,
)
from .expectimax_agent import ExpectimaxAgent
from .llm_agent import AgentStats, Fallback, LLMAgent
from .pubeval_agent import PubevalAgent
from .random_agent import RandomAgent
from .value_agent import ValueAgent

__all__ = [
    "Agent",
    "AgentStats",
    "CubeCapable",
    "ExpectimaxAgent",
    "Fallback",
    "LLMAgent",
    "LearningAgent",
    "NetEvaluator",
    "PositionEvaluator",
    "PubevalAgent",
    "RandomAgent",
    "ValueAgent",
    "evaluator_for",
    "onroll_outcome",
    "wants_to_double",
    "wants_to_take",
]
