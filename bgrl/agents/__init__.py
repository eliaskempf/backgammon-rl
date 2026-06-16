"""Agents: the Agent interface and implementations (random, value, td, ...)."""

from .base import Agent, LearningAgent
from .random_agent import RandomAgent
from .value_agent import ValueAgent

__all__ = ["Agent", "LearningAgent", "RandomAgent", "ValueAgent"]
