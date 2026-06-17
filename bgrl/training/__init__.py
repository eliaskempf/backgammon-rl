"""Training: the algorithm-agnostic self-play loop, CRN evaluation, and trainers."""

from .evaluate import MatchResult, play_match
from .loop import train
from .td_lambda import TDLambda
from .tournament import RoundRobinResult, round_robin

__all__ = ["MatchResult", "RoundRobinResult", "TDLambda", "play_match", "round_robin", "train"]
