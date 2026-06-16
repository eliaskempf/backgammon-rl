"""Training: the algorithm-agnostic self-play loop, CRN evaluation, and trainers."""

from .evaluate import MatchResult, play_match
from .loop import train
from .td_lambda import TDLambda

__all__ = ["MatchResult", "TDLambda", "play_match", "train"]
