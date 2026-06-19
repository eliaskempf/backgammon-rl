"""Training: the algorithm-agnostic self-play loop, CRN evaluation, and trainers."""

from .calibration import CalibrationReport, HeadCalibration, calibration_report
from .evaluate import MatchResult, play_match
from .loop import train
from .td_lambda import TDLambda
from .tournament import RoundRobinResult, round_robin

__all__ = [
    "CalibrationReport",
    "HeadCalibration",
    "MatchResult",
    "RoundRobinResult",
    "TDLambda",
    "calibration_report",
    "play_match",
    "round_robin",
    "train",
]
