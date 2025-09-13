"""
TastyAlgo strategies package
contains all the trading strategy implementations
"""

from .moving_average import MAcross
from .momentum import MomStrat
from .volatility_breakout import VolBreak
from .pairs_trading import PairsTrade

__all__ = [
    'MAcross',
    'MomStrat', 
    'VolBreak',
    'PairsTrade'
]