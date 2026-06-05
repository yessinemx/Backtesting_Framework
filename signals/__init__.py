"""Signal strategies and base interface."""
from signals.base import BaseStrategy

from .pairs_trading_base import PairsTradingBase
from .moving_average import MovingAverageCrossover  
from .momentum import MomentumStrategy  
from .pairs_trading_wavelet import PairsTradingWavelet  
from .pairs_trading_cointegration import PairsTradingCointegration
from .pairs_trading_partial_cointegration import PairsTradingPartialCointegration

__all__ = [
    "BaseStrategy",
    "PairsTradingBase",
    "MovingAverageCrossover",
    "MomentumStrategy",
    "PairsTradingWavelet",
    "PairsTradingCointegration",
    "PairsTradingPartialCointegration",
]
