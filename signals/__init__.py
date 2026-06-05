""" 
Signals module.
Base interface + concrete signal strategies
"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def generate_signals(self, prices, date, members):
        # retourne {ticker: +1 LONG | -1 SHORT | 0 neutre}
        pass

    @abstractmethod
    def get_parameters_schema():
        pass

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
