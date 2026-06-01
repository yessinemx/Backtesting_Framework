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

from .moving_average import MovingAverageCrossover  
from .momentum import MomentumStrategy  
from .pairs_trading import PairsTradingWavelet  

__all__ = ["BaseStrategy", "MovingAverageCrossover", "MomentumStrategy", "PairsTradingWavelet"]
