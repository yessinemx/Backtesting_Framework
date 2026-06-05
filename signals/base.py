"""Abstract base class for all trading strategies."""
from abc import ABC, abstractmethod


class BaseStrategy(ABC):

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def generate_signals(self, prices, date, members):
        # returns {ticker: +1/0/-1 for long/neutral/short}
        pass

    @abstractmethod
    def get_parameters_schema():
        pass
