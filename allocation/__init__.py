"""Allocation methods for portfolio construction."""
from abc import ABC, abstractmethod
import pandas as pd


class BaseAllocator(ABC):

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def allocate(self, signals, returns, date):
        # signals : {ticker: +1 / -1 / 0}
        # returns {ticker: signed weight} (positive = long, negative = short)
        pass

    @staticmethod
    @abstractmethod
    def get_parameters_schema():
        pass

from .equal_weight import EqualWeightAllocator 
from .risk_parity import RiskParityAllocator  

__all__ = ["BaseAllocator", "EqualWeightAllocator", "RiskParityAllocator"]
