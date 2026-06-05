"""Equal-weight allocator: 1/N across active positions."""
import pandas as pd
from allocation import BaseAllocator

class EqualWeightAllocator(BaseAllocator):

    def __init__(self, parameters=None):
        super().__init__("Equal Weight", parameters or {})

    def allocate(self, signals, returns, date):
        longs = [t for t, s in signals.items() if s == 1]
        shorts = [t for t, s in signals.items() if s == -1]

        if not longs or not shorts:
            return {}

        weights = {}
        if longs:
            w = 1.0 / len(longs)
            for t in longs:
                weights[t] = +w
        if shorts:
            w = 1.0 / len(shorts)
            for t in shorts:
                weights[t] = -w
        return weights

    @staticmethod
    def get_parameters_schema():
        return {}
