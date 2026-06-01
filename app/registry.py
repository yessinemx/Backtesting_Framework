"""Registres strategies / allocators."""
from signals.moving_average import MovingAverageCrossover
from signals.momentum import MomentumStrategy
from signals.pairs_trading import PairsTradingWavelet
from allocation.equal_weight import EqualWeightAllocator
from allocation.risk_parity import RiskParityAllocator

STRATEGIES = {
    "Moving Average Crossover": MovingAverageCrossover,
    "Momentum": MomentumStrategy,
    "Pairs Trading (Wavelet)": PairsTradingWavelet,
}

ALLOCATORS = {
    "Equal Weight (EW)": EqualWeightAllocator,
    "ERC (Risk Parity)": RiskParityAllocator,
}
