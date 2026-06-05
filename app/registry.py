"""Strategy and allocator registry."""
from signals.moving_average import MovingAverageCrossover
from signals.momentum import MomentumStrategy
from signals.pairs_trading_wavelet import PairsTradingWavelet
from signals.pairs_trading_cointegration import PairsTradingCointegration
from signals.pairs_trading_partial_cointegration import PairsTradingPartialCointegration
from allocation.equal_weight import EqualWeightAllocator
from allocation.risk_parity import RiskParityAllocator

# Main strategies exposed in Streamlit; Pairs Trading runs composite mode.
STRATEGIES = {
    "Moving Average Crossover": MovingAverageCrossover,
    "Momentum": MomentumStrategy,
    "Pairs Trading": PairsTradingWavelet,
}

# Benchmark strategies executed alongside "Pairs Trading" (composite mode).
# Key = display label used in the comparison chart.
PAIRS_BENCHMARKS = {
    "Cointegration": PairsTradingCointegration,
    "Partial Cointegration": PairsTradingPartialCointegration,
}

ALLOCATORS = {
    "Equal Weight (EW)": EqualWeightAllocator,
    "ERC (Risk Parity)": RiskParityAllocator,
}
