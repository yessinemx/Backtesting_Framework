"""
Loaders package.

Unified loading for prices, membership, returns, and risk-free data in
Polars for speed.

Two interchangeable sources:
  - "data": local parquet files from data/
  - "bloomberg": live extraction through the Bloomberg API (blpapi)

Public API:
  - load_prices(source=..., index_id=..., ...)
  - load_returns(...)
  - load_membership(source=..., index_id=...)
  - load_riskfree(...)
"""
from loaders.data_loader import (
    load_prices,
    load_returns,
    load_membership,
    load_riskfree,
    members_asof,
)
from loaders.bloomberg_loader import BloombergLoader

__all__ = [
    "load_prices",
    "load_returns",
    "load_membership",
    "load_riskfree",
    "members_asof",
    "BloombergLoader",
]
