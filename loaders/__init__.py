"""
Package loaders
===============
Chargement unifié des données (prix, membership, rendements, taux sans risque)
en **Polars** pour la rapidité.

Deux sources interchangeables :
  - "data"      : fichiers Parquet locaux du dossier data/ (défaut)
  - "bloomberg" : extraction live via l'API Bloomberg (blpapi)

API publique :
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
)
from loaders.bloomberg_loader import BloombergLoader

__all__ = [
    "load_prices",
    "load_returns",
    "load_membership",
    "load_riskfree",
    "BloombergLoader",
]
