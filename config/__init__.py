"""Public backtester configuration package.

Importing `config` exposes only the generic backtester settings. The paper
replication keeps its fixed parameters in `config.config_paper`.
"""

from .config_backtester import *
from . import config_paper
