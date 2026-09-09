"""
vecoffload -- simulator for feasibility-aware, ZKP-budgeted vehicular edge
offloading.

Reproduces the experiments reported in "Spatiotemporal Feasibility-Aware
Offloading in Vehicular Edge Computing with Budgeted Zero-Knowledge
Verification".

See ``vecoffload.config`` for the calibration note that must be read before
reusing these numbers.
"""

from .config import Config, GHZ, MB
from .channel import mac_contention, phy_rate_bps
from .zkp import zkp_costs_ms, zkp_profile
from .nodes import NodePool
from .cost import evaluate
from .scenario import pair_features, predict, sample_event
from .policy import ABLATION, DEFAULT_SEEDS, POLICIES, Runner, run_policy

__version__ = "1.0.0"

__all__ = [
    "Config", "GHZ", "MB",
    "phy_rate_bps", "mac_contention",
    "zkp_costs_ms", "zkp_profile",
    "NodePool", "evaluate",
    "sample_event", "predict", "pair_features",
    "Runner", "run_policy", "POLICIES", "ABLATION", "DEFAULT_SEEDS",
]
