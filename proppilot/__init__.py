"""PropPilot: Turn any crypto strategy into a prop-firm pass plan."""

from proppilot.ingest import TradeDataset, load_trades
from proppilot.policy import FundedPolicy, SizingPolicy
from proppilot.rules import PhaseRule, PropRules, list_presets, load_preset, load_rules

__version__ = "0.1.0"

__all__ = [
    "FundedPolicy",
    "PhaseRule",
    "PropRules",
    "SizingPolicy",
    "TradeDataset",
    "__version__",
    "list_presets",
    "load_preset",
    "load_rules",
    "load_trades",
]
