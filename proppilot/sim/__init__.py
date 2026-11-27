"""Simulation engine exports for challenge evaluation, bootstrap, and funded phases."""

from proppilot.sim.bootstrap import WeekBlock, WeekPool, build_week_blocks, simulate_bootstrap_challenge
from proppilot.sim.challenge import (
    analyze_failures,
    sample_challenge_fan_paths,
    simulate_challenge_batch,
    simulate_challenge_single,
    simulate_rolling_starts,
)
from proppilot.sim.funded import simulate_funded_bootstrap, simulate_funded_single
from proppilot.sim.models import (
    BatchChallengeSummary,
    ChallengeRunResult,
    FailureAttribution,
    FundedSummary,
    format_days,
    format_expected_attempts,
    format_expected_cost,
    format_pct,
    format_usd,
)

__all__ = [
    "BatchChallengeSummary",
    "ChallengeRunResult",
    "FailureAttribution",
    "FundedSummary",
    "WeekBlock",
    "WeekPool",
    "analyze_failures",
    "build_week_blocks",
    "format_days",
    "format_expected_attempts",
    "format_expected_cost",
    "format_pct",
    "format_usd",
    "sample_challenge_fan_paths",
    "simulate_bootstrap_challenge",
    "simulate_challenge_batch",
    "simulate_challenge_single",
    "simulate_funded_bootstrap",
    "simulate_funded_single",
    "simulate_rolling_starts",
]
