"""Simulation result dataclasses and summary containers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


OUTCOME_UNFINISHED = 0
OUTCOME_PASS = 1
OUTCOME_FAIL = 2
OUTCOME_RESTART = 3

FAIL_NONE = 0
FAIL_DAILY_LOSS = 1
FAIL_MAX_LOSS = 2
FAIL_MAX_DAYS = 3
FAIL_RESTART = 4

FAIL_CODE_TO_NAME = {
    FAIL_NONE: "none",
    FAIL_DAILY_LOSS: "daily_loss",
    FAIL_MAX_LOSS: "max_loss",
    FAIL_MAX_DAYS: "max_days_timeout",
    FAIL_RESTART: "restart_timeout",
}


@dataclass
class ChallengeRunResult:
    """Result of a single evaluation challenge run from one start time."""

    outcome: str  # "pass" | "fail" | "restart" | "unfinished"
    days_to_pass: Optional[float]
    fail_reason: Optional[str]
    fail_phase: Optional[int]
    final_balance: float
    trades_taken: int
    start_ms: int
    end_ms: int
    breach_day_utc: Optional[str] = None
    breach_symbol: Optional[str] = None
    breach_day_symbols: tuple[str, ...] = ()
    breach_day_r_sum: Optional[float] = None
    curve_days: Optional[np.ndarray] = None
    curve_balances: Optional[np.ndarray] = None


@dataclass
class BatchChallengeSummary:
    """Aggregated evaluation statistics across rolling starts or bootstrap paths."""

    n_total: int
    n_passed: int
    n_failed: int
    n_restarted: int
    n_unfinished: int
    n_finished: int
    pass_rate: float
    fail_rate: float  # hard rule breaches / finished
    restart_rate: float  # policy restarts / finished
    non_pass_rate: float  # (failed + restarted) / finished = 1 - pass_rate
    expected_attempts: float
    expected_fees: float
    days_min: Optional[float]
    days_median: Optional[float]
    days_mean: Optional[float]
    days_p90: Optional[float]
    days_max: Optional[float]
    days_to_pass: np.ndarray
    fail_reason_counts: dict[str, int]
    outcomes: np.ndarray
    fail_reasons: np.ndarray
    days_per_run: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert numeric metrics to a JSON-serializable summary dict."""
        return {
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_restarted": self.n_restarted,
            "n_unfinished": self.n_unfinished,
            "n_finished": self.n_finished,
            "pass_rate": round(self.pass_rate, 4),
            "fail_rate": round(self.fail_rate, 4),
            "restart_rate": round(self.restart_rate, 4),
            "non_pass_rate": round(self.non_pass_rate, 4),
            "expected_attempts": round(self.expected_attempts, 3)
            if np.isfinite(self.expected_attempts)
            else None,
            "expected_fees": round(self.expected_fees, 2)
            if np.isfinite(self.expected_fees)
            else None,
            "days_min": round(self.days_min, 2) if self.days_min is not None else None,
            "days_median": round(self.days_median, 2) if self.days_median is not None else None,
            "days_mean": round(self.days_mean, 2) if self.days_mean is not None else None,
            "days_p90": round(self.days_p90, 2) if self.days_p90 is not None else None,
            "days_max": round(self.days_max, 2) if self.days_max is not None else None,
            "fail_reason_counts": dict(self.fail_reason_counts),
        }


@dataclass
class FundedSummary:
    """Aggregated 12-month funded account simulation statistics."""

    n_paths: int
    account_size: float
    p_lose_account: float
    fail_reason_counts: dict[str, int]
    monthly_payout_mean_usd: float
    monthly_payout_median_usd: float
    monthly_payout_p10_usd: float
    monthly_payout_p25_usd: float
    monthly_payout_p75_usd: float
    monthly_payout_p90_usd: float
    monthly_payout_mean_pct: float
    monthly_payout_median_pct: float
    total_12m_payout_mean_usd: float
    total_12m_payout_median_usd: float
    prob_zero_payout: float
    per_month_mean_usd: np.ndarray
    path_monthly_avg_usd: np.ndarray
    path_12m_total_usd: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable funded summary metrics."""
        return {
            "n_paths": self.n_paths,
            "account_size": self.account_size,
            "p_lose_account": round(self.p_lose_account, 4),
            "fail_reason_counts": dict(self.fail_reason_counts),
            "monthly_payout_mean_usd": round(self.monthly_payout_mean_usd, 2),
            "monthly_payout_median_usd": round(self.monthly_payout_median_usd, 2),
            "monthly_payout_p10_usd": round(self.monthly_payout_p10_usd, 2),
            "monthly_payout_p25_usd": round(self.monthly_payout_p25_usd, 2),
            "monthly_payout_p75_usd": round(self.monthly_payout_p75_usd, 2),
            "monthly_payout_p90_usd": round(self.monthly_payout_p90_usd, 2),
            "monthly_payout_mean_pct": round(self.monthly_payout_mean_pct, 3),
            "monthly_payout_median_pct": round(self.monthly_payout_median_pct, 3),
            "total_12m_payout_mean_usd": round(self.total_12m_payout_mean_usd, 2),
            "total_12m_payout_median_usd": round(self.total_12m_payout_median_usd, 2),
            "prob_zero_payout": round(self.prob_zero_payout, 4),
            "per_month_mean_usd": [round(float(x), 2) for x in self.per_month_mean_usd],
        }


@dataclass
class FailureAttribution:
    """Breakdown of which trading days, symbols, and rule clauses caused failures."""

    n_total_starts: int
    n_breaches: int
    fail_reason_counts: dict[str, int]
    top_breach_days: list[dict[str, Any]] = field(default_factory=list)
    top_breach_symbols: list[dict[str, Any]] = field(default_factory=list)
    sample_breaches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total_starts": self.n_total_starts,
            "n_breaches": self.n_breaches,
            "fail_reason_counts": dict(self.fail_reason_counts),
            "top_breach_days": self.top_breach_days,
            "top_breach_symbols": self.top_breach_symbols,
            "sample_breaches": self.sample_breaches,
        }


def summarize_batch_arrays(
    outcomes: np.ndarray,
    fail_reasons: np.ndarray,
    days_per_run: np.ndarray,
    fee: float,
) -> BatchChallengeSummary:
    """Build a BatchChallengeSummary from outcome arrays for one policy."""
    n_total = int(len(outcomes))
    n_passed = int(np.sum(outcomes == OUTCOME_PASS))
    n_failed = int(np.sum(outcomes == OUTCOME_FAIL))
    n_restarted = int(np.sum(outcomes == OUTCOME_RESTART))
    n_unfinished = int(np.sum(outcomes == OUTCOME_UNFINISHED))
    n_finished = n_passed + n_failed + n_restarted

    if n_finished > 0:
        pass_rate = n_passed / n_finished
        fail_rate = n_failed / n_finished
        restart_rate = n_restarted / n_finished
        non_pass_rate = (n_failed + n_restarted) / n_finished
    else:
        pass_rate = 0.0
        fail_rate = 0.0
        restart_rate = 0.0
        non_pass_rate = 1.0

    if pass_rate > 0.0:
        expected_attempts = 1.0 / pass_rate
        expected_fees = expected_attempts * fee
    else:
        expected_attempts = float("inf")
        expected_fees = float("inf")

    pass_days = days_per_run[outcomes == OUTCOME_PASS]
    if len(pass_days) > 0:
        days_min = float(np.min(pass_days))
        days_median = float(np.median(pass_days))
        days_mean = float(np.mean(pass_days))
        days_p90 = float(np.percentile(pass_days, 90))
        days_max = float(np.max(pass_days))
    else:
        days_min = days_median = days_mean = days_p90 = days_max = None

    reason_counts: dict[str, int] = {}
    for code, name in FAIL_CODE_TO_NAME.items():
        if code == FAIL_NONE:
            continue
        cnt = int(np.sum(fail_reasons == code))
        if cnt > 0:
            reason_counts[name] = cnt

    return BatchChallengeSummary(
        n_total=n_total,
        n_passed=n_passed,
        n_failed=n_failed,
        n_restarted=n_restarted,
        n_unfinished=n_unfinished,
        n_finished=n_finished,
        pass_rate=float(pass_rate),
        fail_rate=float(fail_rate),
        restart_rate=float(restart_rate),
        non_pass_rate=float(non_pass_rate),
        expected_attempts=float(expected_attempts),
        expected_fees=float(expected_fees),
        days_min=days_min,
        days_median=days_median,
        days_mean=days_mean,
        days_p90=days_p90,
        days_max=days_max,
        days_to_pass=pass_days.copy(),
        fail_reason_counts=reason_counts,
        outcomes=outcomes.copy(),
        fail_reasons=fail_reasons.copy(),
        days_per_run=days_per_run.copy(),
    )


def format_expected_cost(expected_fees: float) -> str:
    """Format expected evaluation fee cost, returning 'N/A (0% pass rate)' when non-finite."""
    if math.isfinite(expected_fees):
        return f"${expected_fees:,.0f}"
    return "N/A (0% pass rate)"


def format_expected_attempts(expected_attempts: float) -> str:
    """Format expected challenge attempts, guarding non-finite values."""
    if math.isfinite(expected_attempts):
        return f"{expected_attempts:.2f}"
    return "N/A (0% pass rate)"


def format_days(days: Optional[float], suffix: str = "d") -> str:
    """Format a day duration if present and finite, otherwise 'N/A'."""
    if days is not None and math.isfinite(days):
        return f"{days:.1f}{suffix}"
    return "N/A"


def format_pct(fraction: float, decimals: int = 1) -> str:
    """Format a [0, 1] fraction as a percentage string, guarding non-finite values."""
    if math.isfinite(fraction):
        return f"{fraction * 100:.{decimals}f}%"
    return "N/A"


def format_usd(amount: float, suffix: str = "") -> str:
    """Format a USD currency amount, guarding non-finite values."""
    if math.isfinite(amount):
        return f"${amount:,.0f}{suffix}"
    return "N/A"
