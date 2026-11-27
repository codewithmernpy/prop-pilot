"""Honest grid optimisation with strict 60/40 chronological train/validation split."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from proppilot.ingest import TradeDataset
from proppilot.policy import SizingPolicy, build_default_grid, build_funded_grid
from proppilot.rules import PropRules
from proppilot.sim import (
    BatchChallengeSummary,
    FailureAttribution,
    FundedSummary,
    WeekPool,
    analyze_failures,
    build_week_blocks,
    format_days,
    format_expected_cost,
    format_pct,
    format_usd,
    simulate_bootstrap_challenge,
    simulate_challenge_batch,
    simulate_funded_bootstrap,
    simulate_rolling_starts,
)

ObjectiveName = Literal["fastest", "cheapest", "max_value"]

DEFAULT_MAX_LOSE_FUNDED = 0.15
FUNDED_SELECT_PATHS = 300


def _guard_key(pol: SizingPolicy) -> tuple[object, ...]:
    """Policy fields (besides the funded ones) that change funded-phase results."""
    return (
        pol.own_daily_stop_pct,
        pol.max_open_risk_pct,
        pol.per_symbol_daily_cap_pct,
        pol.risk_basis,
    )


def _funded_key(pol: SizingPolicy) -> tuple[object, ...]:
    return _guard_key(pol) + (pol.funded_risk_pct, pol.keep_buffer_pct, pol.cppi_cap, pol.cppi_c0)


def expected_net_12m_usd(fees: float, funded: FundedSummary) -> float:
    """Expected 12-month value of one funded attempt cycle.

    = E[12m trader payouts] (lost accounts contribute only what they paid before the loss)
      - E[challenge fees to get funded] * (1 + P(lose funded within 12m)),
    i.e. a lost account has to be bought (and passed) again.
    """
    if not np.isfinite(fees):
        return float("-inf")
    return float(funded.total_12m_payout_mean_usd - fees * (1.0 + funded.p_lose_account))


@dataclass
class FundedCandidate:
    """One funded-phase variant evaluated on the optimisation slice."""

    policy: SizingPolicy
    summary: FundedSummary
    meets_constraint: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "funded_risk_pct": self.policy.funded_risk_pct,
            "cppi_c0": self.policy.cppi_c0,
            "cppi_cap": self.policy.cppi_cap,
            "keep_buffer_pct": self.policy.keep_buffer_pct,
            "p_lose_account": round(self.summary.p_lose_account, 4),
            "total_12m_payout_mean_usd": round(self.summary.total_12m_payout_mean_usd, 2),
            "monthly_payout_mean_usd": round(self.summary.monthly_payout_mean_usd, 2),
            "meets_constraint": self.meets_constraint,
        }


@dataclass
class GridCandidateRecord:
    """Single evaluated policy row on the optimisation (first 60%) split."""

    policy: SizingPolicy
    summary: BatchChallengeSummary
    feasible: bool
    score: float
    expected_net_monthly_usd: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_pct": self.policy.risk_pct,
            "near_frac": self.policy.near_frac,
            "near_mult": self.policy.near_mult,
            "own_daily_stop_pct": self.policy.own_daily_stop_pct,
            "restart_after_days": self.policy.restart_after_days,
            "funded_risk_pct": self.policy.funded_risk_pct,
            "pass_rate": round(self.summary.pass_rate, 4),
            "fail_rate": round(self.summary.fail_rate, 4),
            "restart_rate": round(self.summary.restart_rate, 4),
            "non_pass_rate": round(self.summary.non_pass_rate, 4),
            "days_median": round(self.summary.days_median, 2)
            if self.summary.days_median is not None
            else None,
            "expected_fees": round(self.summary.expected_fees, 2)
            if np.isfinite(self.summary.expected_fees)
            else None,
            "feasible": self.feasible,
            "expected_net_monthly_usd": round(self.expected_net_monthly_usd, 2)
            if self.expected_net_monthly_usd is not None and np.isfinite(self.expected_net_monthly_usd)
            else None,
        }


@dataclass
class TrainOptimisationOutput:
    """Result from optimising strictly on the training slice (never sees validation data)."""

    objective: ObjectiveName
    max_fail: float
    max_lose_funded: float
    best_policy: SizingPolicy
    best_opt_summary: BatchChallengeSummary
    grid_records: list[GridCandidateRecord]
    feasible_found: bool
    funded_candidates: list[FundedCandidate]
    funded_constraint_met: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Complete optimisation + out-of-sample validation + bootstrap + funded bundle."""

    objective: ObjectiveName
    max_fail: float
    rules: PropRules
    best_policy: SizingPolicy
    opt_summary: BatchChallengeSummary
    val_summary: BatchChallengeSummary
    boot_summary: BatchChallengeSummary
    funded_summary: FundedSummary
    failure_attribution: FailureAttribution
    grid_records: list[GridCandidateRecord]
    opt_trade_count: int
    val_trade_count: int
    feasible_found: bool
    warnings: list[str] = field(default_factory=list)
    funded_val_summary: Optional[FundedSummary] = None
    max_lose_funded: float = DEFAULT_MAX_LOSE_FUNDED
    funded_constraint_met: bool = True
    funded_candidates: list[FundedCandidate] = field(default_factory=list)

    @property
    def funded_p_lose_worst(self) -> float:
        """Worst P(lose funded within 12m) across the optimisation and validation bootstraps."""
        vals = [self.funded_summary.p_lose_account]
        if self.funded_val_summary is not None:
            vals.append(self.funded_val_summary.p_lose_account)
        return float(max(vals))

    @property
    def funded_risk_ok(self) -> bool:
        return self.funded_constraint_met and self.funded_p_lose_worst <= self.max_lose_funded + 1e-12

    @property
    def expected_net_12m(self) -> float:
        return expected_net_12m_usd(self.val_summary.expected_fees, self.funded_summary)

    def funded_risk_line(self) -> str:
        """Prominent one-line funded-account risk statement for plan cards."""
        opt_part = format_pct(self.funded_summary.p_lose_account)
        val_part = (
            f", validation-slice bootstrap {format_pct(self.funded_val_summary.p_lose_account)}"
            if self.funded_val_summary is not None
            else ""
        )
        status = "within" if self.funded_risk_ok else "ABOVE"
        return (
            f"P(lose funded account within 12m): optimise-slice bootstrap "
            f"{opt_part}{val_part} "
            f"({status} the {format_pct(self.max_lose_funded, decimals=0)} limit)"
        )

    def plan_card_summary(self, include_funded_risk: bool = True) -> str:
        """Return a single-line factual PLAN CARD summary using measured metrics only."""
        days_str = format_days(self.val_summary.days_median, suffix="")
        if math.isfinite(self.val_summary.expected_fees):
            cost_str = f"{format_expected_cost(self.val_summary.expected_fees)}/funded account"
        else:
            cost_str = format_expected_cost(self.val_summary.expected_fees)
        fn = self.funded_summary
        mean_usd = format_usd(fn.monthly_payout_mean_usd, suffix="/mo")
        mean_pct = f"{fn.monthly_payout_mean_pct:.2f}%/mo" if math.isfinite(fn.monthly_payout_mean_pct) else "N/A"
        med_usd = format_usd(fn.monthly_payout_median_usd, suffix="/mo")
        parts = [
            (
                f"{self.best_policy.label()} -> "
                f"validate pass {format_pct(self.val_summary.pass_rate)} "
                f"(optimise {format_pct(self.opt_summary.pass_rate)}, "
                f"bootstrap {format_pct(self.boot_summary.pass_rate)}), "
                f"median {days_str} days, expected cost {cost_str}"
            )
        ]
        if include_funded_risk:
            parts.append(f"FUNDED RISK: {self.funded_risk_line()}")
        parts.append(
            f"{self.best_policy.funded_label()}: mean payout {mean_usd} "
            f"({mean_pct}, lost accounts included), "
            f"median {med_usd}"
        )
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full optimisation and validation report to JSON-ready dict."""
        net_12m = self.expected_net_12m
        return {
            "objective": self.objective,
            "max_fail": self.max_fail,
            "max_lose_funded": self.max_lose_funded,
            "feasible_found": self.feasible_found,
            "funded_constraint_met": self.funded_constraint_met,
            "funded_risk_ok": self.funded_risk_ok,
            "warnings": list(self.warnings),
            "plan_card": self.plan_card_summary(),
            "funded_risk_line": self.funded_risk_line(),
            "expected_net_12m_usd": round(net_12m, 2) if np.isfinite(net_12m) else None,
            "rules": self.rules.model_dump(mode="json"),
            "best_policy": self.best_policy.model_dump(mode="json"),
            "split": {
                "opt_trade_count": self.opt_trade_count,
                "val_trade_count": self.val_trade_count,
            },
            "optimise": self.opt_summary.to_dict(),
            "validate": self.val_summary.to_dict(),
            "bootstrap": self.boot_summary.to_dict(),
            "funded": self.funded_summary.to_dict(),
            "funded_validate": self.funded_val_summary.to_dict()
            if self.funded_val_summary is not None
            else None,
            "funded_candidates": [c.to_dict() for c in self.funded_candidates],
            "failures": self.failure_attribution.to_dict(),
            "grid": [rec.to_dict() for rec in self.grid_records],
        }


def select_funded_variant(
    opt_trades: TradeDataset,
    rules: PropRules,
    base_policy: SizingPolicy,
    max_lose_funded: float = DEFAULT_MAX_LOSE_FUNDED,
    n_paths: int = FUNDED_SELECT_PATHS,
    seed: int = 42,
    pool: WeekPool | None = None,
    cache: dict[tuple[object, ...], FundedSummary] | None = None,
) -> tuple[SizingPolicy, list[FundedCandidate], bool]:
    """Pick the funded-phase variant of `base_policy` using ONLY `opt_trades`.

    Among variants with P(lose within 12m) <= `max_lose_funded`, choose the highest expected
    12-month payout. If none qualifies, fall back to the lowest P(lose) (flagged as not met).
    """
    week_pool = pool if pool is not None else build_week_blocks(opt_trades)
    memo = cache if cache is not None else {}
    candidates: list[FundedCandidate] = []
    for variant in build_funded_grid(rules, base_policy):
        key = _funded_key(variant)
        summary = memo.get(key)
        if summary is None:
            summary = simulate_funded_bootstrap(
                opt_trades, rules, variant, n_paths=n_paths, seed=seed, pool=week_pool
            )
            memo[key] = summary
        candidates.append(
            FundedCandidate(
                policy=variant,
                summary=summary,
                meets_constraint=summary.p_lose_account <= max_lose_funded + 1e-12,
            )
        )
    passing = [c for c in candidates if c.meets_constraint]
    if passing:
        best = max(passing, key=lambda c: (c.summary.total_12m_payout_mean_usd, -c.summary.p_lose_account))
        return best.policy, candidates, True
    best = min(candidates, key=lambda c: (c.summary.p_lose_account, -c.summary.total_12m_payout_mean_usd))
    return best.policy, candidates, False


def optimise_on_train(
    opt_trades: TradeDataset,
    rules: PropRules,
    objective: ObjectiveName = "fastest",
    max_fail: float = 0.20,
    grid: list[SizingPolicy] | None = None,
    seed: int = 42,
    max_lose_funded: float = DEFAULT_MAX_LOSE_FUNDED,
    funded_paths: int = FUNDED_SELECT_PATHS,
) -> TrainOptimisationOutput:
    """Search `grid` using ONLY `opt_trades` (first 60% split).

    This function intentionally accepts only `opt_trades` so validation data cannot leak
    into policy selection. Evaluation parameters are chosen by the objective; the funded
    phase (risk per trade, CPPI) is then chosen to maximise expected payouts subject to
    P(lose funded within 12m) <= `max_lose_funded`.

    - fastest: lowest median days to pass among policies meeting P(non-pass) <= max_fail.
    - cheapest: lowest expected challenge fees among the same feasible set.
    - max_value: highest expected 12-month net value
      E[payouts] - E[fees] * (1 + P(lose funded)), funded-risk constraint first.
    """
    if objective not in ("fastest", "cheapest", "max_value"):
        raise ValueError(f"Unknown objective '{objective}'. Expected fastest, cheapest, or max_value.")
    if not (0.0 <= max_fail <= 1.0):
        raise ValueError("max_fail must be in [0, 1].")
    if not (0.0 <= max_lose_funded <= 1.0):
        raise ValueError("max_lose_funded must be in [0, 1].")

    policies = grid if grid is not None else build_default_grid(rules)
    if not policies:
        raise ValueError("Policy grid is empty.")

    summaries = simulate_challenge_batch(opt_trades, rules, policies)
    notes: list[str] = []

    min_finished = min(10, max(1, summaries[0].n_total // 4)) if summaries else 1

    feasible_indices: list[int] = []
    for idx, sm in enumerate(summaries):
        meets_fail = sm.non_pass_rate <= max_fail + 1e-9
        has_passes = sm.n_passed > 0 and sm.n_finished >= min_finished
        if meets_fail and has_passes:
            feasible_indices.append(idx)

    feasible_found = len(feasible_indices) > 0
    candidate_indices = feasible_indices
    if not feasible_found:
        notes.append(
            f"No policy in the grid met P(non-pass) <= {max_fail * 100:.1f}% on the optimisation "
            "split; falling back to the lowest failure-rate policy."
        )
        candidate_indices = [
            i for i, sm in enumerate(summaries) if sm.n_passed > 0 and sm.n_finished >= min_finished
        ]
        if not candidate_indices:
            candidate_indices = list(range(len(policies)))

    opt_pool = build_week_blocks(opt_trades)
    funded_cache: dict[tuple[object, ...], FundedSummary] = {}
    guard_choice: dict[tuple[object, ...], tuple[SizingPolicy, list[FundedCandidate], bool]] = {}

    def _funded_for(idx: int) -> tuple[SizingPolicy, list[FundedCandidate], bool]:
        gkey = _guard_key(policies[idx])
        if gkey not in guard_choice:
            guard_choice[gkey] = select_funded_variant(
                opt_trades,
                rules,
                policies[idx],
                max_lose_funded=max_lose_funded,
                n_paths=funded_paths,
                seed=seed,
                pool=opt_pool,
                cache=funded_cache,
            )
        return guard_choice[gkey]

    net_monthly_map: dict[int, float] = {}
    funded_ok_map: dict[int, bool] = {}
    if objective == "max_value":
        for idx in candidate_indices:
            funded_pol, _, funded_ok = _funded_for(idx)
            fsum = funded_cache[_funded_key(funded_pol)]
            net_monthly_map[idx] = expected_net_12m_usd(summaries[idx].expected_fees, fsum) / 12.0
            funded_ok_map[idx] = funded_ok

    def _rank_key(idx: int) -> tuple[float, ...]:
        sm = summaries[idx]
        gate = 0.0 if feasible_found else sm.non_pass_rate
        fees = sm.expected_fees if np.isfinite(sm.expected_fees) else 1e12
        med_days = sm.days_median if sm.days_median is not None else 1e9
        if objective == "fastest":
            return (gate, med_days, sm.non_pass_rate, fees)
        if objective == "cheapest":
            return (gate, fees, sm.non_pass_rate, med_days)
        net_m = net_monthly_map.get(idx, float("-inf"))
        return (
            gate,
            0.0 if funded_ok_map.get(idx, False) else 1.0,
            -net_m if np.isfinite(net_m) else 1e18,
            sm.non_pass_rate,
        )

    best_idx = min(candidate_indices, key=_rank_key)
    best_funded_policy, funded_candidates, funded_ok = _funded_for(best_idx)
    best_policy = policies[best_idx].model_copy(
        update={
            "funded_risk_pct": best_funded_policy.funded_risk_pct,
            "keep_buffer_pct": best_funded_policy.keep_buffer_pct,
            "cppi_cap": best_funded_policy.cppi_cap,
            "cppi_c0": best_funded_policy.cppi_c0,
        }
    )
    if not funded_ok:
        notes.append(
            f"No funded variant met P(lose funded within 12m) <= {max_lose_funded * 100:.0f}% on the "
            "optimisation slice; the lowest-risk funded variant is shown. Treat the funded plan as HIGH RISK."
        )

    records: list[GridCandidateRecord] = []
    feasible_set = set(feasible_indices)
    for idx, (pol, sm) in enumerate(zip(policies, summaries)):
        if objective == "fastest":
            score = -(sm.days_median if sm.days_median is not None else 1e9)
        elif objective == "cheapest":
            score = -(sm.expected_fees if np.isfinite(sm.expected_fees) else 1e12)
        else:
            score = net_monthly_map.get(idx, float("-inf"))
        records.append(
            GridCandidateRecord(
                policy=pol,
                summary=sm,
                feasible=idx in feasible_set,
                score=float(score),
                expected_net_monthly_usd=net_monthly_map.get(idx),
            )
        )

    return TrainOptimisationOutput(
        objective=objective,
        max_fail=max_fail,
        max_lose_funded=max_lose_funded,
        best_policy=best_policy,
        best_opt_summary=summaries[best_idx],
        grid_records=records,
        feasible_found=feasible_found,
        funded_candidates=funded_candidates,
        funded_constraint_met=funded_ok,
        notes=notes,
    )


def run_honest_optimisation(
    dataset: TradeDataset,
    rules: PropRules,
    objective: ObjectiveName = "fastest",
    max_fail: float = 0.20,
    n_bootstrap: int = 2000,
    seed: int = 42,
    grid: list[SizingPolicy] | None = None,
    train_frac: float = 0.60,
    max_lose_funded: float = DEFAULT_MAX_LOSE_FUNDED,
    funded_paths: int = FUNDED_SELECT_PATHS,
) -> OptimizationResult:
    """Run 60/40 chronological optimisation + out-of-sample validation + bootstrap + funded projection."""
    opt_trades, val_trades = dataset.split_chronological(train_frac=train_frac)

    train_out = optimise_on_train(
        opt_trades=opt_trades,
        rules=rules,
        objective=objective,
        max_fail=max_fail,
        grid=grid,
        seed=seed,
        max_lose_funded=max_lose_funded,
        funded_paths=funded_paths,
    )
    best_pol = train_out.best_policy
    opt_summary = train_out.best_opt_summary

    # Out-of-sample validation on last 40% (never seen by optimiser)
    val_summary = simulate_rolling_starts(val_trades, rules, best_pol)

    # Weekly block bootstrap on the optimisation slice (zero leakage from validation)
    opt_pool = build_week_blocks(opt_trades)
    boot_summary = simulate_bootstrap_challenge(
        opt_trades,
        rules,
        best_pol,
        n_paths=n_bootstrap,
        seed=seed,
        pool=opt_pool,
    )
    # Fresh seed: the reported funded figures are not the ones used to pick the variant.
    funded_summary = simulate_funded_bootstrap(
        opt_trades,
        rules,
        best_pol,
        n_paths=n_bootstrap,
        seed=seed + 1,
        pool=opt_pool,
    )
    # Out-of-sample funded check: bootstrap weeks of the unseen validation slice.
    funded_val_summary = simulate_funded_bootstrap(
        val_trades,
        rules,
        best_pol,
        n_paths=n_bootstrap,
        seed=seed + 2,
    )
    failure_attr = analyze_failures(val_trades, rules, best_pol, batch_summary=val_summary)

    warnings = list(train_out.notes) + rules.approximation_notes
    if val_summary.n_finished < 15:
        warnings.append(
            f"Validation slice has only {val_summary.n_finished} finished rolling starts; "
            "out-of-sample variance may be high."
        )
    if opt_summary.pass_rate - val_summary.pass_rate >= 0.15 or (
        opt_summary.pass_rate > 0 and val_summary.pass_rate < 0.6 * opt_summary.pass_rate
    ):
        warnings.append(
            f"Possible overfit / regime shift: validation pass rate ({val_summary.pass_rate * 100:.1f}%) "
            f"is substantially lower than optimisation ({opt_summary.pass_rate * 100:.1f}%)."
        )
    if (
        opt_summary.days_median is not None
        and val_summary.days_median is not None
        and math.isfinite(opt_summary.days_median)
        and math.isfinite(val_summary.days_median)
        and val_summary.days_median > opt_summary.days_median * 1.5
    ):
        warnings.append(
            f"Validation median time-to-pass ({val_summary.days_median:.1f}d) is >1.5x slower "
            f"than optimisation ({opt_summary.days_median:.1f}d)."
        )
    for label, fsum in (("optimisation", funded_summary), ("validation", funded_val_summary)):
        if fsum.p_lose_account > max_lose_funded + 1e-12:
            warnings.append(
                f"Funded risk: P(lose funded account within 12m) is {fsum.p_lose_account * 100:.1f}% on the "
                f"{label}-slice bootstrap, above the {max_lose_funded * 100:.0f}% limit."
            )

    return OptimizationResult(
        objective=objective,
        max_fail=max_fail,
        rules=rules,
        best_policy=best_pol,
        opt_summary=opt_summary,
        val_summary=val_summary,
        boot_summary=boot_summary,
        funded_summary=funded_summary,
        failure_attribution=failure_attr,
        grid_records=train_out.grid_records,
        opt_trade_count=len(opt_trades),
        val_trade_count=len(val_trades),
        feasible_found=train_out.feasible_found,
        warnings=warnings,
        funded_val_summary=funded_val_summary,
        max_lose_funded=max_lose_funded,
        funded_constraint_met=train_out.funded_constraint_met,
        funded_candidates=train_out.funded_candidates,
    )
