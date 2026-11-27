"""Funded-phase 12-month projection simulator with payouts, kept buffer, and CPPI sizing.

Both the deterministic historical run (`simulate_funded_single`) and the weekly block
bootstrap (`simulate_funded_bootstrap`) execute the same per-path state machine
(`_run_funded_path`), so the two can never drift apart.

Funded-account mechanics (per path, time 0 = funding day):
- Entry reserves `risk = ref_balance * eff_risk_pct / 100`; PnL `R * risk` is realised at exit.
- Daily loss (firm): breach if `day_start_ref - balance > day_start_ref * daily_loss_pct / 100`
  at any realised exit; `day_start_ref` is the closed balance at the daily reset hour.
- Max loss (firm): static floor `initial * (1 - max_loss_pct/100)`, or trailing floor
  `HWM * (1 - max_loss_pct/100)`; with `rules.trailing_lock_at_initial` the trailing floor
  locks permanently at the initial balance once it reaches it.
- Payouts every `payout_frequency_days`: everything above
  `initial * (1 + keep_buffer_pct/100)` is withdrawn; the trader receives
  `gross * profit_split / 100`. A withdrawal is not a trading loss: it lowers the trailing
  high-water mark by the same amount and never counts toward the daily loss.
- Consistency (optional): a payout is deferred while the best day of the payout cycle
  (completed days' final PnL, plus the current day) exceeds `consistency_pct` % of the
  cycle profit.
- CPPI (optional, `cppi_c0` set): risk multiplier `min(cppi_cap, cushion_pct / cppi_c0)` with
  `cushion_pct = max(0, balance - max_loss_floor) / initial * 100`. Risk shrinks toward zero as
  the account approaches the floor and grows (up to the cap) as the cushion builds.
- Own daily stop, per-symbol daily cap, and max open risk entry guards are shared with the
  evaluation policy.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from proppilot.ingest import DAY_MS, HOUR_MS, TradeDataset
from proppilot.policy import FundedPolicy, SizingPolicy
from proppilot.rules import PropRules
from proppilot.sim.bootstrap import WEEK_MS, WeekPool, build_week_blocks
from proppilot.sim.models import FundedSummary

_EPS = 1e-9


@dataclass(frozen=True)
class _FundedParams:
    initial: float
    kept_target: float
    reset_offset_ms: int
    is_static: bool
    trail_lock: bool
    static_floor: float
    trail_factor: float
    daily_loss_frac: float
    has_consistency: bool
    consistency_frac: float
    split_frac: float
    payout_freq_ms: int
    firm_cap: Optional[float]
    base_risk_pct: float
    use_cppi: bool
    cppi_cap: float
    cppi_c0: float
    own_daily_frac: float
    sym_cap_frac: Optional[float]
    max_open_frac: Optional[float]
    use_init_basis: bool


def _build_params(rules: PropRules, policy: SizingPolicy | FundedPolicy) -> _FundedParams:
    if isinstance(policy, SizingPolicy):
        fpol = policy.funded_config()
        own_daily_pct = policy.own_daily_stop_pct
        sym_cap_pct = policy.per_symbol_daily_cap_pct
        max_open_pct = policy.max_open_risk_pct
        use_init_basis = policy.risk_basis == "initial_balance"
    else:
        fpol = policy
        own_daily_pct = min(rules.daily_loss_pct * 0.65, max(0.5, rules.daily_loss_pct - 0.5))
        sym_cap_pct = None
        max_open_pct = own_daily_pct
        use_init_basis = False

    initial = float(rules.account_size)
    max_loss_frac = rules.max_loss_pct / 100.0
    is_static = rules.max_loss_type == "static"
    use_cppi = fpol.cppi_c0 is not None and fpol.cppi_c0 > 0.0
    return _FundedParams(
        initial=initial,
        kept_target=initial * (1.0 + fpol.keep_buffer_pct / 100.0),
        reset_offset_ms=rules.daily_reset * HOUR_MS,
        is_static=is_static,
        trail_lock=(not is_static) and rules.trailing_lock_at_initial,
        static_floor=initial * (1.0 - max_loss_frac),
        trail_factor=1.0 - max_loss_frac,
        daily_loss_frac=rules.daily_loss_pct / 100.0,
        has_consistency=rules.consistency_pct is not None,
        consistency_frac=(rules.consistency_pct / 100.0) if rules.consistency_pct is not None else 1.0,
        split_frac=rules.profit_split / 100.0,
        payout_freq_ms=rules.payout_frequency_days * DAY_MS,
        firm_cap=rules.max_risk_per_trade_pct,
        base_risk_pct=fpol.risk_pct,
        use_cppi=use_cppi,
        cppi_cap=fpol.cppi_cap,
        cppi_c0=float(fpol.cppi_c0) if use_cppi and fpol.cppi_c0 is not None else 0.0,
        own_daily_frac=own_daily_pct / 100.0,
        sym_cap_frac=sym_cap_pct / 100.0 if sym_cap_pct is not None else None,
        max_open_frac=max_open_pct / 100.0 if max_open_pct is not None else None,
        use_init_basis=use_init_basis,
    )


def _run_funded_path(
    prm: _FundedParams,
    e_list: Sequence[int],
    x_list: Sequence[int],
    s_list: Sequence[int],
    r_list: Sequence[float],
    horizon_ms: int,
    origin_ms: int = 0,
) -> tuple[Optional[str], float, list[tuple[int, float]]]:
    """Run one funded account over trades sorted by entry time (times relative to funding).

    `origin_ms` is the absolute UTC timestamp of relative time 0 (used for daily resets).
    Returns (loss_reason or None, final_balance, [(payout_time_rel_ms, net_payout_usd), ...]).
    """
    initial = prm.initial
    kept_target = prm.kept_target
    is_static = prm.is_static
    trail_lock = prm.trail_lock
    trail_factor = prm.trail_factor
    daily_loss_frac = prm.daily_loss_frac
    has_consistency = prm.has_consistency
    consistency_frac = prm.consistency_frac
    split_frac = prm.split_frac
    payout_freq_ms = prm.payout_freq_ms
    firm_cap = prm.firm_cap
    base_risk_pct = prm.base_risk_pct
    use_cppi = prm.use_cppi
    cppi_cap = prm.cppi_cap
    cppi_c0 = prm.cppi_c0
    own_daily_frac = prm.own_daily_frac
    sym_cap_frac = prm.sym_cap_frac
    max_open_frac = prm.max_open_frac
    use_init_basis = prm.use_init_basis
    reset_base = origin_ms - prm.reset_offset_ms

    balance = initial
    hwm = initial
    locked = False
    floor = prm.static_floor if is_static else initial * trail_factor
    day_start_ref = initial
    open_risk = 0.0
    cur_day = reset_base // DAY_MS
    day_pnl = 0.0
    cycle_best_day = 0.0  # best COMPLETED day's final PnL in the current payout cycle
    cycle_start_balance = initial
    sym_today: dict[int, float] = {}
    heap: list[tuple[int, int, int, float, float]] = []
    seq = 0
    next_pay = payout_freq_ms
    payouts: list[tuple[int, float]] = []
    loss_reason: Optional[str] = None

    def refresh_floor() -> None:
        nonlocal floor, locked
        if is_static:
            return
        raw = hwm * trail_factor
        if trail_lock and (locked or raw >= initial):
            locked = True
            floor = initial
        else:
            floor = raw

    def new_day(day: int) -> None:
        nonlocal cur_day, day_start_ref, day_pnl, cycle_best_day
        if day_pnl > cycle_best_day:
            cycle_best_day = day_pnl
        cur_day = day
        day_start_ref = balance
        day_pnl = 0.0
        sym_today.clear()

    def run_payouts(t_lim: int) -> None:
        nonlocal next_pay, balance, hwm, day_start_ref, cycle_start_balance, cycle_best_day, day_pnl
        while next_pay <= t_lim and next_pay <= horizon_ms:
            p_day = (reset_base + next_pay) // DAY_MS
            if p_day != cur_day:
                new_day(p_day)
            if balance > kept_target + _EPS:
                ok = True
                if has_consistency:
                    cycle_profit = balance - cycle_start_balance
                    best = cycle_best_day if cycle_best_day > day_pnl else day_pnl
                    if cycle_profit <= 0.0 or best > cycle_profit * consistency_frac + _EPS:
                        ok = False
                if ok:
                    gross = balance - kept_target
                    balance = kept_target
                    if not is_static:
                        hwm -= gross
                        refresh_floor()
                    if balance < day_start_ref:
                        day_start_ref = balance
                    cycle_start_balance = balance
                    cycle_best_day = 0.0
                    day_pnl = 0.0
                    payouts.append((next_pay, gross * split_frac))
            next_pay += payout_freq_ms

    def do_exit(ex_ms: int, ex_r: float, ex_res: float) -> bool:
        nonlocal balance, open_risk, day_pnl, hwm, loss_reason
        run_payouts(ex_ms)
        ex_day = (reset_base + ex_ms) // DAY_MS
        if ex_day != cur_day:
            new_day(ex_day)
        pnl = ex_r * ex_res
        open_risk -= ex_res
        balance += pnl
        day_pnl += pnl
        if (day_start_ref - balance) > day_start_ref * daily_loss_frac + _EPS:
            loss_reason = "daily_loss"
            return True
        if balance > hwm:
            hwm = balance
            refresh_floor()
        if balance < floor - _EPS:
            loss_reason = "max_loss"
            return True
        return False

    heappush = heapq.heappush
    heappop = heapq.heappop
    for j in range(len(e_list)):
        t_entry = e_list[j]
        if t_entry >= horizon_ms:
            break
        # Exits strictly before the entry, and normal exits at the same timestamp, come first.
        while heap and (heap[0][0] < t_entry or (heap[0][0] == t_entry and heap[0][1] == 0)):
            ex_ms, _, _, ex_r, ex_res = heappop(heap)
            if do_exit(ex_ms, ex_r, ex_res):
                return loss_reason, balance, payouts
        run_payouts(t_entry)
        en_day = (reset_base + t_entry) // DAY_MS
        if en_day != cur_day:
            new_day(en_day)

        eff_pct = base_risk_pct
        if use_cppi:
            cushion_pct = (balance - floor) / initial * 100.0
            mult = cushion_pct / cppi_c0 if cushion_pct > 0.0 else 0.0
            if mult > cppi_cap:
                mult = cppi_cap
            eff_pct *= mult
        if firm_cap is not None and eff_pct > firm_cap:
            eff_pct = firm_cap
        ref_bal = initial if use_init_basis else balance
        new_risk = ref_bal * (eff_pct / 100.0)
        if new_risk <= 0.0:
            continue

        realised_loss = day_start_ref - balance
        if realised_loss < 0.0:
            realised_loss = 0.0
        if realised_loss + open_risk + new_risk > day_start_ref * own_daily_frac + _EPS:
            continue
        sym_id = s_list[j]
        if sym_cap_frac is not None:
            if sym_today.get(sym_id, 0.0) + new_risk > day_start_ref * sym_cap_frac + _EPS:
                continue
        if max_open_frac is not None and open_risk + new_risk > ref_bal * max_open_frac + _EPS:
            continue

        open_risk += new_risk
        if sym_cap_frac is not None:
            sym_today[sym_id] = sym_today.get(sym_id, 0.0) + new_risk
        seq += 1
        t_exit = x_list[j]
        heappush(heap, (t_exit, 0 if t_exit > t_entry else 1, seq, r_list[j], new_risk))

    while heap:
        ex_ms, _, _, ex_r, ex_res = heappop(heap)
        if ex_ms > horizon_ms:
            break
        if do_exit(ex_ms, ex_r, ex_res):
            return loss_reason, balance, payouts
    run_payouts(horizon_ms)
    return None, balance, payouts


def simulate_funded_bootstrap(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy | FundedPolicy,
    n_paths: int = 2000,
    seed: int = 123,
    horizon_months: int = 12,
    pool: WeekPool | None = None,
) -> FundedSummary:
    """Simulate N funded accounts over `horizon_months` via weekly block bootstrap.

    See the module docstring for the exact funded rules. Monthly payout figures average every
    path, including the zero months after an account is lost, so they already price in
    account losses (P(lose) is reported separately).
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0.")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be > 0.")

    prm = _build_params(rules, policy)
    week_pool = pool if pool is not None else build_week_blocks(dataset)
    rng = np.random.default_rng(seed)

    horizon_days = 365 if horizon_months == 12 else int(round(horizon_months * 30.4375))
    horizon_ms = horizon_days * DAY_MS
    n_weeks_needed = max(1, (horizon_days + 6) // 7)
    week_matrix = rng.integers(0, len(week_pool), size=(n_paths, n_weeks_needed))
    month_ms = horizon_ms / float(horizon_months)

    blocks = week_pool.blocks
    monthly_payouts = np.zeros((n_paths, horizon_months), dtype=np.float64)
    lost_flags = np.zeros(n_paths, dtype=bool)
    reasons: Counter[str] = Counter()

    for path_idx in range(n_paths):
        e_path: list[int] = []
        x_path: list[int] = []
        s_path: list[int] = []
        r_path: list[float] = []
        for w_pos in range(n_weeks_needed):
            off = w_pos * WEEK_MS
            if off >= horizon_ms:
                break
            blk = blocks[int(week_matrix[path_idx, w_pos])]
            if not blk.entry_rel_ms:
                continue
            e_path.extend([off + v for v in blk.entry_rel_ms])
            x_path.extend([off + v for v in blk.exit_rel_ms])
            s_path.extend(blk.symbol_ids)
            r_path.extend(blk.r_vals)

        reason, _, payouts = _run_funded_path(prm, e_path, x_path, s_path, r_path, horizon_ms)
        if reason is not None:
            lost_flags[path_idx] = True
            reasons[reason] += 1
        row = monthly_payouts[path_idx]
        for t_pay, net in payouts:
            m_idx = int((t_pay - 1) // month_ms)
            row[min(max(m_idx, 0), horizon_months - 1)] += net

    path_totals = np.sum(monthly_payouts, axis=1)
    path_monthly_avg = path_totals / float(horizon_months)
    per_month_mean = np.mean(monthly_payouts, axis=0)
    mean_m = float(np.mean(path_monthly_avg))
    med_m = float(np.median(path_monthly_avg))
    initial = prm.initial

    return FundedSummary(
        n_paths=n_paths,
        account_size=initial,
        p_lose_account=float(np.mean(lost_flags)),
        fail_reason_counts=dict(reasons),
        monthly_payout_mean_usd=mean_m,
        monthly_payout_median_usd=med_m,
        monthly_payout_p10_usd=float(np.percentile(path_monthly_avg, 10)),
        monthly_payout_p25_usd=float(np.percentile(path_monthly_avg, 25)),
        monthly_payout_p75_usd=float(np.percentile(path_monthly_avg, 75)),
        monthly_payout_p90_usd=float(np.percentile(path_monthly_avg, 90)),
        monthly_payout_mean_pct=mean_m / initial * 100.0,
        monthly_payout_median_pct=med_m / initial * 100.0,
        total_12m_payout_mean_usd=float(np.mean(path_totals)),
        total_12m_payout_median_usd=float(np.median(path_totals)),
        prob_zero_payout=float(np.mean(path_totals <= 1e-6)),
        per_month_mean_usd=per_month_mean,
        path_monthly_avg_usd=path_monthly_avg,
        path_12m_total_usd=path_totals,
    )


def simulate_funded_single(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy | FundedPolicy,
    horizon_days: int = 365,
) -> tuple[bool, str | None, float, list[tuple[float, float]]]:
    """Deterministic funded run over the historical trades, funded at the first entry.

    Returns (lost, fail_reason, final_balance, payouts as [(day_since_start, net_usd), ...]).
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0.")
    prm = _build_params(rules, policy)
    start_ms = dataset.start_ms
    e_rel = (dataset.entry_ms - start_ms).tolist()
    x_rel = (dataset.exit_ms - start_ms).tolist()
    reason, balance, payouts = _run_funded_path(
        prm,
        e_rel,
        x_rel,
        dataset.symbol_ids.tolist(),
        dataset.r_vals.tolist(),
        horizon_days * DAY_MS,
        origin_ms=start_ms,
    )
    return (
        reason is not None,
        reason,
        balance,
        [(t / DAY_MS, net) for t, net in payouts],
    )
