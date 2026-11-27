"""Event-driven challenge evaluation simulator (auditable scalar + vectorized batch).

Note on intraday mark-to-market approximation:
PropPilot operates on trade-level logs (entry_time, exit_time, symbol, R).
Unrealised intra-trade drawdown is approximated at trade exit times; entry guards
protect against simultaneous multi-trade exposure via open-risk + realised daily loss.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from proppilot.ingest import (
    DAY_MS,
    EV_ENTRY,
    HOUR_MS,
    TradeDataset,
)
from proppilot.policy import SizingPolicy
from proppilot.rules import PropRules
from proppilot.sim.models import (
    FAIL_DAILY_LOSS,
    FAIL_MAX_DAYS,
    FAIL_MAX_LOSS,
    FAIL_NONE,
    FAIL_RESTART,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_RESTART,
    OUTCOME_UNFINISHED,
    BatchChallengeSummary,
    ChallengeRunResult,
    FailureAttribution,
    summarize_batch_arrays,
)

_EPS = 1e-9
_INF_MS = 10**16


def _reset_day_index(ts_ms: int, reset_offset_ms: int) -> int:
    return (ts_ms - reset_offset_ms) // DAY_MS


def _format_utc_day(day_idx: int, reset_offset_ms: int) -> str:
    dt = datetime.fromtimestamp((day_idx * DAY_MS + reset_offset_ms) / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def simulate_challenge_single(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy,
    start_trade_idx: int = 0,
    record_curve: bool = False,
    record_breach_detail: bool = False,
) -> ChallengeRunResult:
    """Simulate a single challenge starting at trade index `start_trade_idx`.

    Only trades entering at or after `dataset.entry_ms[start_trade_idx]` are eligible.
    """
    initial_balance = float(rules.account_size)
    start_ms = int(dataset.entry_ms[start_trade_idx])

    if not rules.phases:
        curve_d = np.array([0.0], dtype=np.float64) if record_curve else None
        curve_b = np.array([initial_balance], dtype=np.float64) if record_curve else None
        return ChallengeRunResult(
            outcome="pass",
            days_to_pass=0.0,
            fail_reason=None,
            fail_phase=None,
            final_balance=initial_balance,
            trades_taken=0,
            start_ms=start_ms,
            end_ms=start_ms,
            curve_days=curve_d,
            curve_balances=curve_b,
        )

    ev_ms, ev_kind, ev_tidx = dataset.events
    r_vals = dataset.r_vals
    sym_ids = dataset.symbol_ids
    symbols = dataset.symbols

    reset_offset_ms = rules.daily_reset * HOUR_MS
    is_static_max = rules.max_loss_type == "static"
    trail_lock = (not is_static_max) and rules.trailing_lock_at_initial
    has_consistency = rules.consistency_pct is not None
    consistency_frac = (rules.consistency_pct / 100.0) if has_consistency else 1.0
    restart_ms = policy.restart_after_days * DAY_MS if policy.restart_after_days is not None else _INF_MS

    n_phases = len(rules.phases)
    phase_idx = 0
    phase = rules.phases[0]
    phase_start_ms = start_ms
    phase_max_ms = phase.max_days * DAY_MS if phase.max_days is not None else _INF_MS
    target_balance = initial_balance * (1.0 + phase.target_pct / 100.0)
    target_gain = target_balance - initial_balance

    balance = initial_balance
    hwm = initial_balance
    day_start_ref = initial_balance
    open_risk = 0.0
    trading_days_count = 0
    last_entry_day = -10**9
    cur_day = _reset_day_index(start_ms, reset_offset_ms)
    current_day_pnl = 0.0
    best_day_pnl = 0.0  # best COMPLETED day's final PnL in the current phase
    trades_taken = 0

    open_reserved: dict[int, float] = {}
    sym_risk_today: dict[int, float] = {}
    day_closed_trades: list[tuple[str, float]] = []

    curve_t: list[float] = [0.0] if record_curve else []
    curve_b: list[float] = [initial_balance] if record_curve else []

    start_ev_pos = int(dataset._entry_to_ev[start_trade_idx])
    n_events = len(ev_ms)

    for ev_pos in range(start_ev_pos, n_events):
        t_ms = int(ev_ms[ev_pos])
        kind = int(ev_kind[ev_pos])
        tidx = int(ev_tidx[ev_pos])

        if kind == EV_ENTRY and tidx < start_trade_idx:
            continue

        # Check phase max_days timeout and policy restart timeout prior to processing event.
        if (t_ms - phase_start_ms) > phase_max_ms:
            if record_curve:
                curve_t.append((t_ms - start_ms) / DAY_MS)
                curve_b.append(balance)
            return ChallengeRunResult(
                outcome="fail",
                days_to_pass=None,
                fail_reason="max_days_timeout",
                fail_phase=phase_idx,
                final_balance=balance,
                trades_taken=trades_taken,
                start_ms=start_ms,
                end_ms=t_ms,
                curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
                curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
            )

        if (t_ms - start_ms) > restart_ms:
            if record_curve:
                curve_t.append((t_ms - start_ms) / DAY_MS)
                curve_b.append(balance)
            return ChallengeRunResult(
                outcome="restart",
                days_to_pass=None,
                fail_reason="restart_timeout",
                fail_phase=phase_idx,
                final_balance=balance,
                trades_taken=trades_taken,
                start_ms=start_ms,
                end_ms=t_ms,
                curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
                curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
            )

        ev_day = _reset_day_index(t_ms, reset_offset_ms)
        if ev_day != cur_day:
            if has_consistency:
                best_day_pnl = max(best_day_pnl, current_day_pnl)
            cur_day = ev_day
            day_start_ref = balance
            current_day_pnl = 0.0
            sym_risk_today.clear()
            if record_breach_detail:
                day_closed_trades.clear()

        if kind == EV_ENTRY:
            eff_risk_pct = policy.risk_pct
            if policy.near_frac is not None and policy.near_mult != 1.0 and target_gain > 0.0:
                progress = (balance - initial_balance) / target_gain
                if progress >= policy.near_frac - 1e-12:
                    eff_risk_pct *= policy.near_mult

            if policy.behind_dd_pct is not None and policy.behind_mult != 1.0 and hwm > 0.0:
                dd_pct = (hwm - balance) / hwm * 100.0
                if dd_pct >= policy.behind_dd_pct - 1e-12:
                    eff_risk_pct *= policy.behind_mult

            if rules.max_risk_per_trade_pct is not None:
                eff_risk_pct = min(eff_risk_pct, rules.max_risk_per_trade_pct)

            ref_bal = initial_balance if policy.risk_basis == "initial_balance" else balance
            new_risk = max(0.0, ref_bal * (eff_risk_pct / 100.0))

            if new_risk <= 0.0:
                continue

            realised_day_loss = max(0.0, day_start_ref - balance)
            own_daily_limit = day_start_ref * (policy.own_daily_stop_pct / 100.0)
            if realised_day_loss + open_risk + new_risk > own_daily_limit + _EPS:
                continue

            sym_id = int(sym_ids[tidx])
            if policy.per_symbol_daily_cap_pct is not None:
                sym_cap = day_start_ref * (policy.per_symbol_daily_cap_pct / 100.0)
                if sym_risk_today.get(sym_id, 0.0) + new_risk > sym_cap + _EPS:
                    continue

            if policy.max_open_risk_pct is not None:
                open_cap = ref_bal * (policy.max_open_risk_pct / 100.0)
                if open_risk + new_risk > open_cap + _EPS:
                    continue

            open_reserved[tidx] = new_risk
            open_risk += new_risk
            sym_risk_today[sym_id] = sym_risk_today.get(sym_id, 0.0) + new_risk
            trades_taken += 1
            if ev_day != last_entry_day:
                trading_days_count += 1
                last_entry_day = ev_day
        else:
            reserved = open_reserved.pop(tidx, None)
            if reserved is None:
                continue

            r_val = float(r_vals[tidx])
            pnl = r_val * reserved
            open_risk -= reserved
            balance += pnl
            current_day_pnl += pnl
            if record_breach_detail:
                day_closed_trades.append((symbols[int(sym_ids[tidx])], r_val))
            if record_curve:
                curve_t.append((t_ms - start_ms) / DAY_MS)
                curve_b.append(balance)

            daily_limit = day_start_ref * (rules.daily_loss_pct / 100.0)
            if (day_start_ref - balance) > daily_limit + _EPS:
                return _build_breach_result(
                    "daily_loss",
                    phase_idx,
                    balance,
                    trades_taken,
                    start_ms,
                    t_ms,
                    ev_day,
                    reset_offset_ms,
                    symbols[int(sym_ids[tidx])],
                    day_closed_trades,
                    record_breach_detail,
                    curve_t,
                    curve_b,
                    record_curve,
                )

            if balance > hwm:
                hwm = balance
            loss_floor = (
                initial_balance * (1.0 - rules.max_loss_pct / 100.0)
                if is_static_max
                else hwm * (1.0 - rules.max_loss_pct / 100.0)
            )
            if trail_lock and loss_floor > initial_balance:
                loss_floor = initial_balance
            if balance < loss_floor - _EPS:
                return _build_breach_result(
                    "max_loss",
                    phase_idx,
                    balance,
                    trades_taken,
                    start_ms,
                    t_ms,
                    ev_day,
                    reset_offset_ms,
                    symbols[int(sym_ids[tidx])],
                    day_closed_trades,
                    record_breach_detail,
                    curve_t,
                    curve_b,
                    record_curve,
                )

            consistency_ok = True
            if has_consistency:
                total_profit = balance - initial_balance
                best_day = max(best_day_pnl, current_day_pnl)
                if total_profit <= 0.0 or best_day > total_profit * consistency_frac + _EPS:
                    consistency_ok = False

            if balance >= target_balance - _EPS and trading_days_count >= phase.min_days and consistency_ok:
                if phase_idx + 1 < n_phases:
                    phase_idx += 1
                    phase = rules.phases[phase_idx]
                    phase_start_ms = t_ms
                    phase_max_ms = phase.max_days * DAY_MS if phase.max_days is not None else _INF_MS
                    target_balance = initial_balance * (1.0 + phase.target_pct / 100.0)
                    target_gain = target_balance - initial_balance
                    balance = initial_balance
                    hwm = initial_balance
                    day_start_ref = initial_balance
                    open_risk = 0.0
                    trading_days_count = 0
                    last_entry_day = -10**9
                    current_day_pnl = 0.0
                    best_day_pnl = 0.0
                    open_reserved.clear()
                    sym_risk_today.clear()
                    if record_breach_detail:
                        day_closed_trades.clear()
                    if record_curve:
                        # Record phase reset step cleanly on the combined equity path.
                        curve_t.append((t_ms - start_ms) / DAY_MS)
                        curve_b.append(initial_balance)
                else:
                    return ChallengeRunResult(
                        outcome="pass",
                        days_to_pass=(t_ms - start_ms) / DAY_MS,
                        fail_reason=None,
                        fail_phase=None,
                        final_balance=balance,
                        trades_taken=trades_taken,
                        start_ms=start_ms,
                        end_ms=t_ms,
                        curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
                        curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
                    )

    data_end_ms = dataset.end_ms
    if (data_end_ms - phase_start_ms) > phase_max_ms:
        return ChallengeRunResult(
            outcome="fail",
            days_to_pass=None,
            fail_reason="max_days_timeout",
            fail_phase=phase_idx,
            final_balance=balance,
            trades_taken=trades_taken,
            start_ms=start_ms,
            end_ms=data_end_ms,
            curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
            curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
        )

    if (data_end_ms - start_ms) > restart_ms:
        return ChallengeRunResult(
            outcome="restart",
            days_to_pass=None,
            fail_reason="restart_timeout",
            fail_phase=phase_idx,
            final_balance=balance,
            trades_taken=trades_taken,
            start_ms=start_ms,
            end_ms=data_end_ms,
            curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
            curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
        )

    return ChallengeRunResult(
        outcome="unfinished",
        days_to_pass=None,
        fail_reason=None,
        fail_phase=None,
        final_balance=balance,
        trades_taken=trades_taken,
        start_ms=start_ms,
        end_ms=data_end_ms,
        curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
        curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
    )


def _build_breach_result(
    reason: str,
    phase_idx: int,
    balance: float,
    trades_taken: int,
    start_ms: int,
    t_ms: int,
    ev_day: int,
    reset_offset_ms: int,
    breach_symbol: str,
    day_closed_trades: list[tuple[str, float]],
    record_breach_detail: bool,
    curve_t: list[float],
    curve_b: list[float],
    record_curve: bool,
) -> ChallengeRunResult:
    day_utc = _format_utc_day(ev_day, reset_offset_ms) if record_breach_detail else None
    day_syms = tuple(sorted({s for s, _ in day_closed_trades})) if record_breach_detail else ()
    day_r_sum = round(sum(r for _, r in day_closed_trades), 3) if record_breach_detail else None
    return ChallengeRunResult(
        outcome="fail",
        days_to_pass=None,
        fail_reason=reason,
        fail_phase=phase_idx,
        final_balance=balance,
        trades_taken=trades_taken,
        start_ms=start_ms,
        end_ms=t_ms,
        breach_day_utc=day_utc,
        breach_symbol=breach_symbol if record_breach_detail else None,
        breach_day_symbols=day_syms,
        breach_day_r_sum=day_r_sum,
        curve_days=np.array(curve_t, dtype=np.float64) if record_curve else None,
        curve_balances=np.array(curve_b, dtype=np.float64) if record_curve else None,
    )


def simulate_challenge_batch(
    dataset: TradeDataset,
    rules: PropRules,
    policies: list[SizingPolicy],
    start_trade_indices: np.ndarray | None = None,
) -> list[BatchChallengeSummary]:
    """Vectorized batch evaluation across all rolling starts and policies simultaneously.

    Maintains contiguous active window `[lo:hi, :]` across rolling starts so 200 policies x
    1,500 rolling starts execute in a few seconds in pure NumPy without DataFrame access.
    """
    if not policies:
        raise ValueError("policies list must be non-empty.")

    starts = (
        start_trade_indices
        if start_trade_indices is not None
        else dataset.rolling_start_trade_indices()
    )
    starts = np.asarray(starts, dtype=np.int32)
    if len(starts) > 1 and np.any(np.diff(starts) <= 0):
        raise ValueError("start_trade_indices must be strictly increasing (one row per rolling start).")
    if len(starts) and (starts[0] < 0 or starts[-1] >= len(dataset)):
        raise ValueError("start_trade_indices out of range for this dataset.")
    k_starts = len(starts)
    p_count = len(policies)

    if k_starts == 0:
        empty_out = np.empty(0, dtype=np.int8)
        empty_fail = np.empty(0, dtype=np.int8)
        empty_days = np.empty(0, dtype=np.float64)
        return [summarize_batch_arrays(empty_out, empty_fail, empty_days, rules.fee) for _ in policies]

    if not rules.phases:
        ones_out = np.full(k_starts, OUTCOME_PASS, dtype=np.int8)
        zeros_fail = np.full(k_starts, FAIL_NONE, dtype=np.int8)
        zeros_days = np.zeros(k_starts, dtype=np.float64)
        return [summarize_batch_arrays(ones_out, zeros_fail, zeros_days, rules.fee) for _ in policies]

    initial_balance = float(rules.account_size)
    reset_offset_ms = rules.daily_reset * HOUR_MS
    is_static_max = rules.max_loss_type == "static"
    trail_lock = (not is_static_max) and rules.trailing_lock_at_initial
    static_floor = initial_balance * (1.0 - rules.max_loss_pct / 100.0)
    trail_factor = 1.0 - rules.max_loss_pct / 100.0
    daily_loss_frac = rules.daily_loss_pct / 100.0
    has_consistency = rules.consistency_pct is not None
    consistency_frac = (rules.consistency_pct / 100.0) if has_consistency else 1.0
    firm_trade_cap = rules.max_risk_per_trade_pct

    n_phases = len(rules.phases)
    ph_target_bal = np.array(
        [initial_balance * (1.0 + ph.target_pct / 100.0) for ph in rules.phases],
        dtype=np.float64,
    )
    ph_target_gain = ph_target_bal - initial_balance
    ph_min_days = np.array([ph.min_days for ph in rules.phases], dtype=np.int32)
    ph_max_ms = np.array(
        [ph.max_days * DAY_MS if ph.max_days is not None else _INF_MS for ph in rules.phases],
        dtype=np.int64,
    )
    any_max_days = any(ph.max_days is not None for ph in rules.phases)

    # Policy arrays of shape (1, P)
    p_risk_pct = np.array([[pol.risk_pct for pol in policies]], dtype=np.float64)
    p_init_basis = np.array([[pol.risk_basis == "initial_balance" for pol in policies]], dtype=bool)
    p_has_near = np.array(
        [[pol.near_frac is not None and pol.near_mult != 1.0 for pol in policies]],
        dtype=bool,
    )
    p_near_frac = np.array(
        [[pol.near_frac if pol.near_frac is not None else 1.0 for pol in policies]],
        dtype=np.float64,
    )
    p_near_mult = np.array([[pol.near_mult for pol in policies]], dtype=np.float64)
    any_near = bool(np.any(p_has_near))

    p_has_behind = np.array(
        [[pol.behind_dd_pct is not None and pol.behind_mult != 1.0 for pol in policies]],
        dtype=bool,
    )
    p_behind_dd = np.array(
        [[pol.behind_dd_pct if pol.behind_dd_pct is not None else 1e9 for pol in policies]],
        dtype=np.float64,
    )
    p_behind_mult = np.array([[pol.behind_mult for pol in policies]], dtype=np.float64)
    any_behind = bool(np.any(p_has_behind))

    p_own_daily_frac = np.array(
        [[pol.own_daily_stop_pct / 100.0 for pol in policies]],
        dtype=np.float64,
    )
    p_has_sym_cap = np.array(
        [[pol.per_symbol_daily_cap_pct is not None for pol in policies]],
        dtype=bool,
    )
    p_sym_cap_frac = np.array(
        [[(pol.per_symbol_daily_cap_pct / 100.0) if pol.per_symbol_daily_cap_pct is not None else 1e9 for pol in policies]],
        dtype=np.float64,
    )
    any_sym_cap = bool(np.any(p_has_sym_cap))

    p_has_max_open = np.array([[pol.max_open_risk_pct is not None for pol in policies]], dtype=bool)
    p_max_open_frac = np.array(
        [[(pol.max_open_risk_pct / 100.0) if pol.max_open_risk_pct is not None else 1e9 for pol in policies]],
        dtype=np.float64,
    )
    any_max_open = bool(np.any(p_has_max_open))

    p_has_restart = np.array([[pol.restart_after_days is not None for pol in policies]], dtype=bool)
    p_restart_ms = np.array(
        [[pol.restart_after_days * DAY_MS if pol.restart_after_days is not None else _INF_MS for pol in policies]],
        dtype=np.int64,
    )
    any_restart = bool(np.any(p_has_restart))

    # State arrays of shape (K, P)
    alive = np.zeros((k_starts, p_count), dtype=bool)
    outcome = np.full((k_starts, p_count), OUTCOME_UNFINISHED, dtype=np.int8)
    fail_reason = np.full((k_starts, p_count), FAIL_NONE, dtype=np.int8)
    days_to_pass = np.full((k_starts, p_count), np.nan, dtype=np.float64)

    phase_idx = np.zeros((k_starts, p_count), dtype=np.int32)
    balance = np.full((k_starts, p_count), initial_balance, dtype=np.float64)
    hwm = np.full((k_starts, p_count), initial_balance, dtype=np.float64)
    day_start_ref = np.full((k_starts, p_count), initial_balance, dtype=np.float64)
    open_risk = np.zeros((k_starts, p_count), dtype=np.float64)
    trading_days = np.zeros((k_starts, p_count), dtype=np.int32)
    last_entry_day = np.full((k_starts, p_count), -10**9, dtype=np.int64)
    phase_start_ms = np.zeros((k_starts, p_count), dtype=np.int64)
    current_day_pnl = np.zeros((k_starts, p_count), dtype=np.float64) if has_consistency else None
    best_day_pnl = np.zeros((k_starts, p_count), dtype=np.float64) if has_consistency else None
    sym_risk_today = (
        np.zeros((k_starts, p_count, dataset.n_symbols), dtype=np.float64)
        if any_sym_cap
        else None
    )

    start_ms_vec = dataset.entry_ms[starts]
    start_ms_col = start_ms_vec[:, None]

    ev_ms_arr, ev_kind_arr, ev_tidx_arr = dataset.events
    ev_ms_list = ev_ms_arr.tolist()
    ev_kind_list = ev_kind_arr.tolist()
    ev_tidx_list = ev_tidx_arr.tolist()
    r_list = dataset.r_vals.tolist()
    sym_list = dataset.symbol_ids.tolist()
    starts_list = starts.tolist()

    open_trades: dict[int, tuple[int, int, np.ndarray]] = {}
    lo = 0
    hi = 0
    prev_day = _reset_day_index(int(start_ms_vec[0]), reset_offset_ms)
    prev_timeout_t = -10**18
    first_ev = int(dataset._entry_to_ev[starts_list[0]])

    for ev_pos in range(first_ev, len(ev_ms_list)):
        t_ms = ev_ms_list[ev_pos]
        kind = ev_kind_list[ev_pos]
        tidx = ev_tidx_list[ev_pos]

        # Timeout checks when timestamp advances and active window is non-empty
        if t_ms > prev_timeout_t and lo < hi:
            prev_timeout_t = t_ms
            if any_max_days:
                act_w = alive[lo:hi]
                timed = act_w & ((t_ms - phase_start_ms[lo:hi]) > ph_max_ms[phase_idx[lo:hi]])
                if np.any(timed):
                    outcome[lo:hi][timed] = OUTCOME_FAIL
                    fail_reason[lo:hi][timed] = FAIL_MAX_DAYS
                    act_w[timed] = False
            if any_restart:
                act_w = alive[lo:hi]
                restarted = act_w & p_has_restart & ((t_ms - start_ms_col[lo:hi]) > p_restart_ms)
                if np.any(restarted):
                    outcome[lo:hi][restarted] = OUTCOME_RESTART
                    fail_reason[lo:hi][restarted] = FAIL_RESTART
                    act_w[restarted] = False

        # Activate new rolling start(s) whose first trade is `tidx`
        if kind == EV_ENTRY and hi < k_starts and tidx == starts_list[hi]:
            alive[hi, :] = True
            phase_start_ms[hi, :] = t_ms
            day_start_ref[hi, :] = initial_balance
            hi += 1

        if lo >= hi:
            continue

        # Advance `lo` over completely finished leading start rows
        while lo < hi and not np.any(alive[lo]):
            lo += 1
        if lo >= hi:
            continue

        ev_day = (t_ms - reset_offset_ms) // DAY_MS
        if ev_day != prev_day:
            act_w = alive[lo:hi]
            if has_consistency and best_day_pnl is not None and current_day_pnl is not None:
                best_day_pnl[lo:hi] = np.maximum(best_day_pnl[lo:hi], current_day_pnl[lo:hi])
                current_day_pnl[lo:hi][act_w] = 0.0
            day_start_ref[lo:hi] = np.where(act_w, balance[lo:hi], day_start_ref[lo:hi])
            if sym_risk_today is not None:
                sym_risk_today[lo:hi] = 0.0
            prev_day = ev_day

        if kind == EV_ENTRY:
            act_w = alive[lo:hi]
            bal_w = balance[lo:hi]
            pidx_w = phase_idx[lo:hi]

            eff_pct = np.broadcast_to(p_risk_pct, bal_w.shape)
            if any_near:
                tgain = ph_target_gain[pidx_w]
                prog = (bal_w - initial_balance) / tgain
                near_mask = p_has_near & (prog >= p_near_frac - 1e-12)
                if np.any(near_mask):
                    eff_pct = np.where(near_mask, eff_pct * p_near_mult, eff_pct)
            if any_behind:
                hwm_w = hwm[lo:hi]
                dd_pct = np.where(hwm_w > 0.0, (hwm_w - bal_w) / hwm_w * 100.0, 0.0)
                behind_mask = p_has_behind & (dd_pct >= p_behind_dd - 1e-12)
                if np.any(behind_mask):
                    eff_pct = np.where(behind_mask, eff_pct * p_behind_mult, eff_pct)
            if firm_trade_cap is not None:
                eff_pct = np.minimum(eff_pct, firm_trade_cap)

            ref_bal = np.where(p_init_basis, initial_balance, bal_w)
            new_risk = np.maximum(0.0, ref_bal * (eff_pct / 100.0))

            dref_w = day_start_ref[lo:hi]
            orisk_w = open_risk[lo:hi]
            realised_loss = np.maximum(0.0, dref_w - bal_w)

            ok = act_w & (new_risk > 0.0) & (
                (realised_loss + orisk_w + new_risk) <= dref_w * p_own_daily_frac + _EPS
            )
            sym_id = sym_list[tidx]
            if sym_risk_today is not None:
                sym_used = sym_risk_today[lo:hi, :, sym_id]
                ok &= ~p_has_sym_cap | ((sym_used + new_risk) <= dref_w * p_sym_cap_frac + _EPS)
            if any_max_open:
                ok &= ~p_has_max_open | ((orisk_w + new_risk) <= ref_bal * p_max_open_frac + _EPS)

            if np.any(ok):
                res = np.where(ok, new_risk, 0.0)
                open_trades[tidx] = (lo, hi, res)
                orisk_w += res
                if sym_risk_today is not None:
                    sym_risk_today[lo:hi, :, sym_id] += res
                led_w = last_entry_day[lo:hi]
                new_day = ok & (led_w != ev_day)
                if np.any(new_day):
                    trading_days[lo:hi] += new_day
                    led_w[new_day] = ev_day
        else:
            rec = open_trades.pop(tidx, None)
            if rec is None:
                continue
            lo_e, hi_e, res_e = rec
            r0 = max(lo, lo_e)
            r1 = min(hi, hi_e)
            if r0 >= r1:
                continue

            res_sub = res_e[r0 - lo_e : r1 - lo_e]
            act_sub = alive[r0:r1]
            active_open = act_sub & (res_sub > 0.0)
            if not np.any(active_open):
                continue

            r_val = r_list[tidx]
            res_active = np.where(active_open, res_sub, 0.0)
            pnl = res_active * r_val
            open_risk[r0:r1] -= res_active

            bal_sub = balance[r0:r1]
            bal_sub += pnl
            if has_consistency and current_day_pnl is not None:
                current_day_pnl[r0:r1] += pnl

            dref_sub = day_start_ref[r0:r1]
            daily_breach = active_open & ((dref_sub - bal_sub) > dref_sub * daily_loss_frac + _EPS)
            if np.any(daily_breach):
                outcome[r0:r1][daily_breach] = OUTCOME_FAIL
                fail_reason[r0:r1][daily_breach] = FAIL_DAILY_LOSS
                act_sub[daily_breach] = False

            still = active_open & ~daily_breach
            if not np.any(still):
                continue

            hwm_sub = hwm[r0:r1]
            grow = still & (bal_sub > hwm_sub)
            if np.any(grow):
                hwm_sub[grow] = bal_sub[grow]

            if is_static_max:
                max_breach = still & (bal_sub < static_floor - _EPS)
            elif trail_lock:
                max_breach = still & (bal_sub < np.minimum(hwm_sub * trail_factor, initial_balance) - _EPS)
            else:
                max_breach = still & (bal_sub < hwm_sub * trail_factor - _EPS)

            if np.any(max_breach):
                outcome[r0:r1][max_breach] = OUTCOME_FAIL
                fail_reason[r0:r1][max_breach] = FAIL_MAX_LOSS
                act_sub[max_breach] = False

            still &= ~max_breach
            if not np.any(still):
                continue

            pidx_sub = phase_idx[r0:r1]
            passed_phase = (
                still
                & (bal_sub >= ph_target_bal[pidx_sub] - _EPS)
                & (trading_days[r0:r1] >= ph_min_days[pidx_sub])
            )
            if has_consistency and best_day_pnl is not None and current_day_pnl is not None:
                tot_prof = bal_sub - initial_balance
                best_day = np.maximum(best_day_pnl[r0:r1], current_day_pnl[r0:r1])
                passed_phase &= (tot_prof > 0.0) & (best_day <= tot_prof * consistency_frac + _EPS)

            if np.any(passed_phase):
                final_pass = passed_phase & (pidx_sub + 1 >= n_phases)
                if np.any(final_pass):
                    outcome[r0:r1][final_pass] = OUTCOME_PASS
                    act_sub[final_pass] = False
                    elapsed_days = (t_ms - start_ms_col[r0:r1]) / DAY_MS
                    days_to_pass[r0:r1][final_pass] = np.broadcast_to(elapsed_days, bal_sub.shape)[
                        final_pass
                    ]

                adv = passed_phase & ~final_pass
                if np.any(adv):
                    pidx_sub[adv] += 1
                    bal_sub[adv] = initial_balance
                    hwm_sub[adv] = initial_balance
                    dref_sub[adv] = initial_balance
                    open_risk[r0:r1][adv] = 0.0
                    trading_days[r0:r1][adv] = 0
                    last_entry_day[r0:r1][adv] = -10**9
                    phase_start_ms[r0:r1][adv] = t_ms
                    if has_consistency and current_day_pnl is not None and best_day_pnl is not None:
                        current_day_pnl[r0:r1][adv] = 0.0
                        best_day_pnl[r0:r1][adv] = 0.0
                    if sym_risk_today is not None:
                        sym_risk_today[r0:r1][adv] = 0.0

                    # Zero out any lingering Phase-1 open trades for cells advancing to next phase
                    if open_trades:
                        for o_lo, o_hi, o_res in open_trades.values():
                            c0 = max(r0, o_lo)
                            c1 = min(r1, o_hi)
                            if c0 < c1:
                                adv_slice = adv[c0 - r0 : c1 - r0]
                                if np.any(adv_slice):
                                    o_res[c0 - o_lo : c1 - o_lo][adv_slice] = 0.0

    # End-of-dataset timeout checks for remaining active challenges
    data_end_ms = dataset.end_ms
    if lo < hi:
        if any_max_days:
            act_w = alive[lo:hi]
            timed = act_w & ((data_end_ms - phase_start_ms[lo:hi]) > ph_max_ms[phase_idx[lo:hi]])
            if np.any(timed):
                outcome[lo:hi][timed] = OUTCOME_FAIL
                fail_reason[lo:hi][timed] = FAIL_MAX_DAYS
                act_w[timed] = False
        if any_restart:
            act_w = alive[lo:hi]
            restarted = act_w & p_has_restart & ((data_end_ms - start_ms_col[lo:hi]) > p_restart_ms)
            if np.any(restarted):
                outcome[lo:hi][restarted] = OUTCOME_RESTART
                fail_reason[lo:hi][restarted] = FAIL_RESTART
                act_w[restarted] = False

    return [
        summarize_batch_arrays(outcome[:, j], fail_reason[:, j], days_to_pass[:, j], rules.fee)
        for j in range(p_count)
    ]


def simulate_rolling_starts(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy,
) -> BatchChallengeSummary:
    """Evaluate a single policy across every calendar-day rolling start in `dataset`."""
    return simulate_challenge_batch(dataset, rules, [policy])[0]


def sample_challenge_fan_paths(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy,
    max_paths: int = 80,
) -> list[ChallengeRunResult]:
    """Simulate evenly spaced rolling starts with full equity curves for fan-chart plotting."""
    starts = dataset.rolling_start_trade_indices()
    if len(starts) == 0:
        return []
    if len(starts) > max_paths:
        indices = np.linspace(0, len(starts) - 1, max_paths, dtype=int)
        starts = starts[indices]
    return [
        simulate_challenge_single(dataset, rules, policy, start_trade_idx=int(s), record_curve=True)
        for s in starts
    ]


def analyze_failures(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy,
    max_samples: int = 8,
    max_detail_runs: int = 150,
    batch_summary: BatchChallengeSummary | None = None,
) -> FailureAttribution:
    """Identify which trading days, symbols, and rule clauses caused failures."""
    starts = dataset.rolling_start_trade_indices()
    batch = (
        batch_summary
        if batch_summary is not None
        else simulate_challenge_batch(dataset, rules, [policy], starts)[0]
    )
    failed_positions = np.flatnonzero(batch.outcomes == OUTCOME_FAIL)

    day_counts: Counter[str] = Counter()
    day_symbols: dict[str, set[str]] = defaultdict(set)
    day_r_sums: dict[str, list[float]] = defaultdict(list)
    sym_counts: Counter[str] = Counter()
    samples: list[dict[str, object]] = []

    if len(failed_positions) > max_detail_runs:
        sample_positions = failed_positions[
            np.linspace(0, len(failed_positions) - 1, max_detail_runs, dtype=int)
        ]
    else:
        sample_positions = failed_positions

    for pos in sample_positions:
        s = int(starts[int(pos)])
        res = simulate_challenge_single(
            dataset, rules, policy, start_trade_idx=s, record_breach_detail=True
        )
        if res.outcome == "fail" and res.fail_reason is not None:
            if res.breach_day_utc:
                day_counts[res.breach_day_utc] += 1
                day_symbols[res.breach_day_utc].update(res.breach_day_symbols)
                if res.breach_day_r_sum is not None:
                    day_r_sums[res.breach_day_utc].append(res.breach_day_r_sum)
            if res.breach_symbol:
                sym_counts[res.breach_symbol] += 1
            if len(samples) < max_samples:
                samples.append(
                    {
                        "fail_reason": res.fail_reason,
                        "fail_phase": (res.fail_phase + 1) if res.fail_phase is not None else None,
                        "breach_day_utc": res.breach_day_utc,
                        "breach_symbol": res.breach_symbol,
                        "same_day_symbols": list(res.breach_day_symbols),
                        "same_day_r_sum": res.breach_day_r_sum,
                        "final_balance": round(res.final_balance, 2),
                    }
                )

    top_days = [
        {
            "day_utc": day,
            "breach_count": cnt,
            "symbols_involved": sorted(day_symbols[day]),
            "mean_day_r_sum": round(float(np.mean(day_r_sums[day])), 2) if day_r_sums[day] else None,
        }
        for day, cnt in day_counts.most_common(6)
    ]
    top_syms = [{"symbol": sym, "breach_count": cnt} for sym, cnt in sym_counts.most_common(6)]

    return FailureAttribution(
        n_total_starts=len(starts),
        n_breaches=batch.n_failed,
        fail_reason_counts=dict(batch.fail_reason_counts),
        top_breach_days=top_days,
        top_breach_symbols=top_syms,
        sample_breaches=samples,
    )
