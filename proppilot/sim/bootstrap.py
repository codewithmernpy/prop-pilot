"""Weekly block-bootstrap Monte Carlo for challenge evaluation paths."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from proppilot.ingest import DAY_MS, HOUR_MS, TradeDataset
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
    summarize_batch_arrays,
)

WEEK_MS = 7 * DAY_MS
_EPS = 1e-9
_INF_MS = 10**16


@dataclass(frozen=True)
class WeekBlock:
    """Relative trade sequence within a 7-day block, preserving intra-week overlap."""

    entry_rel_ms: tuple[int, ...]
    exit_rel_ms: tuple[int, ...]
    symbol_ids: tuple[int, ...]
    r_vals: tuple[float, ...]


@dataclass(frozen=True)
class WeekPool:
    """Collection of 7-day blocks extracted from a TradeDataset for block bootstrap."""

    blocks: tuple[WeekBlock, ...]
    n_symbols: int

    def __len__(self) -> int:
        return len(self.blocks)


def build_week_blocks(dataset: TradeDataset) -> WeekPool:
    """Partition trades into whole 7-day calendar week blocks (Monday 00:00 UTC aligned).

    Every calendar week between the first and last trade becomes one block, including weeks
    with no trades (idle weeks are real outcomes; dropping them would inflate trade density).
    Partial boundary weeks are trimmed when the history spans at least 4 weeks so weekly
    trade density is not biased downward by edge fragments.
    """
    entry_ms = dataset.entry_ms
    exit_ms = dataset.exit_ms
    sym_ids = dataset.symbol_ids
    r_vals = dataset.r_vals

    # 1970-01-05 00:00:00 UTC was a Monday (offset = 4 days from Thursday 1970-01-01).
    monday_offset_ms = 4 * DAY_MS
    week_idx = (entry_ms - monday_offset_ms) // WEEK_MS
    first_w = int(week_idx[0])
    last_w = int(week_idx[-1])

    # Trim leading/trailing partial weeks if enough full weeks exist.
    span_weeks = last_w - first_w + 1
    if span_weeks >= 4:
        first_w += 1
        last_w -= 1

    blocks: list[WeekBlock] = []
    for w in range(first_w, last_w + 1):
        w_start = monday_offset_ms + w * WEEK_MS
        mask = week_idx == w
        e_rel = tuple(int(x - w_start) for x in entry_ms[mask].tolist())
        x_rel = tuple(int(x - w_start) for x in exit_ms[mask].tolist())
        s_ids = tuple(int(x) for x in sym_ids[mask].tolist())
        rv = tuple(float(x) for x in r_vals[mask].tolist())
        blocks.append(WeekBlock(entry_rel_ms=e_rel, exit_rel_ms=x_rel, symbol_ids=s_ids, r_vals=rv))

    if not blocks:
        # Ultra-short dataset fallback: single block relative to first day midnight.
        anchor = (int(entry_ms[0]) // DAY_MS) * DAY_MS
        blocks.append(
            WeekBlock(
                entry_rel_ms=tuple(int(x - anchor) for x in entry_ms.tolist()),
                exit_rel_ms=tuple(int(x - anchor) for x in exit_ms.tolist()),
                symbol_ids=tuple(int(x) for x in sym_ids.tolist()),
                r_vals=tuple(float(x) for x in r_vals.tolist()),
            )
        )

    return WeekPool(blocks=tuple(blocks), n_symbols=dataset.n_symbols)


def simulate_bootstrap_challenge(
    dataset: TradeDataset,
    rules: PropRules,
    policy: SizingPolicy,
    n_paths: int = 2000,
    seed: int = 42,
    horizon_days: int = 365,
    pool: WeekPool | None = None,
) -> BatchChallengeSummary:
    """Run N 1-year block-bootstrap synthetic challenge paths."""
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0.")

    week_pool = pool if pool is not None else build_week_blocks(dataset)
    rng = np.random.default_rng(seed)

    if not rules.phases:
        outcomes = np.full(n_paths, OUTCOME_PASS, dtype=np.int8)
        fail_reasons = np.full(n_paths, FAIL_NONE, dtype=np.int8)
        days_arr = np.zeros(n_paths, dtype=np.float64)
        return summarize_batch_arrays(outcomes, fail_reasons, days_arr, rules.fee)

    n_weeks_needed = max(1, (horizon_days + 6) // 7)
    horizon_ms = horizon_days * DAY_MS
    week_matrix = rng.integers(0, len(week_pool), size=(n_paths, n_weeks_needed))

    initial_balance = float(rules.account_size)
    reset_offset_ms = rules.daily_reset * HOUR_MS
    is_static_max = rules.max_loss_type == "static"
    trail_lock = (not is_static_max) and rules.trailing_lock_at_initial
    daily_loss_frac = rules.daily_loss_pct / 100.0
    max_loss_frac = rules.max_loss_pct / 100.0
    static_floor = initial_balance * (1.0 - max_loss_frac)
    trail_factor = 1.0 - max_loss_frac
    has_consistency = rules.consistency_pct is not None
    consistency_frac = (rules.consistency_pct / 100.0) if has_consistency else 1.0
    firm_trade_cap = rules.max_risk_per_trade_pct

    use_init_basis = policy.risk_basis == "initial_balance"
    base_risk_pct = policy.risk_pct
    near_frac = policy.near_frac
    near_mult = policy.near_mult
    has_near = near_frac is not None and near_mult != 1.0
    behind_dd = policy.behind_dd_pct
    behind_mult = policy.behind_mult
    has_behind = behind_dd is not None and behind_mult != 1.0
    own_daily_frac = policy.own_daily_stop_pct / 100.0
    sym_cap_frac = (
        policy.per_symbol_daily_cap_pct / 100.0
        if policy.per_symbol_daily_cap_pct is not None
        else None
    )
    max_open_frac = (
        policy.max_open_risk_pct / 100.0 if policy.max_open_risk_pct is not None else None
    )
    restart_ms = policy.restart_after_days * DAY_MS if policy.restart_after_days is not None else _INF_MS

    phases = rules.phases
    n_phases = len(phases)
    ph_targets = [initial_balance * (1.0 + ph.target_pct / 100.0) for ph in phases]
    ph_gains = [tb - initial_balance for tb in ph_targets]
    ph_min_days = [ph.min_days for ph in phases]
    ph_max_ms = [ph.max_days * DAY_MS if ph.max_days is not None else _INF_MS for ph in phases]

    outcomes = np.full(n_paths, OUTCOME_UNFINISHED, dtype=np.int8)
    fail_reasons = np.full(n_paths, FAIL_NONE, dtype=np.int8)
    days_arr = np.full(n_paths, np.nan, dtype=np.float64)

    blocks = week_pool.blocks

    for path_idx in range(n_paths):
        chosen_weeks = week_matrix[path_idx]

        # Lazily determine start timestamp from first trade of week 0
        start_ms = -1
        for w_pos in range(n_weeks_needed):
            blk0 = blocks[int(chosen_weeks[w_pos])]
            if blk0.entry_rel_ms:
                start_ms = w_pos * WEEK_MS + blk0.entry_rel_ms[0]
                break

        if start_ms < 0 or start_ms >= horizon_ms:
            continue

        phase_idx = 0
        phase_start_ms = start_ms
        phase_max_ms = ph_max_ms[0]
        target_bal = ph_targets[0]
        target_gain = ph_gains[0]
        min_days_req = ph_min_days[0]

        balance = initial_balance
        hwm = initial_balance
        day_start_ref = initial_balance
        open_risk = 0.0
        trading_days = 0
        last_entry_day = -10**9
        cur_day = (start_ms - reset_offset_ms) // DAY_MS
        current_day_pnl = 0.0
        best_day_pnl = 0.0  # best COMPLETED day's final PnL in the current phase
        sym_risk_today: dict[int, float] = {}
        # Heap items: (exit_ms, tie_prio [0 if exit > entry else 1], seq_id, sym_id, r_val, reserved_risk, phase_epoch)
        open_heap: list[tuple[int, int, int, int, float, float, int]] = []
        phase_epoch = 0
        seq_id = 0
        done = False

        def _process_exit(
            ex_ms: int,
            ex_r: float,
            ex_res: float,
            ex_epoch: int,
        ) -> bool:
            nonlocal phase_idx, phase_start_ms, phase_max_ms, target_bal, target_gain, min_days_req
            nonlocal balance, hwm, day_start_ref, open_risk, trading_days, last_entry_day
            nonlocal cur_day, current_day_pnl, best_day_pnl, phase_epoch, done

            if ex_epoch != phase_epoch:
                return False

            if (ex_ms - phase_start_ms) > phase_max_ms:
                outcomes[path_idx] = OUTCOME_FAIL
                fail_reasons[path_idx] = FAIL_MAX_DAYS
                done = True
                return True
            if (ex_ms - start_ms) > restart_ms:
                outcomes[path_idx] = OUTCOME_RESTART
                fail_reasons[path_idx] = FAIL_RESTART
                done = True
                return True

            ev_day = (ex_ms - reset_offset_ms) // DAY_MS
            if ev_day != cur_day:
                if has_consistency:
                    best_day_pnl = max(best_day_pnl, current_day_pnl)
                cur_day = ev_day
                day_start_ref = balance
                current_day_pnl = 0.0
                if sym_cap_frac is not None:
                    sym_risk_today.clear()

            pnl = ex_r * ex_res
            open_risk -= ex_res
            balance += pnl
            if has_consistency:
                current_day_pnl += pnl

            if (day_start_ref - balance) > day_start_ref * daily_loss_frac + _EPS:
                outcomes[path_idx] = OUTCOME_FAIL
                fail_reasons[path_idx] = FAIL_DAILY_LOSS
                done = True
                return True

            if balance > hwm:
                hwm = balance
            floor = static_floor if is_static_max else hwm * trail_factor
            if trail_lock and floor > initial_balance:
                floor = initial_balance
            if balance < floor - _EPS:
                outcomes[path_idx] = OUTCOME_FAIL
                fail_reasons[path_idx] = FAIL_MAX_LOSS
                done = True
                return True

            consistency_ok = True
            if has_consistency:
                tot_prof = balance - initial_balance
                best_day = best_day_pnl if best_day_pnl > current_day_pnl else current_day_pnl
                if tot_prof <= 0.0 or best_day > tot_prof * consistency_frac + _EPS:
                    consistency_ok = False

            if balance >= target_bal - _EPS and trading_days >= min_days_req and consistency_ok:
                if phase_idx + 1 < n_phases:
                    phase_idx += 1
                    phase_epoch += 1
                    phase_start_ms = ex_ms
                    phase_max_ms = ph_max_ms[phase_idx]
                    target_bal = ph_targets[phase_idx]
                    target_gain = ph_gains[phase_idx]
                    min_days_req = ph_min_days[phase_idx]
                    balance = initial_balance
                    hwm = initial_balance
                    day_start_ref = initial_balance
                    open_risk = 0.0
                    trading_days = 0
                    last_entry_day = -10**9
                    current_day_pnl = 0.0
                    best_day_pnl = 0.0
                    if sym_cap_frac is not None:
                        sym_risk_today.clear()
                else:
                    outcomes[path_idx] = OUTCOME_PASS
                    days_arr[path_idx] = (ex_ms - start_ms) / DAY_MS
                    done = True
                    return True
            return False

        horizon_reached = False
        for w_pos in range(n_weeks_needed):
            if done or horizon_reached:
                break
            w_offset = w_pos * WEEK_MS
            blk = blocks[int(chosen_weeks[w_pos])]
            e_list = blk.entry_rel_ms
            x_list = blk.exit_rel_ms
            s_list = blk.symbol_ids
            r_list = blk.r_vals

            for j in range(len(e_list)):
                t_entry = w_offset + e_list[j]
                if t_entry >= horizon_ms:
                    horizon_reached = True
                    break
                t_exit = w_offset + x_list[j]

                # Drain exits strictly before t_entry, or normal exits exactly at t_entry
                while open_heap and (
                    open_heap[0][0] < t_entry
                    or (open_heap[0][0] == t_entry and open_heap[0][1] == 0)
                ):
                    ex_ms, _, _, _, ex_r, ex_res, ex_ep = heapq.heappop(open_heap)
                    if _process_exit(ex_ms, ex_r, ex_res, ex_ep):
                        break
                if done:
                    break

                if (t_entry - phase_start_ms) > phase_max_ms:
                    outcomes[path_idx] = OUTCOME_FAIL
                    fail_reasons[path_idx] = FAIL_MAX_DAYS
                    done = True
                    break
                if (t_entry - start_ms) > restart_ms:
                    outcomes[path_idx] = OUTCOME_RESTART
                    fail_reasons[path_idx] = FAIL_RESTART
                    done = True
                    break

                ev_day = (t_entry - reset_offset_ms) // DAY_MS
                if ev_day != cur_day:
                    if has_consistency:
                        best_day_pnl = max(best_day_pnl, current_day_pnl)
                    cur_day = ev_day
                    day_start_ref = balance
                    current_day_pnl = 0.0
                    if sym_cap_frac is not None:
                        sym_risk_today.clear()

                eff_pct = base_risk_pct
                if has_near and target_gain > 0.0:
                    if (balance - initial_balance) / target_gain >= near_frac - 1e-12:
                        eff_pct *= near_mult
                if has_behind and hwm > 0.0:
                    if (hwm - balance) / hwm * 100.0 >= behind_dd - 1e-12:
                        eff_pct *= behind_mult
                if firm_trade_cap is not None and eff_pct > firm_trade_cap:
                    eff_pct = firm_trade_cap

                ref_bal = initial_balance if use_init_basis else balance
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
                    if sym_risk_today.get(sym_id, 0.0) + new_risk > day_start_ref * sym_cap_frac + _EPS:
                        continue
                if max_open_frac is not None:
                    if open_risk + new_risk > ref_bal * max_open_frac + _EPS:
                        continue

                open_risk += new_risk
                if sym_cap_frac is not None:
                    sym_risk_today[sym_id] = sym_risk_today.get(sym_id, 0.0) + new_risk
                if ev_day != last_entry_day:
                    trading_days += 1
                    last_entry_day = ev_day

                seq_id += 1
                prio = 0 if t_exit > t_entry else 1
                heapq.heappush(open_heap, (t_exit, prio, seq_id, sym_id, r_list[j], new_risk, phase_epoch))

        # Drain remaining open trades whose exits occur within horizon
        while not done and open_heap:
            ex_ms, _, _, _, ex_r, ex_res, ex_ep = heapq.heappop(open_heap)
            if ex_ms > horizon_ms:
                break
            _process_exit(ex_ms, ex_r, ex_res, ex_ep)

        if not done:
            if (horizon_ms - phase_start_ms) > phase_max_ms:
                outcomes[path_idx] = OUTCOME_FAIL
                fail_reasons[path_idx] = FAIL_MAX_DAYS
            elif (horizon_ms - start_ms) > restart_ms:
                outcomes[path_idx] = OUTCOME_RESTART
                fail_reasons[path_idx] = FAIL_RESTART

    return summarize_batch_arrays(outcomes, fail_reasons, days_arr, rules.fee)
