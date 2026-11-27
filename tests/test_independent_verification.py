"""Independent verification tests with hand-calculated expected values.

Every expected number below is derived by hand in the comments (account $100,000 unless
stated). Random-equivalence tests check that the vectorised batch simulator, the weekly
block bootstrap, and the funded bootstrap reproduce the auditable scalar state machines.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # the anthropic 2.x SDK ships httpx2 (plain httpx is not installed with it)
import numpy as np
import pandas as pd
import pydantic
import pytest

from proppilot.ai import (
    EXTRACT_PROMPT,
    PlanSession,
    RuleExtractionError,
    build_copilot_tools,
    extract_rules_from_text,
    extract_rules_offline,
    run_copilot_query,
)
from proppilot.ai.copilot import COPILOT_PROMPT
from proppilot.ai.session import compact_json
from proppilot.ingest import trades_from_dataframe
from proppilot.optimize import expected_net_12m_usd, optimise_on_train, run_honest_optimisation
from proppilot.policy import SizingPolicy
from proppilot.rules import PhaseRule, PropRules, load_preset
from proppilot.sim import (
    build_week_blocks,
    simulate_bootstrap_challenge,
    simulate_challenge_batch,
    simulate_challenge_single,
    simulate_funded_bootstrap,
    simulate_funded_single,
    simulate_rolling_starts,
)
from proppilot.sim.models import FAIL_CODE_TO_NAME
from proppilot.synthetic import SyntheticConfig, generate_synthetic_trades

ROOT = Path(__file__).resolve().parent.parent
BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)  # a Monday (week-aligned for bootstrap tests)
OUTCOME_NAME = {0: "unfinished", 1: "pass", 2: "fail", 3: "restart"}
REASON_CODE = {name: code for code, name in FAIL_CODE_TO_NAME.items()}


def _ts(day: int, hour: int = 0, minute: int = 0) -> str:
    return (BASE + timedelta(days=day, hours=hour, minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ds(rows: list[tuple[str, str, str, float]]):
    return trades_from_dataframe(pd.DataFrame(rows, columns=["entry_time", "exit_time", "symbol", "R"]))


def _rules(**kw: object) -> PropRules:
    params: dict[str, object] = dict(
        name="T",
        phases=[PhaseRule(name="P1", target_pct=50.0, min_days=0)],
        daily_loss_pct=50.0,
        daily_reset=0,
        max_loss_pct=50.0,
        max_loss_type="static",
        fee=500.0,
        profit_split=80.0,
        payout_frequency_days=14,
        account_size=100_000.0,
    )
    params.update(kw)
    return PropRules(**params)


def _pol(**kw: object) -> SizingPolicy:
    params: dict[str, object] = dict(
        risk_pct=1.0,
        risk_basis="initial_balance",
        near_frac=None,
        near_mult=1.0,
        own_daily_stop_pct=100.0,
        max_open_risk_pct=None,
        restart_after_days=None,
        funded_risk_pct=1.0,
        keep_buffer_pct=4.0,
        cppi_cap=1.5,
        cppi_c0=None,
    )
    params.update(kw)
    return SizingPolicy(**params)


# ---------------------------------------------------------------------------------------------
# Challenge mechanics (hand calculated)
# ---------------------------------------------------------------------------------------------


def test_risk_reservation_uses_entry_time_balance_and_pnl_at_exit() -> None:
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 12), "BTC", 2.0),
            (_ts(0, 9), _ts(0, 14), "ETH", -1.0),
            (_ts(0, 13), _ts(0, 15), "SOL", 1.0),
        ]
    )
    # current_balance basis:
    #  A 08:00 reserves 1% of 100,000 = 1,000; exits 12:00 +2R -> 102,000
    #  B 09:00 reserves 1% of 100,000 (A still open) = 1,000; exits 14:00 -1R -> 101,000
    #  C 13:00 reserves 1% of 102,000 = 1,020; exits 15:00 +1R -> 102,020
    res = simulate_challenge_single(ds, _rules(), _pol(risk_basis="current_balance"))
    assert res.final_balance == pytest.approx(102_020.0)
    # initial basis: +2,000 - 1,000 + 1,000 = 102,000
    res_i = simulate_challenge_single(ds, _rules(), _pol())
    assert res_i.final_balance == pytest.approx(102_000.0)
    assert res_i.trades_taken == 3


def test_daily_loss_measured_from_day_start_balance_with_reset_hour() -> None:
    # Reset at 22:00 UTC, firm daily limit 5%, risk 2% of initial ($2,000 per R).
    # T1 d0 10:00-11:00 +5R -> 110,000. New reset-day starts d0 22:00: day_start_ref = 110,000.
    # T2 d0 23:00 -> d1 01:00 -1.5R = -3,000 -> 107,000; T3 d1 02:00-03:00.
    rules = _rules(daily_loss_pct=5.0, daily_reset=22)
    pol = _pol(risk_pct=2.0)
    rows = [(_ts(0, 10), _ts(0, 11), "BTC", 5.0), (_ts(0, 23), _ts(1, 1), "ETH", -1.5)]
    # T3 -1.2R = -2,400: day loss 5,400 <= 5% x 110,000 = 5,500 -> survives (a limit measured
    # from the initial 100,000 would have breached at 5,000).
    ok = simulate_challenge_single(_ds(rows + [(_ts(1, 2), _ts(1, 3), "SOL", -1.2)]), rules, pol)
    assert ok.outcome == "unfinished"
    assert ok.final_balance == pytest.approx(104_600.0)
    # T3 -1.3R = -2,600: day loss 5,600 > 5,500 -> daily_loss breach.
    bad = simulate_challenge_single(_ds(rows + [(_ts(1, 2), _ts(1, 3), "SOL", -1.3)]), rules, pol)
    assert (bad.outcome, bad.fail_reason) == ("fail", "daily_loss")
    assert bad.final_balance == pytest.approx(104_400.0)
    # With a 00:00 reset, T2's -3,000 is booked on d1 (exit 01:00): d1 day_start = 110,000 too,
    # so the same trades breach on d1 as well -> both resets agree here (sanity).
    bad0 = simulate_challenge_single(
        _ds(rows + [(_ts(1, 2), _ts(1, 3), "SOL", -1.3)]), _rules(daily_loss_pct=5.0), pol
    )
    assert bad0.fail_reason == "daily_loss"


def test_own_daily_stop_counts_realised_loss_plus_open_risk_plus_new_risk() -> None:
    # Own stop 2.5% = $2,500, risk $1,000/trade.
    # T1 08-09 -1R: realised 1,000. T2 10:00 (open to 15:00): 1,000 + 0 + 1,000 = 2,000 <= 2,500 -> taken.
    # T3 11:00: 1,000 + 1,000 open + 1,000 = 3,000 > 2,500 -> BLOCKED (would lose -1R).
    # T2 exits +0.5R (+500). d1: new day, T4 +1R (+1,000).
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 9), "BTC", -1.0),
            (_ts(0, 10), _ts(0, 15), "ETH", 0.5),
            (_ts(0, 11), _ts(0, 12), "SOL", -1.0),
            (_ts(1, 9), _ts(1, 10), "BTC", 1.0),
        ]
    )
    res = simulate_challenge_single(ds, _rules(), _pol(own_daily_stop_pct=2.5))
    assert res.trades_taken == 3
    assert res.final_balance == pytest.approx(100_000 - 1_000 + 500 + 1_000)


def test_static_trailing_and_trailing_lock_floors_in_challenge() -> None:
    # +8R -> 108,000 (HWM), -3R -> 105,000, -3.6R -> 101,400 (max loss 6%).
    ds = _ds(
        [
            (_ts(0, 9), _ts(0, 10), "BTC", 8.0),
            (_ts(1, 9), _ts(1, 10), "ETH", -3.0),
            (_ts(2, 9), _ts(2, 10), "SOL", -3.6),
        ]
    )
    pol = _pol()
    static = simulate_challenge_single(ds, _rules(max_loss_pct=6.0), pol)
    assert static.outcome == "unfinished"  # floor 94,000
    trail = simulate_challenge_single(ds, _rules(max_loss_pct=6.0, max_loss_type="trailing_closed"), pol)
    assert (trail.outcome, trail.fail_reason) == ("fail", "max_loss")  # floor 108,000*0.94 = 101,520
    lock = simulate_challenge_single(
        ds, _rules(max_loss_pct=6.0, max_loss_type="trailing_closed", trailing_lock_at_initial=True), pol
    )
    assert lock.outcome == "unfinished"  # floor min(100,000, 101,520) = 100,000
    # The batch simulator applies the same floors.
    for rules, expected in (
        (_rules(max_loss_pct=6.0), "unfinished"),
        (_rules(max_loss_pct=6.0, max_loss_type="trailing_closed"), "fail"),
        (_rules(max_loss_pct=6.0, max_loss_type="trailing_closed", trailing_lock_at_initial=True), "unfinished"),
    ):
        batch = simulate_challenge_batch(ds, rules, [pol], np.array([0], dtype=np.int32))[0]
        assert OUTCOME_NAME[int(batch.outcomes[0])] == expected


def test_min_trading_days_count_only_admitted_entries() -> None:
    # Target +5% (105,000). max_open_risk 1% => a second concurrent trade is blocked.
    # d0 T1 08:00-20:00 +3R -> 103,000 (day 1); d1 T2 +1R -> 104,000 (day 2);
    # d2 T3 09:00 -> d3 06:00 +0.5R (day 3); d3 T4 05:00 BLOCKED (T3 open) -> d3 not a trading day;
    # T3 exit -> 104,500; d4 T5 09:00-10:00 +0.5R -> 105,000 = target, trading days = 4.
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 20), "BTC", 3.0),
            (_ts(1, 9), _ts(1, 10), "ETH", 1.0),
            (_ts(2, 9), _ts(3, 6), "SOL", 0.5),
            (_ts(3, 5), _ts(3, 7), "BTC", 2.0),
            (_ts(4, 9), _ts(4, 10), "ETH", 0.5),
        ]
    )
    pol = _pol(max_open_risk_pct=1.0)
    four = simulate_challenge_single(ds, _rules(phases=[PhaseRule(target_pct=5.0, min_days=4)]), pol)
    assert four.outcome == "pass"
    assert four.trades_taken == 4
    assert four.days_to_pass == pytest.approx(4.0 + 2.0 / 24.0)  # d0 08:00 -> d4 10:00
    five = simulate_challenge_single(ds, _rules(phases=[PhaseRule(target_pct=5.0, min_days=5)]), pol)
    assert five.outcome == "unfinished"  # the blocked d3 attempt must not count as a trading day


def test_consistency_uses_final_day_pnl_not_intraday_peak() -> None:
    # Target +10%, consistency 40%.
    # d0: +6R then -3R -> day PnL +3,000 (intraday peak 6,000); d1 +3R -> total 6,000;
    # d2 +4R -> total 10,000 = target. Best day = max(3,000, 3,000, 4,000) = 4,000 <= 40% x 10,000.
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 9), "BTC", 6.0),
            (_ts(0, 10), _ts(0, 11), "ETH", -3.0),
            (_ts(1, 9), _ts(1, 10), "SOL", 3.0),
            (_ts(2, 9), _ts(2, 10), "BTC", 4.0),
        ]
    )
    rules = _rules(phases=[PhaseRule(target_pct=10.0)], consistency_pct=40.0)
    res = simulate_challenge_single(ds, rules, _pol())
    assert res.outcome == "pass"
    assert res.days_to_pass == pytest.approx(2.0 + 2.0 / 24.0)
    batch = simulate_challenge_batch(ds, rules, [_pol()], np.array([0], dtype=np.int32))[0]
    assert OUTCOME_NAME[int(batch.outcomes[0])] == "pass"
    # Bootstrap (one week = one block) applies the same rule.
    boot = simulate_bootstrap_challenge(ds, rules, _pol(), n_paths=1, seed=0, horizon_days=7)
    assert boot.n_passed == 1
    # 35% cap: 4,000 > 3,500 -> not passed.
    strict = simulate_challenge_single(ds, _rules(phases=[PhaseRule(target_pct=10.0)], consistency_pct=35.0), _pol())
    assert strict.outcome == "unfinished"


def test_restart_fees_attempts_and_days() -> None:
    # Target +10%, restart after 3 days, fee $500.
    # start d0: +1% then the d4 entry is > 3 days after start -> restart.
    # start d4 09:00: +1% (101,000), d5 +10R -> 111,000 >= 110,000 -> pass at d5 10:00 = 1 day + 1 hour.
    # start d5 09:00: +10R -> 110,000 -> pass at 10:00 = 1 hour.
    ds = _ds(
        [
            (_ts(0, 9), _ts(0, 10), "BTC", 1.0),
            (_ts(4, 9), _ts(4, 10), "ETH", 1.0),
            (_ts(5, 9), _ts(5, 10), "SOL", 10.0),
        ]
    )
    sm = simulate_rolling_starts(ds, _rules(phases=[PhaseRule(target_pct=10.0)]), _pol(restart_after_days=3))
    assert (sm.n_passed, sm.n_restarted, sm.n_failed, sm.n_unfinished) == (2, 1, 0, 0)
    assert sm.pass_rate == pytest.approx(2 / 3)
    assert sm.expected_attempts == pytest.approx(1.5)  # 1 / (2/3)
    assert sm.expected_fees == pytest.approx(750.0)  # 1.5 x $500
    assert sm.days_median == pytest.approx(((1.0 + 1.0 / 24.0) + 1.0 / 24.0) / 2.0)


def test_phase_max_days_timeout_and_zero_pass_rate() -> None:
    # Phase limit 5 days: start d0 sees the d6 entry 6 days later -> max_days_timeout fail;
    # start d6 ends unfinished (1 hour of a 5-day window). pass_rate 0 -> infinite attempts/fees.
    ds = _ds([(_ts(0, 9), _ts(0, 10), "BTC", 1.0), (_ts(6, 9), _ts(6, 10), "ETH", 1.0)])
    sm = simulate_rolling_starts(ds, _rules(phases=[PhaseRule(target_pct=10.0, max_days=5)]), _pol())
    assert sm.fail_reason_counts == {"max_days_timeout": 1}
    assert sm.n_unfinished == 1
    assert sm.pass_rate == 0.0
    assert not np.isfinite(sm.expected_attempts)
    assert sm.to_dict()["expected_fees"] is None


def test_near_target_cut_behind_mode_and_per_symbol_cap() -> None:
    # Target +8%: near_frac 0.75 => cut once profit >= 6,000.
    # T1 +6R (1,000) -> 106,000; T2 risk 0.5x = 500, +2R -> 107,000;
    # T3 BTC: per-symbol cap 1.6% = 1,600; BTC reserved today 1,000 + 500 + 500 = 2,000 > 1,600 -> blocked.
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 9), "BTC", 6.0),
            (_ts(0, 10), _ts(0, 11), "BTC", 2.0),
            (_ts(0, 12), _ts(0, 13), "BTC", 4.0),
        ]
    )
    pol = _pol(near_frac=0.75, near_mult=0.5, per_symbol_daily_cap_pct=1.6)
    res = simulate_challenge_single(ds, _rules(phases=[PhaseRule(target_pct=8.0)]), pol)
    assert (res.trades_taken, res.final_balance) == (2, pytest.approx(107_000.0))
    # Behind mode: -2R at 2% -> 96,000 (4% below HWM >= 3.5%) -> next risk 1% -> +2R = +2,000.
    ds2 = _ds([(_ts(0, 8), _ts(0, 9), "BTC", -2.0), (_ts(1, 8), _ts(1, 9), "ETH", 2.0)])
    res2 = simulate_challenge_single(
        ds2, _rules(phases=[PhaseRule(target_pct=8.0)]), _pol(risk_pct=2.0, behind_dd_pct=3.5, behind_mult=0.5)
    )
    assert res2.final_balance == pytest.approx(98_000.0)


def test_same_timestamp_ordering_exits_first_and_zero_duration_last() -> None:
    # max_open_risk 1%: only one open trade at a time.
    # T1 08:00-10:00 +1R; T2 enters 10:00 exactly when T1 exits -> exit first frees risk -> taken (+1R).
    # T3 12:00-12:00 (zero duration) +2R and T4 12:00-13:00 -1R enter at the same ms: T3's entry
    # comes first (earlier exit), T4 is blocked because T3's zero-duration exit is processed only
    # after all entries at 12:00. Total = +1,000 +1,000 +2,000.
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 10), "BTC", 1.0),
            (_ts(0, 10), _ts(0, 11), "ETH", 1.0),
            (_ts(0, 12), _ts(0, 12), "SOL", 2.0),
            (_ts(0, 12), _ts(0, 13), "BTC", -1.0),
        ]
    )
    pol = _pol(max_open_risk_pct=1.0)
    res = simulate_challenge_single(ds, _rules(), pol)
    assert (res.trades_taken, res.final_balance) == (3, pytest.approx(104_000.0))
    batch = simulate_challenge_batch(ds, _rules(), [pol], np.array([0], dtype=np.int32))[0]
    assert OUTCOME_NAME[int(batch.outcomes[0])] == res.outcome


def test_timezone_and_epoch_parsing() -> None:
    df = pd.DataFrame(
        {
            "entry_time": ["2024-01-01T01:30:00+02:00", "2024-01-01 10:00:00"],
            "exit_time": ["2024-01-01T02:30:00+02:00", "2024-01-01 11:00:00"],
            "symbol": ["BTC", "ETH"],
            "R": [1.0, -1.0],
        }
    )
    ds = trades_from_dataframe(df)
    # 01:30 +02:00 = 2023-12-31 23:30 UTC; naive strings are UTC.
    assert int(ds.entry_ms[0]) == int(datetime(2023, 12, 31, 23, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert int(ds.entry_ms[1]) == int(datetime(2024, 1, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)
    assert len(ds.rolling_start_trade_indices()) == 2  # two distinct UTC calendar days
    epoch = trades_from_dataframe(
        pd.DataFrame({"entry_time": [1704067200], "exit_time": [1704070800], "symbol": ["BTC"], "R": [0.5]})
    )
    assert int(epoch.entry_ms[0]) == 1_704_067_200_000


def test_edge_cases_empty_all_losers_single_day_bad_rows() -> None:
    with pytest.raises(ValueError):
        trades_from_dataframe(pd.DataFrame(columns=["entry_time", "exit_time", "symbol", "R"]))
    with pytest.raises(ValueError, match="exit_time < entry_time"):
        _ds([(_ts(0, 10), _ts(0, 9), "BTC", 1.0)])
    with pytest.raises(ValueError, match="missing required column"):
        trades_from_dataframe(pd.DataFrame({"entry_time": [_ts(0)], "symbol": ["BTC"], "R": [1.0]}))
    # All losers, 1% of initial each on separate days, static 10%: after 10 losses 90,000 is not
    # below the floor; the 11th loss (89,000) breaches max_loss.
    losers = _ds([(_ts(d, 9), _ts(d, 10), "BTC", -1.0) for d in range(12)])
    res = simulate_challenge_single(losers, _rules(max_loss_pct=10.0), _pol(), record_breach_detail=True)
    assert (res.outcome, res.fail_reason, res.trades_taken) == ("fail", "max_loss", 11)
    assert res.final_balance == pytest.approx(89_000.0)
    assert res.breach_day_utc == "2024-01-11"
    # Single-day dataset: one rolling start, bootstrap and optimiser still run.
    one_day = _ds([(_ts(0, 9), _ts(0, 10), "BTC", 1.0), (_ts(0, 11), _ts(0, 12), "ETH", -0.5)])
    assert len(one_day.rolling_start_trade_indices()) == 1
    assert len(build_week_blocks(one_day)) == 1
    boot = simulate_bootstrap_challenge(one_day, _rules(), _pol(), n_paths=5, seed=1)
    assert boot.n_total == 5


# ---------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------


def test_week_blocks_keep_empty_weeks() -> None:
    # Trades in weeks 0, 1, 3, 5 (Monday-aligned); edge weeks 0 and 5 are trimmed (span >= 4 weeks),
    # leaving weeks 1..4 = 4 blocks of which weeks 2 and 4 are empty (idle weeks are real outcomes).
    ds = _ds([(_ts(7 * w + 1, 9), _ts(7 * w + 1, 10), "BTC", 1.0) for w in (0, 1, 3, 5)])
    pool = build_week_blocks(ds)
    assert len(pool) == 4
    assert [len(b.entry_rel_ms) for b in pool.blocks] == [1, 0, 1, 0]
    # Relative timestamps keep the time of week: Tuesday 09:00 = 1 day 9 hours.
    assert pool.blocks[0].entry_rel_ms[0] == (24 + 9) * 3_600_000


def _random_week_dataset(rng: np.random.Generator):
    """Random trades inside the Monday-Sunday week of BASE, plus an R=0 trade exiting at week end."""
    n = int(rng.integers(8, 30))
    rows = []
    for _ in range(n):
        ent = int(rng.integers(0, 6 * 24 * 60))
        dur = 0 if rng.random() < 0.1 else int(rng.integers(1, 36 * 60))
        ext = min(ent + dur, 7 * 24 * 60 - 1)
        rows.append(
            (
                _ts(0, 0, ent),
                _ts(0, 0, ext),
                str(rng.choice(["BTC", "ETH", "SOL"])),
                float(np.round(rng.choice([-1.1, -1.0, -1.0, -0.5, 0.4, 1.5, 2.0, 3.0]), 2)),
            )
        )
    rows.append((_ts(0, 0, 7 * 24 * 60 - 30), _ts(7), "BTC", 0.0))  # last exit == horizon (7 days)
    return _ds(rows)


def _random_rules(rng: np.random.Generator, allow_max_days: bool = True) -> PropRules:
    n_ph = int(rng.integers(1, 3))
    phases = [
        PhaseRule(
            target_pct=float(rng.choice([2.0, 3.0, 5.0, 8.0])),
            min_days=int(rng.integers(0, 4)),
            max_days=int(rng.choice([3, 5, 9])) if allow_max_days and rng.random() < 0.3 else None,
        )
        for _ in range(n_ph)
    ]
    trailing = rng.random() < 0.5
    return PropRules(
        name="R",
        phases=phases,
        daily_loss_pct=float(rng.choice([2.0, 3.0, 5.0])),
        daily_reset=int(rng.choice([0, 5, 17, 22])),
        max_loss_pct=float(rng.choice([4.0, 6.0, 10.0])),
        max_loss_type="trailing_closed" if trailing else "static",
        trailing_lock_at_initial=bool(trailing and rng.random() < 0.5),
        consistency_pct=float(rng.choice([30.0, 50.0])) if rng.random() < 0.3 else None,
        fee=500.0,
    )


def _random_policy(rng: np.random.Generator, restart_choices: tuple) -> SizingPolicy:
    return SizingPolicy(
        risk_pct=float(rng.choice([0.5, 1.0, 1.5, 2.5])),
        risk_basis=str(rng.choice(["initial_balance", "current_balance"])),
        near_frac=float(rng.choice([0.5, 0.75])) if rng.random() < 0.6 else None,
        near_mult=float(rng.choice([0.35, 0.5])),
        behind_dd_pct=float(rng.choice([1.0, 2.0])) if rng.random() < 0.4 else None,
        behind_mult=0.5,
        own_daily_stop_pct=float(rng.choice([1.5, 3.0, 6.0])),
        per_symbol_daily_cap_pct=float(rng.choice([1.0, 2.0])) if rng.random() < 0.4 else None,
        max_open_risk_pct=float(rng.choice([1.0, 2.0, 4.0])) if rng.random() < 0.6 else None,
        restart_after_days=restart_choices[int(rng.integers(0, len(restart_choices)))],
    )


def test_bootstrap_single_block_reproduces_scalar_challenge() -> None:
    rng = np.random.default_rng(2024)
    checked = 0
    for _ in range(40):
        ds = _random_week_dataset(rng)
        rules = _random_rules(rng)
        pol = _random_policy(rng, (None, 2, 4))
        scalar = simulate_challenge_single(ds, rules, pol, start_trade_idx=0)
        boot = simulate_bootstrap_challenge(ds, rules, pol, n_paths=1, seed=0, horizon_days=7)
        assert OUTCOME_NAME[int(boot.outcomes[0])] == scalar.outcome
        if scalar.outcome == "pass":
            assert boot.days_per_run[0] == pytest.approx(scalar.days_to_pass)
        if scalar.fail_reason is not None:
            assert int(boot.fail_reasons[0]) == REASON_CODE[scalar.fail_reason]
        checked += 1
    assert checked == 40


# ---------------------------------------------------------------------------------------------
# Scalar vs vectorised batch on random trade lists
# ---------------------------------------------------------------------------------------------


def test_batch_equals_scalar_on_random_trade_lists() -> None:
    rng = np.random.default_rng(7)
    for _case in range(12):
        n = int(rng.integers(40, 90))
        rows = []
        for _ in range(n):
            ent = int(rng.integers(0, 25 * 24 * 60))
            ent -= ent % 30  # many shared timestamps
            dur = 0 if rng.random() < 0.08 else int(rng.integers(30, 3 * 24 * 60))
            rows.append(
                (
                    _ts(0, 0, ent),
                    _ts(0, 0, ent + dur),
                    str(rng.choice(["BTC", "ETH", "SOL", "AVAX"])),
                    float(np.round(rng.normal(0.15, 1.4), 2)),
                )
            )
        ds = _ds(rows)
        rules = _random_rules(rng)
        policies = [_random_policy(rng, (None, 4, 9)) for _ in range(4)]
        starts = ds.rolling_start_trade_indices()
        batch = simulate_challenge_batch(ds, rules, policies, starts)
        for p_idx, pol in enumerate(policies):
            for s_idx, trade_idx in enumerate(starts):
                sc = simulate_challenge_single(ds, rules, pol, int(trade_idx))
                assert OUTCOME_NAME[int(batch[p_idx].outcomes[s_idx])] == sc.outcome
                expected_code = REASON_CODE[sc.fail_reason] if sc.fail_reason else 0
                assert int(batch[p_idx].fail_reasons[s_idx]) == expected_code
                if sc.outcome == "pass":
                    assert batch[p_idx].days_per_run[s_idx] == pytest.approx(sc.days_to_pass)


# ---------------------------------------------------------------------------------------------
# Funded phase (hand calculated)
# ---------------------------------------------------------------------------------------------


def test_funded_payouts_above_buffer_with_split() -> None:
    # Instant account, static 10%, payouts every 7 days, keep 4% buffer (104,000), split 80%.
    # d0 09:00 +5R (1% initial = 1,000/R) -> 105,000; payout d7 09:00: gross 1,000 -> net 800.
    # d8 +3R -> 107,000; payout d14 09:00: gross 3,000 -> net 2,400. Horizon 20 days.
    rules = _rules(phases=[], max_loss_pct=10.0, payout_frequency_days=7)
    ds = _ds([(_ts(0, 9), _ts(0, 10), "BTC", 5.0), (_ts(8, 9), _ts(8, 10), "ETH", 3.0)])
    lost, reason, bal, payouts = simulate_funded_single(ds, rules, _pol(), horizon_days=20)
    assert (lost, reason) == (False, None)
    assert payouts == [(pytest.approx(7.0), pytest.approx(800.0)), (pytest.approx(14.0), pytest.approx(2_400.0))]
    assert bal == pytest.approx(104_000.0)


def test_funded_trailing_floor_lock_and_withdrawals() -> None:
    # Trailing 6%, payout every 7 days above 104,000, daily limit 10%.
    # d0 +8R -> 108,000 (HWM). d7 payout: gross 4,000 -> balance 104,000.
    # d8 -5R -> 99,000.
    ds = _ds([(_ts(0, 9), _ts(0, 10), "BTC", 8.0), (_ts(8, 9), _ts(8, 10), "ETH", -5.0)])
    base = dict(phases=[], max_loss_pct=6.0, max_loss_type="trailing_closed", daily_loss_pct=10.0, payout_frequency_days=7)
    # No lock: the withdrawal lowers the HWM to 104,000 -> floor 97,760 -> 99,000 survives.
    lost, _, bal, payouts = simulate_funded_single(ds, _rules(**base), _pol(), horizon_days=10)
    assert not lost and bal == pytest.approx(99_000.0)
    assert payouts == [(pytest.approx(7.0), pytest.approx(3_200.0))]
    # Lock at initial: 108,000 x 0.94 = 101,520 >= 100,000 -> floor locked at 100,000 for good,
    # so 99,000 breaches max loss.
    lost_l, reason_l, bal_l, _ = simulate_funded_single(
        ds, _rules(**base, trailing_lock_at_initial=True), _pol(), horizon_days=10
    )
    assert (lost_l, reason_l) == (True, "max_loss")
    assert bal_l == pytest.approx(99_000.0)


def test_funded_cppi_multiplier_scales_down_and_caps() -> None:
    # Static 10% floor (90,000), CPPI c0 = 10% => multiplier = min(1.5, cushion% / 10).
    # T1 cushion 10% -> 1.0x -> risk 1,000, -1R -> 99,000.
    # T2 cushion 9% -> 0.9x -> risk 900, +2R -> 100,800.
    # T3 cushion 10.8% -> 1.08x -> risk 1,080, +1R -> 101,880.
    rules = _rules(phases=[], max_loss_pct=10.0, payout_frequency_days=30)
    ds = _ds(
        [
            (_ts(0, 9), _ts(0, 10), "BTC", -1.0),
            (_ts(1, 9), _ts(1, 10), "ETH", 2.0),
            (_ts(2, 9), _ts(2, 10), "SOL", 1.0),
        ]
    )
    _, _, bal, _ = simulate_funded_single(ds, rules, _pol(cppi_c0=10.0, cppi_cap=1.5), horizon_days=10)
    assert bal == pytest.approx(101_880.0)
    # Cushion 20% (20% static max loss) -> 2.0x capped at 1.5x -> +1R = +1,500.
    one = _ds([(_ts(0, 9), _ts(0, 10), "BTC", 1.0)])
    _, _, bal_cap, _ = simulate_funded_single(
        one, _rules(phases=[], max_loss_pct=20.0), _pol(cppi_c0=10.0, cppi_cap=1.5), horizon_days=5
    )
    assert bal_cap == pytest.approx(101_500.0)


def test_funded_consistency_uses_final_day_pnl() -> None:
    # Consistency 50%, keep buffer 0%, payouts every 7 days (first entry d0 08:00 -> payout d7 08:00).
    # d0: +6R then -4R -> day +2,000 (peak 6,000); d1 +2R -> cycle profit 4,000, best day 2,000
    # <= 50% x 4,000 -> paid: gross 4,000, net 3,200.
    rules = _rules(phases=[], consistency_pct=50.0, payout_frequency_days=7)
    ds = _ds(
        [
            (_ts(0, 8), _ts(0, 9), "BTC", 6.0),
            (_ts(0, 10), _ts(0, 11), "ETH", -4.0),
            (_ts(1, 9), _ts(1, 10), "SOL", 2.0),
        ]
    )
    _, _, bal, payouts = simulate_funded_single(ds, rules, _pol(keep_buffer_pct=0.0), horizon_days=10)
    assert payouts == [(pytest.approx(7.0), pytest.approx(3_200.0))]
    assert bal == pytest.approx(100_000.0)
    # 45% cap: 2,000 > 1,800 -> deferred, no payout within 10 days.
    _, _, _, none = simulate_funded_single(
        ds, _rules(phases=[], consistency_pct=45.0, payout_frequency_days=7), _pol(keep_buffer_pct=0.0), horizon_days=10
    )
    assert none == []


def _repeating_weeks(n_weeks: int) -> pd.DataFrame:
    rows = []
    for w in range(n_weeks):
        d = 7 * w
        rows += [
            (_ts(d, 0), _ts(d, 6), "BTC", 1.0),  # Monday 00:00 (first entry = funding time)
            (_ts(d + 1, 10), _ts(d + 1, 12), "ETH", -1.0),
            (_ts(d + 2, 9), _ts(d + 3, 9), "SOL", 2.0),
            (_ts(d + 3, 8), _ts(d + 3, 8, 30), "BTC", 0.5),
        ]
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "symbol", "R"])


@pytest.mark.parametrize("max_loss_type", ["static", "trailing_closed"])
def test_funded_bootstrap_matches_historical_run_on_identical_weeks(max_loss_type: str) -> None:
    ds = trades_from_dataframe(_repeating_weeks(60))
    rules = _rules(phases=[], max_loss_pct=6.0, max_loss_type=max_loss_type, daily_loss_pct=5.0)
    pol = _pol(funded_risk_pct=0.5, cppi_c0=6.0, risk_basis="current_balance")
    lost, _, _, payouts = simulate_funded_single(ds, rules, pol, horizon_days=365)
    fb = simulate_funded_bootstrap(ds, rules, pol, n_paths=3, seed=5)
    total = sum(net for _, net in payouts)
    assert payouts, "expected payouts on a steadily profitable week pattern"
    assert fb.p_lose_account == (1.0 if lost else 0.0)
    assert np.allclose(fb.path_12m_total_usd, total)
    month_days = 365.0 / 12.0
    expected_months = np.zeros(12)
    for day, net in payouts:
        expected_months[min(int((day * 86_400_000 - 1) // (month_days * 86_400_000)), 11)] += net
    assert np.allclose(fb.per_month_mean_usd, expected_months)
    assert fb.monthly_payout_mean_usd == pytest.approx(total / 12.0)


def test_challenge_bootstrap_matches_scalar_on_identical_weeks() -> None:
    ds = trades_from_dataframe(_repeating_weeks(60))
    rules = load_preset("two_step_classic")
    pol = _pol(risk_pct=0.5, own_daily_stop_pct=3.0)
    scalar = simulate_challenge_single(ds, rules, pol)
    boot = simulate_bootstrap_challenge(ds, rules, pol, n_paths=3, seed=9)
    assert scalar.outcome == "pass"
    assert boot.n_passed == 3
    assert np.allclose(boot.days_to_pass, scalar.days_to_pass)


# ---------------------------------------------------------------------------------------------
# Optimiser honesty and funded-risk constraint
# ---------------------------------------------------------------------------------------------


def _small_grid() -> list[SizingPolicy]:
    return [
        SizingPolicy(risk_pct=0.5, own_daily_stop_pct=2.5, max_open_risk_pct=2.5, restart_after_days=60),
        SizingPolicy(risk_pct=1.0, own_daily_stop_pct=3.5, max_open_risk_pct=3.5, restart_after_days=90),
    ]


def test_optimiser_selection_ignores_validation_slice() -> None:
    df = generate_synthetic_trades(SyntheticConfig(days=240, seed=3))
    ds = trades_from_dataframe(df)
    rules = load_preset("two_step_classic")
    opt, _ = ds.split_chronological(0.6)
    clean = optimise_on_train(opt, rules, objective="max_value", max_fail=0.6, grid=_small_grid(), funded_paths=40)

    corrupt_df = df.copy()
    corrupt_df.loc[corrupt_df.index[len(opt):], "R"] = -5.0
    opt_c, _ = trades_from_dataframe(corrupt_df).split_chronological(0.6)
    corrupt = optimise_on_train(opt_c, rules, objective="max_value", max_fail=0.6, grid=_small_grid(), funded_paths=40)
    assert clean.best_policy == corrupt.best_policy
    assert [c.to_dict() for c in clean.funded_candidates] == [c.to_dict() for c in corrupt.funded_candidates]


def test_funded_variant_respects_p_lose_limit_and_value_formula() -> None:
    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=300, seed=19)))
    rules = load_preset("one_step")
    res = run_honest_optimisation(
        ds, rules, objective="max_value", max_fail=0.7, n_bootstrap=60, grid=_small_grid(),
        max_lose_funded=0.10, funded_paths=60,
    )
    chosen = res.best_policy
    cands = res.funded_candidates
    match = [c for c in cands if c.policy.funded_risk_pct == chosen.funded_risk_pct and c.policy.cppi_c0 == chosen.cppi_c0]
    assert len(match) == 1
    ok = [c for c in cands if c.summary.p_lose_account <= 0.10]
    if ok:
        assert res.funded_constraint_met and match[0].meets_constraint
        assert match[0].summary.total_12m_payout_mean_usd == pytest.approx(
            max(c.summary.total_12m_payout_mean_usd for c in ok)
        )
    else:
        assert not res.funded_constraint_met
        assert any("HIGH RISK" in w for w in res.warnings)
    # The plan card always states the funded-account risk.
    card = res.plan_card_summary()
    assert "FUNDED RISK" in card and "P(lose funded account within 12m)" in card
    # Value = E[payouts] - fees x (1 + P(lose)): 12,000 - 800 x 1.25 = 11,000.
    fake = SimpleNamespace(total_12m_payout_mean_usd=12_000.0, p_lose_account=0.25)
    assert expected_net_12m_usd(800.0, fake) == pytest.approx(11_000.0)  # type: ignore[arg-type]
    assert expected_net_12m_usd(float("inf"), fake) == float("-inf")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# AI layer
# ---------------------------------------------------------------------------------------------


def test_compact_json_stays_valid_and_below_limit_with_escaping() -> None:
    payload = {"blob": ['"quoted" \\ value ' * 20 for _ in range(60)], "nested": {"x": "y" * 4000}}
    out = compact_json(payload)
    assert len(out) < 3000
    assert json.loads(out)["truncated"] is True


def test_extractor_exact_sdk_call_and_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PP_MODEL", raising=False)
    captured: dict[str, object] = {}
    good = load_preset("two_step_classic")

    class _Client:
        class beta:
            class messages:
                @staticmethod
                def parse(**kwargs: object) -> object:
                    captured.update(kwargs)
                    return SimpleNamespace(stop_reason="end_turn", parsed_output=good)

    out = extract_rules_from_text("Phase 1 8%", client=_Client())
    assert out == good
    assert captured == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
        "output_config": {"effort": "high"},
        "system": EXTRACT_PROMPT,
        "messages": [{"role": "user", "content": "Phase 1 8%"}],
        "output_format": PropRules,
    }

    class _InvalidClient:
        class beta:
            class messages:
                @staticmethod
                def parse(**kwargs: object) -> object:
                    # What the SDK raises when the model's JSON violates PropRules validators.
                    pydantic.TypeAdapter(PropRules).validate_json('{"daily_loss_pct": 5, "max_loss_pct": 10, "fee": 1, "daily_reset": 30}')
                    raise AssertionError("unreachable")

    with pytest.raises(RuleExtractionError, match="did not validate"):
        extract_rules_from_text("x", client=_InvalidClient())

    class _ApiErrorClient:
        class beta:
            class messages:
                @staticmethod
                def parse(**kwargs: object) -> object:
                    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                    raise anthropic.APIStatusError("overloaded", response=httpx.Response(529, request=req), body=None)

    with pytest.raises(RuleExtractionError, match="API error"):
        extract_rules_from_text("x", client=_ApiErrorClient())


def test_offline_rules_parser_matches_preset_and_flags_gaps() -> None:
    text = (ROOT / "examples" / "rules_sample.txt").read_text(encoding="utf-8")
    parsed = extract_rules_offline(text)
    preset = load_preset("two_step_classic")
    keys = ("daily_loss_pct", "daily_reset", "max_loss_pct", "max_loss_type", "fee", "profit_split",
            "payout_frequency_days", "account_size", "consistency_pct")
    assert {k: getattr(parsed, k) for k in keys} == {k: getattr(preset, k) for k in keys}
    assert [(p.target_pct, p.min_days, p.max_days) for p in parsed.phases] == [(8.0, 4, None), (5.0, 4, None)]
    assert any("basis" in a for a in parsed.ambiguities)  # balance vs equity was not stated
    vague = extract_rules_offline("Trailing drawdown 6%. Daily loss 3%.")
    assert vague.max_loss_type == "trailing_closed" and vague.trailing_lock_at_initial is False
    assert any("locks" in a for a in vague.ambiguities)
    assert any("fee" in a.lower() for a in vague.ambiguities)


def test_copilot_tools_return_correct_numbers_and_sdk_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PP_MODEL", raising=False)
    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=150, seed=21)))
    rules = load_preset("two_step_classic")
    session = PlanSession(dataset=ds, rules=rules, n_bootstrap=40, seed=42)
    tools = {t.name: t for t in build_copilot_tools(session)}
    pol = session.active_policy()
    opt, val = ds.split_chronological(0.6)

    rules_out = json.loads(tools["get_rules"]())
    assert rules_out["daily_loss_pct"] == 5.0 and rules_out["max_loss_type"] == "static"
    stats = json.loads(tools["get_trade_stats"]())
    assert stats["full"]["n_trades"] == len(ds)
    assert stats["optimise_60pct"]["n_trades"] == len(opt)
    sim = json.loads(tools["simulate_policy"](policy_json="{}", period="optimise"))
    direct = simulate_rolling_starts(opt, rules, pol)
    assert sim["metrics"]["pass_rate"] == round(direct.pass_rate, 4)
    assert sim["metrics"]["n_finished"] == direct.n_finished
    cmp_out = json.loads(tools["compare_policies"](policies_json=json.dumps([{"risk_pct": 0.5}, {"risk_pct": 1.0}])))
    direct_val = simulate_rolling_starts(val, rules, pol.model_copy(update={"risk_pct": 0.5}))
    assert cmp_out["results"][0]["validate"]["pass"] == round(direct_val.pass_rate, 3)
    bad = json.loads(tools["simulate_policy"](policy_json='{"risk_pct": -1}', period="validate"))
    assert "error" in bad
    for name in ("funded_projection", "explain_failures"):
        out = tools[name](policy_json="{}")
        assert len(out) < 3000 and json.loads(out)
    opt_out = tools["optimise"](objective="cheapest", max_fail=0.6)
    assert len(opt_out) < 3000 and json.loads(opt_out)["objective"] == "cheapest"

    captured: dict[str, object] = {}

    class _Runner:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs: object) -> list[object]:
                    captured.update(kwargs)
                    return [SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")])]

    reply = run_copilot_query(session, "hi", history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], client=_Runner())
    assert reply.used_ai and reply.text == "ok"
    assert captured["model"] == "claude-opus-5-5"
    assert captured["max_tokens"] == 16000 and captured["max_iterations"] == 20
    assert captured["betas"] == ["server-side-fallback-2026-07-01"] and captured["fallbacks"] == "default"
    assert captured["output_config"] == {"effort": "high"} and captured["system"] == COPILOT_PROMPT
    assert [m["role"] for m in captured["messages"]] == ["user", "assistant", "user"]  # type: ignore[index]
    assert sorted(t.name for t in captured["tools"]) == sorted(tools)  # type: ignore[union-attr]

    class _StatusErr:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs: object) -> list[object]:
                    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                    raise anthropic.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)

    fb = run_copilot_query(session, "hi", client=_StatusErr())
    assert not fb.used_ai and "Recommended Pass Plan" in fb.text
    assert "Funded risk" in fb.text and "API error" in (fb.fallback_reason or "")


def test_cli_rules_offline_and_simulate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from proppilot.cli import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_yaml = tmp_path / "r.yaml"
    assert main(["rules", "--from", str(ROOT / "examples" / "rules_sample.txt"), "--out", str(out_yaml)]) == 0
    printed = capsys.readouterr().out
    assert "offline keyword parser" in printed and "AMBIGUITIES" in printed
    assert out_yaml.exists()
    csv = tmp_path / "t.csv"
    generate_synthetic_trades(SyntheticConfig(days=90, seed=4)).to_csv(csv, index=False)
    assert main(["simulate", "--trades", str(csv), "--rules", str(out_yaml), "--bootstrap", "30"]) == 0
    assert "P(lose account)" in capsys.readouterr().out
