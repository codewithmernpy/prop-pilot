"""Line-by-line audit tests for PropPilot ingestion, challenge, and funded simulators."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proppilot.ingest import load_trades, trades_from_dataframe
from proppilot.policy import FundedPolicy, SizingPolicy
from proppilot.rules import PhaseRule, PropRules, load_preset
from proppilot.sim import (
    simulate_challenge_batch,
    simulate_challenge_single,
    simulate_funded_single,
    simulate_rolling_starts,
)


def _ts(day: int, hour: int = 10, minute: int = 0) -> str:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(days=day, hours=hour, minutes=minute)
    return base.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_df(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "symbol", "R"])


def _base_rules(**overrides: object) -> PropRules:
    params: dict[str, object] = {
        "name": "Test Rules",
        "phases": [PhaseRule(name="P1", target_pct=8.0, min_days=0, max_days=None)],
        "daily_loss_pct": 5.0,
        "daily_basis": "day_start_balance",
        "daily_reset": 0,
        "max_loss_pct": 10.0,
        "max_loss_type": "static",
        "consistency_pct": None,
        "fee": 500.0,
        "profit_split": 80.0,
        "payout_frequency_days": 14,
        "account_size": 100_000.0,
        "max_risk_per_trade_pct": None,
        "ambiguities": [],
    }
    params.update(overrides)
    return PropRules(**params)


def _base_policy(**overrides: object) -> SizingPolicy:
    params: dict[str, object] = {
        "risk_pct": 1.0,
        "risk_basis": "initial_balance",
        "near_frac": None,
        "near_mult": 1.0,
        "behind_dd_pct": None,
        "behind_mult": 1.0,
        "own_daily_stop_pct": 10.0,
        "per_symbol_daily_cap_pct": None,
        "max_open_risk_pct": None,
        "restart_after_days": None,
        "funded_risk_pct": 1.0,
        "keep_buffer_pct": 4.0,
        "cppi_cap": 1.0,
        "cppi_c0": None,
    }
    params.update(overrides)
    return SizingPolicy(**params)


def test_target_hit_exact_and_two_phase_pass() -> None:
    rules = _base_rules(
        phases=[
            PhaseRule(name="Phase 1", target_pct=8.0, min_days=2),
            PhaseRule(name="Phase 2", target_pct=5.0, min_days=2),
        ]
    )
    pol = _base_policy(risk_pct=1.0)
    ds = trades_from_dataframe(
        _make_df(
            [
                # Phase 1: +4R day 0, +4R day 1 -> exact +8.0% on day 1
                (_ts(0, 9), _ts(0, 12), "BTCUSDT", 4.0),
                (_ts(1, 9), _ts(1, 12), "ETHUSDT", 4.0),
                # Phase 2 starts fresh at $100k: +2R day 2, +3R day 3 -> exact +5.0%
                (_ts(2, 9), _ts(2, 12), "BTCUSDT", 2.0),
                (_ts(3, 9), _ts(3, 12), "SOLUSDT", 3.0),
            ]
        )
    )
    res = simulate_challenge_single(ds, rules, pol)
    assert res.outcome == "pass"
    assert res.days_to_pass == pytest.approx(3.125)
    assert res.final_balance == pytest.approx(105_000.0)


def test_daily_loss_breach_by_two_same_day_losers() -> None:
    rules = _base_rules(daily_loss_pct=5.0)
    pol = _base_policy(risk_pct=2.0, own_daily_stop_pct=10.0)

    # Exact -5.0% does NOT breach (> limit), second day breaches at -5.2%.
    ds_exact = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 9), "BTCUSDT", -1.5),  # -3.0%
                (_ts(0, 10), _ts(0, 11), "ETHUSDT", -1.0),  # -2.0% -> total -5.0%
            ]
        )
    )
    res_exact = simulate_challenge_single(ds_exact, rules, pol)
    assert res_exact.outcome == "unfinished"

    ds_breach = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 9), "BTCUSDT", -1.5),  # -3.0%
                (_ts(0, 10), _ts(0, 11), "ETHUSDT", -1.1),  # -2.2% -> total -5.2% > 5.0%
            ]
        )
    )
    res_breach = simulate_challenge_single(ds_breach, rules, pol, record_breach_detail=True)
    assert res_breach.outcome == "fail"
    assert res_breach.fail_reason == "daily_loss"
    assert res_breach.breach_symbol == "ETHUSDT"


def test_static_vs_trailing_max_loss() -> None:
    # +5% (105,000 HWM), then two -3.5% losses on separate days -> balance 98,000.
    # Static 6% floor = 94,000 -> survives. Trailing 6% floor = 105,000 * 0.94 = 98,700 -> breaches.
    ds = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 9), _ts(0, 12), "BTCUSDT", 5.0),
                (_ts(1, 9), _ts(1, 12), "ETHUSDT", -3.5),
                (_ts(2, 9), _ts(2, 12), "SOLUSDT", -3.5),
            ]
        )
    )
    pol = _base_policy(risk_pct=1.0, own_daily_stop_pct=5.0)
    static_rules = _base_rules(daily_loss_pct=5.0, max_loss_pct=6.0, max_loss_type="static")
    trail_rules = _base_rules(daily_loss_pct=5.0, max_loss_pct=6.0, max_loss_type="trailing_closed")

    res_static = simulate_challenge_single(ds, static_rules, pol)
    res_trail = simulate_challenge_single(ds, trail_rules, pol)
    assert res_static.outcome == "unfinished"
    assert res_trail.outcome == "fail"
    assert res_trail.fail_reason == "max_loss"


def test_min_trading_days_enforced() -> None:
    rules = _base_rules(phases=[PhaseRule(name="P1", target_pct=8.0, min_days=4)])
    pol = _base_policy(risk_pct=2.0)
    ds = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 10), "BTCUSDT", 4.5),  # +9% on day 0, only 1 trading day
                (_ts(1, 8), _ts(1, 10), "BTCUSDT", 0.1),  # day 2
                (_ts(2, 8), _ts(2, 10), "BTCUSDT", 0.1),  # day 3
                (_ts(3, 8), _ts(3, 10), "BTCUSDT", 0.1),  # day 4 -> passes now!
            ]
        )
    )
    res = simulate_challenge_single(ds, rules, pol)
    assert res.outcome == "pass"
    assert res.days_to_pass == pytest.approx(3.0 + 2 / 24)

    ds_early = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 10), "BTCUSDT", 4.5),
                (_ts(1, 8), _ts(1, 10), "BTCUSDT", 0.1),
                (_ts(2, 8), _ts(2, 10), "BTCUSDT", 0.1),
            ]
        )
    )
    assert simulate_challenge_single(ds_early, rules, pol).outcome == "unfinished"


def test_daily_reset_hour_boundaries() -> None:
    # Daily reset at 17:00 UTC:
    # - 15:00 UTC and 18:00 UTC straddle 17:00 UTC -> DIFFERENT reset days -> no daily breach!
    # - 15:00 UTC and 16:30 UTC are in the SAME reset day -> breaches 5%!
    rules = _base_rules(daily_loss_pct=5.0, daily_reset=17)
    pol = _base_policy(risk_pct=3.0)

    ds_split = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 14), _ts(0, 15), "BTCUSDT", -1.0),  # -3% before 17:00 reset
                (_ts(0, 17, 30), _ts(0, 18), "ETHUSDT", -1.0),  # -3% after 17:00 reset
            ]
        )
    )
    assert simulate_challenge_single(ds_split, rules, pol).outcome == "unfinished"

    ds_same = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 14), _ts(0, 15), "BTCUSDT", -1.0),  # -3% before 17:00
                (_ts(0, 15, 30), _ts(0, 16, 30), "ETHUSDT", -1.0),  # -3% also before 17:00
            ]
        )
    )
    res_same = simulate_challenge_single(ds_same, rules, pol)
    assert res_same.outcome == "fail"
    assert res_same.fail_reason == "daily_loss"


def test_overlapping_trades_reserve_risk_and_max_open_guard() -> None:
    rules = _base_rules(phases=[PhaseRule(name="P1", target_pct=20.0, min_days=0)])
    # Use current_balance basis:
    # Trade 0 opens at 08:00 (balance=100k -> reserves $1,000), closes at 14:00 with +10R (+$10k).
    # Trade 1 opens at 09:00 while Trade 0 is still open (balance is STILL 100k -> reserves $1,000).
    # Trade 2 opens at 09:30 with max_open_risk_pct=2.0 -> blocked because open_risk ($2k) + $1k > $2k!
    pol = _base_policy(risk_pct=1.0, risk_basis="current_balance", max_open_risk_pct=2.0)
    ds = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 14), "BTCUSDT", 10.0),
                (_ts(0, 9), _ts(0, 16), "ETHUSDT", 1.0),
                (_ts(0, 9, 30), _ts(0, 12), "SOLUSDT", -5.0),  # would lose $5k if not blocked!
            ]
        )
    )
    res = simulate_challenge_single(ds, rules, pol)
    assert res.trades_taken == 2
    assert res.final_balance == pytest.approx(111_000.0)


def test_own_daily_stop_blocks_entries_before_firm_breach() -> None:
    rules = _base_rules(daily_loss_pct=5.0)
    pol = _base_policy(risk_pct=1.5, own_daily_stop_pct=3.0)
    ds = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 9), "BTCUSDT", -1.0),   # realised loss = 1.5%
                (_ts(0, 10), _ts(0, 14), "ETHUSDT", -1.0), # new risk 1.5% -> 1.5+1.5=3.0 <= 3.0 -> allowed
                (_ts(0, 11), _ts(0, 15), "SOLUSDT", -2.0), # open=1.5% + realised=1.5% + new=1.5% = 4.5% > 3.0 -> BLOCKED!
            ]
        )
    )
    res = simulate_challenge_single(ds, rules, pol)
    assert res.trades_taken == 2
    assert res.outcome == "unfinished"
    assert res.final_balance == pytest.approx(97_000.0)


def test_per_symbol_daily_cap_and_near_target_and_behind_cut() -> None:
    rules = _base_rules(phases=[PhaseRule(name="P1", target_pct=8.0, min_days=0)])
    # Trade 0: +6R at 1% = +6% (balance 106k, 75% of +8% target reached!)
    # Trade 1 (same day BTCUSDT): near_mult=0.5 cuts risk to 0.5% ($500); +2R = +$1k -> bal=107k
    # Trade 2 (same day BTCUSDT, per_symbol_daily_cap_pct=1.6%): 1% + 0.5% + 0.5% = 2.0% > 1.6% -> BLOCKED!
    pol = _base_policy(
        risk_pct=1.0,
        near_frac=0.75,
        near_mult=0.5,
        per_symbol_daily_cap_pct=1.6,
    )
    ds = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 9), "BTCUSDT", 6.0),
                (_ts(0, 10), _ts(0, 11), "BTCUSDT", 2.0),
                (_ts(0, 12), _ts(0, 13), "BTCUSDT", 4.0),
            ]
        )
    )
    res = simulate_challenge_single(ds, rules, pol)
    assert res.trades_taken == 2
    assert res.final_balance == pytest.approx(107_000.0)

    # Behind mode cut test: lose -4% -> dd=4% >= behind_dd_pct=3.5 -> next risk halved
    pol_dd = _base_policy(risk_pct=2.0, behind_dd_pct=3.5, behind_mult=0.5)
    ds_dd = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 8), _ts(0, 9), "BTCUSDT", -2.0),  # -4% -> balance 96k
                (_ts(1, 8), _ts(1, 9), "ETHUSDT", 2.0),   # risk halved to 1% -> +2k -> 98k
            ]
        )
    )
    res_dd = simulate_challenge_single(ds_dd, rules, pol_dd)
    assert res_dd.final_balance == pytest.approx(98_000.0)


def test_restart_rule_counts_fees_and_unfinished_excluded() -> None:
    rules = _base_rules(fee=500.0, phases=[PhaseRule(name="P1", target_pct=10.0, min_days=0)])
    # Day 0 starts at +2%, Day 5 drops -3%, Day 12 makes +5%, Day 13 makes +5%.
    # Day 0 hits restart at day 12 (>10d). Day 5 reaches -3+5+5 = +7% (span 8d <= 10d -> unfinished).
    # Day 12 reaches +5+5 = +10% on Day 13 -> passes! Day 13 reaches +5% -> unfinished.
    ds = trades_from_dataframe(
        _make_df(
            [
                (_ts(0, 9), _ts(0, 12), "BTCUSDT", 2.0),
                (_ts(5, 9), _ts(5, 12), "BTCUSDT", -3.0),
                (_ts(12, 9), _ts(12, 12), "ETHUSDT", 5.0),
                (_ts(13, 9), _ts(13, 12), "SOLUSDT", 5.0),
            ]
        )
    )
    pol_restart = _base_policy(risk_pct=1.0, restart_after_days=10)
    summary = simulate_rolling_starts(ds, rules, pol_restart)
    assert summary.n_restarted == 1
    assert summary.n_passed == 1
    assert summary.n_unfinished == 2
    assert summary.n_finished == 2
    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.expected_attempts == pytest.approx(2.0)
    assert summary.expected_fees == pytest.approx(1000.0)


def test_consistency_rule_and_funded_payouts_buffer_cppi() -> None:
    instant_rules = load_preset("instant")
    # consistency_pct=15.0, keep_buffer_pct=4.0, profit_split=80.0, payout_frequency_days=14.
    # Use fixed initial_balance sizing ($1,000 per R) so profit is linear:
    # Day 0: +$5,000. Days 1..15: +$2,000/day -> day 14 cycle profit = $31,000 (5/31 = 16.1% > 15% -> deferred).
    # Day 28 cycle profit = $35,000 (5/35 = 14.3% <= 15% -> paid out!).
    rows = [(_ts(0, 9), _ts(0, 12), "BTCUSDT", 5.0)]
    for d in range(1, 16):
        rows.append((_ts(d, 9), _ts(d, 12), "ETHUSDT", 2.0))
    ds = trades_from_dataframe(_make_df(rows))

    pol_fixed = _base_policy(
        risk_basis="initial_balance",
        funded_risk_pct=1.0,
        keep_buffer_pct=4.0,
        cppi_cap=1.0,
        cppi_c0=None,
    )
    lost, reason, final_bal, payouts = simulate_funded_single(ds, instant_rules, pol_fixed, horizon_days=35)
    assert not lost
    assert reason is None
    assert len(payouts) == 1
    assert payouts[0][0] == pytest.approx(28.0)
    assert payouts[0][1] == pytest.approx(31_000.0 * 0.80)
    assert final_bal == pytest.approx(104_000.0)

    # CPPI scaling test: static 10% floor -> initial cushion = 10%. multiplier = min(cap, cushion / c0)
    # = min(2.0, 10 / 5) = 2.0 -> risk 2% of $100k = $2,000; +1R -> $102,000.
    static_rules = _base_rules(phases=[], max_loss_pct=10.0, max_loss_type="static", consistency_pct=None)
    cppi_pol = FundedPolicy(risk_pct=1.0, keep_buffer_pct=4.0, cppi_cap=2.0, cppi_c0=5.0)
    ds_one = trades_from_dataframe(_make_df([(_ts(0, 9), _ts(0, 12), "BTCUSDT", 1.0)]))
    _, _, bal_cppi, _ = simulate_funded_single(ds_one, static_rules, cppi_pol, horizon_days=10)
    assert bal_cppi == pytest.approx(102_000.0)


def test_batch_matches_scalar_identically_and_quantcritic_ingestion(tmp_path: Path) -> None:
    # Include extra QuantCritic columns and JSON export format
    records = [
        {
            "entry_time": _ts(d, 9),
            "exit_time": _ts(d, 14),
            "symbol": "BTCUSDT" if d % 2 == 0 else "ETHUSDT",
            "side": "LONG",
            "R": 1.8 if d % 3 != 0 else -1.0,
            "strategy": "QuantCritic_Trend",
            "confidence": 0.81,
            "fees_usd": 4.2,
        }
        for d in range(25)
    ]
    json_path = tmp_path / "quantcritic_export.json"
    json_path.write_text(json.dumps({"trades": records}), encoding="utf-8")
    ds = load_trades(json_path)
    assert len(ds) == 25

    rules = load_preset("two_step_classic")
    policies = [
        _base_policy(risk_pct=0.8, near_frac=0.75, near_mult=0.5, own_daily_stop_pct=3.0, restart_after_days=12),
        _base_policy(risk_pct=1.2, behind_dd_pct=3.0, behind_mult=0.5, own_daily_stop_pct=4.0, per_symbol_daily_cap_pct=2.5),
    ]
    starts = ds.rolling_start_trade_indices()
    batch = simulate_challenge_batch(ds, rules, policies, starts)
    code_map = {0: "unfinished", 1: "pass", 2: "fail", 3: "restart"}

    for p_idx, pol in enumerate(policies):
        for s_idx, trade_idx in enumerate(starts):
            sc = simulate_challenge_single(ds, rules, pol, int(trade_idx))
            assert code_map[int(batch[p_idx].outcomes[s_idx])] == sc.outcome
            if sc.outcome == "pass":
                assert batch[p_idx].days_per_run[s_idx] == pytest.approx(sc.days_to_pass)
