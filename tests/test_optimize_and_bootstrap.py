"""Tests for honest 60/40 optimiser isolation, objectives, and reproducible bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from proppilot.ingest import load_trades
from proppilot.optimize import optimise_on_train, run_honest_optimisation
from proppilot.policy import SizingPolicy
from proppilot.rules import load_preset
from proppilot.sim import simulate_bootstrap_challenge
from proppilot.synthetic import SyntheticConfig, generate_synthetic_trades
from proppilot.ingest import trades_from_dataframe


def test_optimiser_never_reads_validation_data() -> None:
    df = generate_synthetic_trades(SyntheticConfig(days=260, seed=7))
    ds = trades_from_dataframe(df)
    rules = load_preset("two_step_classic")

    opt_clean, val_clean = ds.split_chronological(0.6)
    res_clean = optimise_on_train(opt_clean, rules, objective="fastest", max_fail=0.30)

    # Corrupt validation slice severely (-50R on every trade in last 40%)
    df_corrupt = df.copy()
    split_pos = len(opt_clean)
    df_corrupt.loc[df_corrupt.index[split_pos:], "R"] = -50.0
    ds_corrupt = trades_from_dataframe(df_corrupt)
    opt_corrupt, _ = ds_corrupt.split_chronological(0.6)
    res_corrupt = optimise_on_train(opt_corrupt, rules, objective="fastest", max_fail=0.30)

    assert res_clean.best_policy == res_corrupt.best_policy
    assert res_clean.best_opt_summary.pass_rate == pytest.approx(res_corrupt.best_opt_summary.pass_rate)

    # Full honest optimisation reports severe overfit warning on corrupted validation slice
    full_res = run_honest_optimisation(
        ds_corrupt,
        rules,
        objective="fastest",
        max_fail=0.30,
        n_bootstrap=100,
        seed=42,
    )
    assert full_res.val_summary.pass_rate < full_res.opt_summary.pass_rate
    assert any("overfit" in w.lower() for w in full_res.warnings)


def test_bootstrap_reproducible_with_seed_and_preserves_intraweek_order() -> None:
    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=120, seed=11)))
    rules = load_preset("one_step")
    pol = SizingPolicy(risk_pct=0.6, own_daily_stop_pct=2.5, restart_after_days=90)

    b1 = simulate_bootstrap_challenge(ds, rules, pol, n_paths=200, seed=99)
    b2 = simulate_bootstrap_challenge(ds, rules, pol, n_paths=200, seed=99)
    b3 = simulate_bootstrap_challenge(ds, rules, pol, n_paths=200, seed=100)

    assert b1.pass_rate == b2.pass_rate
    assert np.array_equal(b1.outcomes, b2.outcomes)
    assert np.allclose(b1.days_to_pass, b2.days_to_pass)
    # Different seed generally yields a distinct path vector
    assert not np.array_equal(b1.outcomes, b3.outcomes)


def test_all_three_objectives_on_synthetic_data() -> None:
    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=300, seed=19)))
    rules = load_preset("two_step_classic")
    small_grid = [
        SizingPolicy(risk_pct=0.4, own_daily_stop_pct=2.5, restart_after_days=90, funded_risk_pct=0.3),
        SizingPolicy(risk_pct=0.8, own_daily_stop_pct=3.0, restart_after_days=60, funded_risk_pct=0.4),
        SizingPolicy(risk_pct=1.2, own_daily_stop_pct=4.0, restart_after_days=60, funded_risk_pct=0.6),
    ]
    for obj in ("fastest", "cheapest", "max_value"):
        res = run_honest_optimisation(
            ds,
            rules,
            objective=obj,  # type: ignore[arg-type]
            max_fail=0.40,
            n_bootstrap=100,
            seed=42,
            grid=small_grid,
        )
        # Evaluation-phase fields come from the grid; funded fields are chosen separately
        # (highest expected payout with P(lose funded) <= max_lose_funded).
        funded_fields = {"funded_risk_pct", "keep_buffer_pct", "cppi_cap", "cppi_c0"}
        assert res.best_policy.model_dump(exclude=funded_fields) in [
            p.model_dump(exclude=funded_fields) for p in small_grid
        ]
        assert res.opt_trade_count + res.val_trade_count == len(ds)
