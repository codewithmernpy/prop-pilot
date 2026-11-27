"""Streamlit AppTest verification for every page and its main interactive controls."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import tomllib

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from proppilot.optimize import OptimizationResult
from proppilot.policy import SizingPolicy
from proppilot.rules import load_preset
from proppilot.sim.models import (
    FAIL_DAILY_LOSS,
    FAIL_NONE,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    FundedSummary,
    format_expected_attempts,
    format_expected_cost,
    summarize_batch_arrays,
)
from proppilot.ui import format_expected_cost as ui_format_expected_cost

ROOT = Path(__file__).resolve().parent.parent


def _page(path: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / path), default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [str(x.value) for x in at.info] + [str(x.value) for x in at.warning]
    parts += [str(x.value) for x in at.success] + [str(x.value) for x in at.error]
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_overview_page() -> None:
    at = _page("streamlit_app.py")
    assert at.title[0].value.startswith("PropPilot")
    assert {m.label for m in at.metric} >= {"Trades Loaded", "Win Rate", "Expectancy (R)", "Trades / Month"}
    assert "QuantCritic | PropPilot | SqueezeRadar | RiskGuard" in _texts(at)
    assert "](#" not in _texts(at)  # no fake anchor links in the footer
    at.selectbox(key="pp_preset_name").set_value("one_step").run()
    assert not at.exception
    assert any("One-Step" in s.value for s in at.subheader)


def test_simulate_page_sliders() -> None:
    at = _page("pages/1_Simulate.py")
    before = {m.label: m.value for m in at.metric}
    assert "Rolling Pass Rate" in before
    at.slider(key="sim_risk_pct").set_value(1.5).run()
    at.slider(key="sim_near_mult").set_value(0.5).run()
    at.slider(key="sim_own_daily").set_value(2.0).run()
    at.slider(key="sim_behind_dd").set_value(3.0).run()
    at.slider(key="sim_restart_days").set_value(60).run()
    assert not at.exception
    after = {m.label: m.value for m in at.metric}
    assert after["Rolling Pass Rate"].endswith("%")
    assert after != before  # the controls actually changed the simulation
    assert len(at.get("plotly_chart")) >= 3


def test_optimise_page_controls_and_plan_card() -> None:
    at = _page("pages/2_Optimise.py")
    assert any("Recommended Plan Card" in m.value and "FUNDED RISK" in m.value for m in at.markdown)
    assert any("Fast Interactive (" in o for o in at.selectbox(key="opt_precision").options)
    at.selectbox(key="opt_objective").set_value("max_value").run()
    at.slider(key="opt_max_fail").set_value(0.5).run()
    at.slider(key="opt_max_lose_funded").set_value(0.05).run()
    assert not at.exception
    card = next(m.value for m in at.markdown if "Recommended Plan Card" in m.value)
    assert "5% limit" in card
    assert len(at.dataframe) >= 1


def test_funded_page_controls() -> None:
    at = _page("pages/3_Funded_Projection.py")
    base = {m.label: m.value for m in at.metric}
    assert "P(Lose Account in 12m)" in base
    at.slider(key="fn_keep_buf").set_value(6.0).run()
    at.checkbox(key="fn_cppi_on").uncheck().run()
    at.slider(key="fn_risk").set_value(1.5).run()
    assert not at.exception
    after = {m.label: m.value for m in at.metric}
    assert after["Mean Monthly Payout"] != base["Mean Monthly Payout"]


def test_rules_lab_offline_extraction_and_manual_form() -> None:
    at = _page("pages/4_Rules_Lab.py")
    at.button(key="rl_extract_btn").click().run()
    assert not at.exception
    assert any("offline keyword parser" in str(w.value) for w in at.warning)
    assert any("Phase 1" in c.value for c in at.code)
    at.button(key="rl_apply_ai_btn").click().run()
    assert not at.exception and not at.error

    at.number_input(key="rl_daily_loss").set_value(4.0).run()
    at.selectbox(key="rl_max_type").set_value("trailing_closed").run()
    at.checkbox(key="rl_trail_lock").check().run()
    assert "trailing_lock_at_initial: true" in at.text_area(key="rl_yaml_editor").value
    at.button(key="rl_apply_manual_btn").click().run()
    assert not at.exception and not at.error
    assert any("saved to session" in str(s.value) for s in at.success)
    # The next run switches the sidebar to the custom rulebook.
    at.run()
    assert not at.exception
    assert at.radio(key="pp_rules_mode").value == "Custom (Rules Lab)"


def test_ai_copilot_offline() -> None:
    at = _page("pages/5_AI_Copilot.py")
    assert "deterministic offline mode" in _texts(at)
    assert "Recommended Pass Plan" in _texts(at)
    at.button(key="cp_q_fail").click().run()
    assert not at.exception
    assert len(at.chat_message) >= 2
    at.chat_input[0].set_value("What is my funded risk?").run()
    assert not at.exception
    assert len(at.chat_message) >= 4
    assert "P(lose funded account within 12m)" in at.chat_message[-1].markdown[0].value


def test_about_page() -> None:
    at = _page("pages/6_About_Methodology.py")
    assert at.title[0].value.startswith("6.")
    assert "CPPI" in _texts(at)


def test_entrypoint_navigation_and_titles() -> None:
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["toolbarMode"] == "viewer"

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.startswith("PropPilot")

    expected_titles = [
        "Home",
        "Simulate",
        "Optimise",
        "Funded Projection",
        "Rules Lab",
        "AI Copilot",
        "About & Methodology",
    ]
    assert [p["page_name"] for p in at._registered_pages.values()] == expected_titles
    assert at._registered_pages[at._finished_page_script_hash]["page_name"] == "Home"
    assert any("stMetricLabel" in m.value and "stMetricValue" in m.value for m in at.markdown)

    for page_path in (
        "pages/1_Simulate.py",
        "pages/2_Optimise.py",
        "pages/3_Funded_Projection.py",
        "pages/4_Rules_Lab.py",
        "pages/5_AI_Copilot.py",
        "pages/6_About_Methodology.py",
    ):
        at.switch_page(page_path).run()
        assert not at.exception
        assert any('stMetricLabel' in m.value and 'stMetricValue' in m.value for m in at.markdown)


def _csv_bytes_for_trades(r_values: list[float], symbol: str = "TESTBTC") -> bytes:
    lines = ["entry_time,exit_time,symbol,side,R"]
    base = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    for idx, r_val in enumerate(r_values):
        entry = base + timedelta(days=idx)
        exit_ = entry + timedelta(hours=4)
        lines.append(
            f"{entry.strftime('%Y-%m-%dT%H:%M:%SZ')},"
            f"{exit_.strftime('%Y-%m-%dT%H:%M:%SZ')},"
            f"{symbol},LONG,{r_val}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_cross_page_sidebar_state_and_upload_persistence() -> None:
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception

    # Select Upload mode, change shared sidebar controls, and seed uploaded CSV into store keys.
    csv_bytes = _csv_bytes_for_trades([1.2] * 25, symbol="SOLTEST")
    at.radio(key="pp_trades_mode").set_value("Upload CSV / JSON").run()
    at.session_state["pp_uploaded_bytes"] = csv_bytes
    at.session_state["pp_uploaded_ext"] = "csv"
    at.session_state["pp_uploaded_key"] = hashlib.sha256(csv_bytes).hexdigest()[:16]
    at.session_state["pp_uploaded_name"] = "custom_strategy.csv"
    at.selectbox(key="pp_account_size").set_value(50_000).run()
    at.selectbox(key="pp_preset_name").set_value("one_step").run()
    at.text_input(key="pp_api_key").set_value("sk-test-session").run()
    assert not at.exception

    metrics_home = {m.label: m.value for m in at.metric}
    assert metrics_home["Trades Loaded"] == "25"
    assert metrics_home["Symbols"] == "1"
    assert any("Using uploaded file: custom_strategy.csv (25 trades)" in c.value for c in at.caption)

    # Switch to Simulate: all shared sidebar choices and stored uploaded trades must persist.
    at.switch_page("pages/1_Simulate.py").run()
    assert not at.exception
    assert at.radio(key="pp_trades_mode").value == "Upload CSV / JSON"
    assert at.selectbox(key="pp_account_size").value == 50_000
    assert at.selectbox(key="pp_preset_name").value == "one_step"
    assert at.text_input(key="pp_api_key").value == "sk-test-session"
    assert any("Using uploaded file: custom_strategy.csv (25 trades)" in c.value for c in at.caption)
    sim_metrics = {m.label: m.value for m in at.metric}
    finished, unfinished = (int(x.strip()) for x in sim_metrics["Finished / Unfinished"].split("/"))
    assert finished + unfinished <= 25

    # Switch to Optimise and back to Home: state remains intact across multiple transitions.
    at.switch_page("pages/2_Optimise.py").run()
    assert not at.exception
    assert at.radio(key="pp_trades_mode").value == "Upload CSV / JSON"
    assert at.selectbox(key="pp_account_size").value == 50_000
    assert at.selectbox(key="pp_preset_name").value == "one_step"

    at.switch_page("streamlit_app.py").run()
    assert not at.exception
    assert at.radio(key="pp_trades_mode").value == "Upload CSV / JSON"
    assert {m.label: m.value for m in at.metric}["Trades Loaded"] == "25"

    # Clear upload removes stored upload keys and falls back to sample dataset.
    at.button(key="pp_clear_trades_upload").click().run()
    assert not at.exception
    assert "pp_uploaded_bytes" not in at.session_state
    assert any("No file uploaded yet" in str(i.value) for i in at.info)
    assert int({m.label: m.value for m in at.metric}["Trades Loaded"].replace(",", "")) > 100

    # Verify Rules Source = Upload YAML also persists across page switches.
    custom_yaml = load_preset("instant", account_size=50_000).to_yaml()
    at.radio(key="pp_rules_mode").set_value("Upload YAML").run()
    at.session_state["pp_uploaded_yaml_text"] = custom_yaml
    at.session_state["pp_uploaded_yaml_name"] = "my_rules.yaml"
    at.run()
    assert any("Using uploaded rules: my_rules.yaml" in c.value for c in at.caption)
    at.switch_page("pages/3_Funded_Projection.py").run()
    assert not at.exception
    assert at.radio(key="pp_rules_mode").value == "Upload YAML"
    assert any("Using uploaded rules: my_rules.yaml" in c.value for c in at.caption)


def test_expected_cost_formatter_and_zero_pass_rate_ui() -> None:
    assert ui_format_expected_cost is format_expected_cost
    assert format_expected_cost(float("inf")) == "N/A (0% pass rate)"
    assert format_expected_cost(float("nan")) == "N/A (0% pass rate)"
    assert format_expected_cost(1234.56) == "$1,235"
    assert format_expected_cost(0.0) == "$0"
    assert format_expected_attempts(float("inf")) == "N/A (0% pass rate)"
    assert format_expected_attempts(2.5) == "2.50"

    # Zero-pass summary formats without raw "inf" in plan card and Simulate UI.
    zero_sm = summarize_batch_arrays(
        outcomes=np.array([OUTCOME_FAIL, OUTCOME_FAIL], dtype=np.int8),
        fail_reasons=np.array([FAIL_DAILY_LOSS, FAIL_DAILY_LOSS], dtype=np.int8),
        days_per_run=np.array([np.nan, np.nan], dtype=np.float64),
        fee=500.0,
    )
    pass_sm = summarize_batch_arrays(
        outcomes=np.array([OUTCOME_PASS, OUTCOME_PASS], dtype=np.int8),
        fail_reasons=np.array([FAIL_NONE, FAIL_NONE], dtype=np.int8),
        days_per_run=np.array([10.0, 12.0], dtype=np.float64),
        fee=500.0,
    )
    fake_funded = FundedSummary(
        n_paths=10,
        account_size=100_000.0,
        p_lose_account=0.05,
        fail_reason_counts={},
        monthly_payout_mean_usd=1000.0,
        monthly_payout_median_usd=1000.0,
        monthly_payout_p10_usd=800.0,
        monthly_payout_p25_usd=900.0,
        monthly_payout_p75_usd=1100.0,
        monthly_payout_p90_usd=1200.0,
        monthly_payout_mean_pct=1.0,
        monthly_payout_median_pct=1.0,
        total_12m_payout_mean_usd=12_000.0,
        total_12m_payout_median_usd=12_000.0,
        prob_zero_payout=0.0,
        per_month_mean_usd=np.full(12, 1000.0),
        path_monthly_avg_usd=np.full(10, 1000.0),
        path_12m_total_usd=np.full(10, 12_000.0),
    )
    opt_res = OptimizationResult(
        objective="fastest",
        max_fail=0.20,
        rules=load_preset("two_step_classic"),
        best_policy=SizingPolicy(),
        opt_summary=pass_sm,
        val_summary=zero_sm,
        boot_summary=pass_sm,
        funded_summary=fake_funded,
        failure_attribution=None,  # type: ignore[arg-type]
        grid_records=[],
        opt_trade_count=10,
        val_trade_count=10,
        feasible_found=True,
    )
    card_summary = opt_res.plan_card_summary()
    assert "N/A (0% pass rate)" in card_summary
    assert re.search(r"\binf\b", card_summary, flags=re.IGNORECASE) is None

    # End-to-end Simulate page with an all-losing dataset (0% pass rate) shows N/A (0% pass rate).
    losing_csv = _csv_bytes_for_trades([-10.0] * 10, symbol="LOSEBTC")
    at = AppTest.from_file(str(ROOT / "pages/1_Simulate.py"), default_timeout=120).run()
    at.radio(key="pp_trades_mode").set_value("Upload CSV / JSON").run()
    at.session_state["pp_uploaded_bytes"] = losing_csv
    at.session_state["pp_uploaded_ext"] = "csv"
    at.session_state["pp_uploaded_key"] = hashlib.sha256(losing_csv).hexdigest()[:16]
    at.session_state["pp_uploaded_name"] = "losing.csv"
    at.run()
    assert not at.exception
    sim_metrics = {m.label: m.value for m in at.metric}
    assert sim_metrics["Expected Cost"] == "N/A (0% pass rate)"


def test_optimise_plan_card_contains_funded_risk_once() -> None:
    at = AppTest.from_file(str(ROOT / "pages/2_Optimise.py"), default_timeout=120).run()
    assert not at.exception
    card = next(m.value for m in at.markdown if "Recommended Plan Card" in m.value)
    assert card.count("FUNDED RISK") == 1
    assert card.count("P(lose funded account within 12m)") == 1
    assert re.search(r"\binf\b", card, flags=re.IGNORECASE) is None
