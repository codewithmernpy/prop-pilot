"""Page 1: Interactive Challenge Simulator."""

from __future__ import annotations

import math

import streamlit as st

from proppilot.policy import SizingPolicy, default_policy_for_rules
from proppilot.report.charts import (
    plot_days_to_pass_histogram,
    plot_equity_fan_chart,
    plot_failure_breakdown,
)
from proppilot.ui import (
    cached_simulate_challenge,
    format_days,
    format_expected_cost,
    format_pct,
    init_page_and_sidebar,
    render_suite_footer,
)

ctx = init_page_and_sidebar("Simulate")
rules = ctx["rules"]
defaults = default_policy_for_rules(rules)

st.title("1. Challenge Policy Simulator")
st.caption("Configure position sizing and daily risk guards, then inspect rolling starts and bootstrap paths.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    risk_pct = st.slider(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=float(min(3.0, max(1.5, rules.daily_loss_pct))),
        value=float(defaults.risk_pct),
        step=0.05,
        key="sim_risk_pct",
    )
    near_frac = st.slider(
        "Near-target threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        key="sim_near_frac",
    )
with c2:
    near_mult = st.slider(
        "Near-target size multiplier",
        min_value=0.20,
        max_value=1.00,
        value=float(defaults.near_mult),
        step=0.05,
        key="sim_near_mult",
    )
    own_daily_stop = st.slider(
        "Own daily stop (%)",
        min_value=0.5,
        max_value=float(rules.daily_loss_pct),
        value=float(min(defaults.own_daily_stop_pct, rules.daily_loss_pct)),
        step=0.25,
        key="sim_own_daily",
    )
with c3:
    max_open_risk = st.slider(
        "Max concurrent open risk (%)",
        min_value=0.5,
        max_value=float(max(1.0, rules.daily_loss_pct * 1.5)),
        value=float(defaults.max_open_risk_pct or defaults.own_daily_stop_pct),
        step=0.25,
        key="sim_max_open",
    )
    per_sym_cap = st.slider(
        "Per-symbol daily cap (%, 0=off)",
        min_value=0.0,
        max_value=float(rules.daily_loss_pct),
        value=0.0,
        step=0.25,
        key="sim_sym_cap",
    )
with c4:
    restart_days = st.slider(
        "Restart unfinished after (days, 0=off)",
        min_value=0,
        max_value=180,
        value=int(defaults.restart_after_days or 0),
        step=15,
        key="sim_restart_days",
    )
    behind_dd = st.slider(
        "Behind mode: drawdown trigger (%, 0=off)",
        min_value=0.0,
        max_value=float(rules.max_loss_pct),
        value=0.0,
        step=0.5,
        key="sim_behind_dd",
    )
    behind_mult = st.slider(
        "Behind mode: size multiplier",
        min_value=0.20,
        max_value=1.00,
        value=0.50,
        step=0.05,
        key="sim_behind_mult",
    )
    n_boot = st.slider(
        "Bootstrap paths",
        min_value=200,
        max_value=2000,
        value=300,
        step=100,
        key="sim_n_boot",
    )

policy = SizingPolicy(
    risk_pct=risk_pct,
    near_frac=near_frac if near_mult < 1.0 else None,
    near_mult=near_mult,
    behind_dd_pct=behind_dd if behind_dd > 0 else None,
    behind_mult=behind_mult,
    own_daily_stop_pct=own_daily_stop,
    per_symbol_daily_cap_pct=per_sym_cap if per_sym_cap > 0 else None,
    max_open_risk_pct=max_open_risk,
    restart_after_days=restart_days if restart_days > 0 else None,
    funded_risk_pct=defaults.funded_risk_pct,
)

rolling, boot, fan_paths, failures = cached_simulate_challenge(
    ctx["source_key"],
    ctx["raw_bytes"],
    ctx["file_ext"],
    ctx["rules_yaml"],
    policy.to_json(),
    n_bootstrap=n_boot,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rolling Pass Rate", format_pct(rolling.pass_rate))
m2.metric("Bootstrap Pass Rate", format_pct(boot.pass_rate))
m3.metric("Breach / Restart", f"{format_pct(rolling.fail_rate)} / {format_pct(rolling.restart_rate)}")
m4.metric("Median Days", format_days(rolling.days_median))
m5.metric("Expected Cost", format_expected_cost(rolling.expected_fees))
m6.metric("Finished / Unfinished", f"{rolling.n_finished} / {rolling.n_unfinished}")

if rolling.days_median is not None and all(
    v is not None and math.isfinite(v)
    for v in (rolling.days_min, rolling.days_median, rolling.days_mean, rolling.days_p90, rolling.days_max)
):
    st.info(
        f"Days to pass (rolling starts): min **{rolling.days_min:.1f}d**, "
        f"median **{rolling.days_median:.1f}d**, mean **{rolling.days_mean:.1f}d**, "
        f"p90 **{rolling.days_p90:.1f}d**, max **{rolling.days_max:.1f}d**."
    )

left, right = st.columns(2)
with left:
    st.plotly_chart(plot_days_to_pass_histogram(rolling, title="Rolling Days to Pass"), width="stretch")
with right:
    st.plotly_chart(plot_failure_breakdown(rolling, title="Failure & Restart Reasons"), width="stretch")

st.plotly_chart(plot_equity_fan_chart(fan_paths, rules), width="stretch")

if failures.top_breach_days:
    with st.expander("Top Correlated Breach Days", expanded=False):
        st.dataframe(failures.top_breach_days, width="stretch", hide_index=True)

render_suite_footer()
