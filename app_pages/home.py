"""PropPilot Streamlit multipage application entrypoint."""

from __future__ import annotations

import math

import streamlit as st

from proppilot.ui import format_pct, init_page_and_sidebar, render_suite_footer

ctx = init_page_and_sidebar("Overview")
dataset = ctx["dataset"]
rules = ctx["rules"]
stats = dataset.summary_stats()

st.title("PropPilot — Crypto Prop-Firm Pass Planner")
st.caption(
    "Quantify challenge pass rates, optimize position sizing with strict 60/40 validation, "
    "project 12-month funded payouts, and interrogate risks with Claude."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trades Loaded", f"{stats['n_trades']:,}")
col2.metric("Symbols", f"{stats['n_symbols']}")
col3.metric("Win Rate", format_pct(stats["win_rate"]))
col4.metric(
    "Expectancy (R)",
    f"{stats['expectancy_r']:+.3f}R" if math.isfinite(stats["expectancy_r"]) else "N/A",
)
col5.metric(
    "Trades / Month",
    f"{stats['trades_per_month']:.1f}" if math.isfinite(stats["trades_per_month"]) else "N/A",
)

st.subheader(f"Active Rulebook: {rules.name} (${rules.account_size:,.0f})")
rc1, rc2, rc3, rc4 = st.columns(4)
phases_label = (
    " -> ".join(f"+{p.target_pct:.0f}% (min {p.min_days}d)" for p in rules.phases)
    if rules.phases
    else "Instant Funded (No Eval)"
)
rc1.metric("Phases", phases_label)
rc2.metric("Daily Loss Limit", f"{rules.daily_loss_pct:.1f}%")
rc3.metric("Max Drawdown", f"{rules.max_loss_pct:.1f}% ({rules.max_loss_type})")
rc4.metric("Fee / Split", f"${rules.fee:,.0f} / {rules.profit_split:.0f}%")

if rules.approximation_notes:
    for note in rules.approximation_notes:
        st.info(note)

st.markdown("### Navigate Pages")
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(
        "**1. Simulate**\n\n"
        "Test custom risk, near-target size cuts, daily stops, and restarts across all rolling starts."
    )
    st.markdown(
        "**4. Rules Lab**\n\n"
        "Paste firm rule pages for AI YAML extraction with ambiguity flags, or edit rules manually offline."
    )
with p2:
    st.markdown(
        "**2. Optimise**\n\n"
        "Find fastest, cheapest, or max-value policies on the first 60% and validate out-of-sample on the last 40%."
    )
    st.markdown(
        "**5. AI Copilot**\n\n"
        "Ask questions via Claude tool_runner or inspect deterministic offline risk plans."
    )
with p3:
    st.markdown(
        "**3. Funded Projection**\n\n"
        "Model 12-month funded payouts, keep-buffer retention, CPPI scaling, and blowup risk."
    )
    st.markdown(
        "**6. About & Methodology**\n\n"
        "Review gambler's-ruin math, rolling starts, weekly block bootstrap, and limitations."
    )

render_suite_footer()
