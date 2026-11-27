"""Page 3: 12-Month Funded Projection & CPPI Sizing."""

from __future__ import annotations

import math

import streamlit as st

from proppilot.policy import SizingPolicy, default_policy_for_rules
from proppilot.report.charts import plot_funded_monthly_curve, plot_funded_payout_distribution
from proppilot.ui import (
    cached_funded_projection,
    format_pct,
    init_page_and_sidebar,
    render_suite_footer,
)

ctx = init_page_and_sidebar("Funded Projection")
rules = ctx["rules"]

base_policy = (
    SizingPolicy.from_json(st.session_state["pp_recommended_policy_json"])
    if "pp_recommended_policy_json" in st.session_state
    else default_policy_for_rules(rules)
)

st.title("3. Funded Phase Projection (12 Months)")
st.caption(
    "Simulate 12-month funded payouts, kept profit buffer, consistency gating, and CPPI cushion scaling."
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    funded_risk = st.slider(
        "Funded base risk (%)",
        min_value=0.10,
        max_value=2.00,
        value=float(min(2.0, base_policy.funded_risk_pct)),
        step=0.05,
        key="fn_risk",
    )
with c2:
    keep_buffer = st.slider(
        "Keep buffer (% above initial)",
        min_value=0.0,
        max_value=15.0,
        value=float(base_policy.keep_buffer_pct),
        step=0.5,
        key="fn_keep_buf",
    )
with c3:
    cppi_on = st.checkbox(
        "CPPI sizing",
        value=base_policy.cppi_c0 is not None,
        key="fn_cppi_on",
        help="Risk multiplier = min(cap, cushion above the max-loss floor / c0).",
    )
    cppi_cap = st.slider(
        "CPPI multiplier cap",
        min_value=1.0,
        max_value=3.0,
        value=float(base_policy.cppi_cap),
        step=0.1,
        key="fn_cppi_cap",
        disabled=not cppi_on,
    )
with c4:
    cppi_c0 = st.slider(
        "CPPI c0: cushion % for 1x risk",
        min_value=1.0,
        max_value=30.0,
        value=float(base_policy.cppi_c0 if base_policy.cppi_c0 is not None else rules.max_loss_pct),
        step=0.5,
        key="fn_cppi_c0",
        disabled=not cppi_on,
    )
with c5:
    n_paths = st.slider(
        "Bootstrap 1y paths",
        min_value=200,
        max_value=2000,
        value=600,
        step=200,
        key="fn_n_paths",
    )
with c6:
    st.metric("Max-loss floor", f"{rules.max_loss_pct:.1f}% {rules.max_loss_type}")
    if rules.max_loss_type != "static":
        st.caption("Trailing floor locks at initial: " + ("yes" if rules.trailing_lock_at_initial else "no"))

funded_policy = base_policy.model_copy(
    update={
        "funded_risk_pct": funded_risk,
        "keep_buffer_pct": keep_buffer,
        "cppi_cap": cppi_cap,
        "cppi_c0": cppi_c0 if cppi_on else None,
    }
)

funded = cached_funded_projection(
    ctx["source_key"],
    ctx["raw_bytes"],
    ctx["file_ext"],
    ctx["rules_yaml"],
    funded_policy.to_json(),
    n_paths=n_paths,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("P(Lose Account in 12m)", format_pct(funded.p_lose_account))
if math.isfinite(funded.p_lose_account) and funded.p_lose_account > 0.15:
    st.error(
        f"High funded risk: {funded.p_lose_account * 100:.1f}% of simulated accounts are lost within 12 months "
        "(PropPilot's optimiser rejects funded plans above 15% by default)."
    )
m2.metric(
    "Mean Monthly Payout",
    f"${funded.monthly_payout_mean_usd:,.0f}" if math.isfinite(funded.monthly_payout_mean_usd) else "N/A",
)
m3.metric(
    "Median Monthly Payout",
    f"${funded.monthly_payout_median_usd:,.0f}" if math.isfinite(funded.monthly_payout_median_usd) else "N/A",
)
m4.metric(
    "Monthly Yield (Mean)",
    f"{funded.monthly_payout_mean_pct:.2f}%/mo" if math.isfinite(funded.monthly_payout_mean_pct) else "N/A",
)
m5.metric(
    "Mean 12m Total",
    f"${funded.total_12m_payout_mean_usd:,.0f}" if math.isfinite(funded.total_12m_payout_mean_usd) else "N/A",
)

left, right = st.columns(2)
with left:
    st.plotly_chart(plot_funded_payout_distribution(funded), width="stretch")
with right:
    st.plotly_chart(plot_funded_monthly_curve(funded), width="stretch")

if rules.consistency_pct is not None:
    st.info(
        f"Consistency rule active: single best day <= {rules.consistency_pct:.1f}% of cycle profit before payout."
    )

render_suite_footer()
