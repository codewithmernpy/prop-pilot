"""Page 2: Honest 60/40 Policy Optimiser."""

from __future__ import annotations

import streamlit as st

from proppilot.report.charts import (
    plot_days_to_pass_histogram,
    plot_pass_rate_heatmap,
)
from proppilot.ui import (
    cached_optimise,
    format_opt_comparison_rows,
    grid_sizes,
    init_page_and_sidebar,
    render_plan_card,
    render_suite_footer,
)

ctx = init_page_and_sidebar("Optimise")

st.title("2. Honest Policy Optimiser (60/40 Split)")
st.caption(
    "Optimises strictly on the first 60% of trades, then validates out-of-sample on the unseen last 40% "
    "and runs weekly block bootstrap."
)

fast_n, full_n = grid_sizes(ctx["rules"])
c1, c2, c3, c4 = st.columns(4)
with c1:
    objective = st.selectbox(
        "Objective",
        options=["fastest", "cheapest", "max_value"],
        index=0,
        key="opt_objective",
        format_func=lambda x: {
            "fastest": "fastest — minimise median days to pass",
            "cheapest": "cheapest — minimise expected challenge fees",
            "max_value": "max_value — maximise expected 12m net value (payouts - fees incl. re-buys)",
        }[x],
    )
with c2:
    max_fail = st.slider(
        "Max non-pass probability constraint",
        min_value=0.05,
        max_value=0.60,
        value=0.20,
        step=0.05,
        key="opt_max_fail",
    )
with c3:
    max_lose_funded = st.slider(
        "Max P(lose funded account in 12m)",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01,
        key="opt_max_lose_funded",
    )
with c4:
    precision_mode = st.selectbox(
        "Search precision",
        options=[
            f"Fast Interactive ({fast_n} policies, 250 MC)",
            f"Full Precision ({full_n} policies, 2000 MC)",
        ],
        index=0,
        key="opt_precision",
    )

full_precision = precision_mode.startswith("Full")
res = cached_optimise(
    ctx["source_key"],
    ctx["raw_bytes"],
    ctx["file_ext"],
    ctx["rules_yaml"],
    objective=objective,  # type: ignore[arg-type]
    max_fail=max_fail,
    full_precision=full_precision,
    max_lose_funded=max_lose_funded,
)

# Save latest policy in session for Copilot / Funded pages
st.session_state["pp_recommended_policy_json"] = res.best_policy.to_json()

render_plan_card(res)

if res.warnings:
    for w in res.warnings:
        st.warning(w)

st.subheader("Optimise (First 60%) vs Validate (Last 40%) vs Block Bootstrap")
st.dataframe(format_opt_comparison_rows(res), width="stretch", hide_index=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(plot_pass_rate_heatmap(res.grid_records), width="stretch")
with right:
    st.plotly_chart(
        plot_days_to_pass_histogram(res.val_summary, title="Out-of-Sample Validation Days to Pass"),
        width="stretch",
    )

with st.expander("Funded Variants Evaluated (Optimisation Split)", expanded=False):
    st.dataframe([c.to_dict() for c in res.funded_candidates], width="stretch", hide_index=True)

with st.expander("Top 15 Grid Policies (Optimisation Split)", expanded=False):
    top_rows = sorted(
        [r.to_dict() for r in res.grid_records],
        key=lambda x: (-x["pass_rate"], x["days_median"] or 9999.0),
    )[:15]
    st.dataframe(top_rows, width="stretch", hide_index=True)

render_suite_footer()
