"""Shared Plotly chart builders used by both HTML reports and the Streamlit multipage app."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from proppilot.optimize import GridCandidateRecord
from proppilot.rules import PropRules
from proppilot.sim.models import BatchChallengeSummary, ChallengeRunResult, FundedSummary

DARK_BG = "#0B1020"
CARD_BG = "#131A2E"
TEXT_COLOR = "#E6E9F2"
GRID_COLOR = "rgba(230, 233, 242, 0.08)"
ACCENT = "#6366F1"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
CYAN = "#06B6D4"


def _apply_dark_theme(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR)),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif", size=12),
        height=height,
        margin=dict(l=48, r=24, t=48, b=44),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(19,26,46,0.7)", bordercolor=GRID_COLOR),
    )
    return fig


def plot_days_to_pass_histogram(
    summary: BatchChallengeSummary,
    title: str = "Days to Pass Distribution",
) -> go.Figure:
    """Create a days-to-pass histogram with min/median/mean/p90/max annotations."""
    fig = go.Figure()
    days = summary.days_to_pass
    if len(days) == 0:
        fig.add_annotation(
            text="No passed attempts in this sample.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=AMBER),
        )
        return _apply_dark_theme(fig, title)

    fig.add_trace(
        go.Histogram(
            x=days,
            nbinsx=min(35, max(10, int(np.sqrt(len(days))))),
            marker=dict(color=ACCENT, line=dict(color="#818CF8", width=1)),
            opacity=0.88,
            name="Passed challenges",
        )
    )

    if summary.days_median is not None and math.isfinite(summary.days_median):
        fig.add_vline(
            x=summary.days_median,
            line_dash="solid",
            line_color=GREEN,
            line_width=2,
            annotation_text=f"Median {summary.days_median:.1f}d",
            annotation_position="top right",
            annotation_font_color=GREEN,
        )
    if summary.days_mean is not None and math.isfinite(summary.days_mean):
        fig.add_vline(
            x=summary.days_mean,
            line_dash="dash",
            line_color=CYAN,
            line_width=1.8,
            annotation_text=f"Mean {summary.days_mean:.1f}d",
            annotation_position="top left",
            annotation_font_color=CYAN,
        )
    if summary.days_p90 is not None and math.isfinite(summary.days_p90):
        fig.add_vline(
            x=summary.days_p90,
            line_dash="dot",
            line_color=AMBER,
            line_width=1.8,
            annotation_text=f"P90 {summary.days_p90:.1f}d",
            annotation_position="bottom right",
            annotation_font_color=AMBER,
        )

    day_vals = (summary.days_min, summary.days_median, summary.days_mean, summary.days_p90, summary.days_max)
    if all(v is not None and math.isfinite(v) for v in day_vals):
        subtitle = (
            f"min {summary.days_min:.1f}d | median {summary.days_median:.1f}d | "
            f"mean {summary.days_mean:.1f}d | p90 {summary.days_p90:.1f}d | max {summary.days_max:.1f}d"
        )
        chart_title = f"{title}<br><sup>{subtitle}</sup>"
    else:
        chart_title = title
    _apply_dark_theme(fig, chart_title)
    fig.update_xaxes(title_text="Calendar Days to Funded")
    fig.update_yaxes(title_text="Rolling Starts / Paths")
    return fig


def plot_equity_fan_chart(
    paths: Sequence[ChallengeRunResult],
    rules: PropRules,
    title: str = "Challenge Equity Paths (Fan Chart)",
) -> go.Figure:
    """Plot sampled challenge equity paths, percentile fan bands, and firm target/loss lines."""
    fig = go.Figure()
    valid = [p for p in paths if p.curve_days is not None and p.curve_balances is not None and len(p.curve_days) > 1]
    if not valid:
        fig.add_annotation(
            text="No multi-step equity paths available for this rulebook.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig, title)

    max_day = max(float(p.curve_days[-1]) for p in valid)
    max_day = max(max_day, 5.0)
    grid_days = np.linspace(0.0, max_day, 80)

    interp_matrix = np.empty((len(valid), len(grid_days)), dtype=np.float64)
    for idx, p in enumerate(valid):
        # np.interp holds final balance flat after completion for clean percentile banding.
        interp_matrix[idx] = np.interp(grid_days, p.curve_days, p.curve_balances)

    p10 = np.percentile(interp_matrix, 10, axis=0)
    p25 = np.percentile(interp_matrix, 25, axis=0)
    p50 = np.percentile(interp_matrix, 50, axis=0)
    p75 = np.percentile(interp_matrix, 75, axis=0)
    p90 = np.percentile(interp_matrix, 90, axis=0)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid_days, grid_days[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.14)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P10-P90 band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid_days, grid_days[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.28)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P25-P75 band",
            hoverinfo="skip",
        )
    )

    for p in valid[:24]:
        color = (
            "rgba(16, 185, 129, 0.28)"
            if p.outcome == "pass"
            else ("rgba(239, 68, 68, 0.30)" if p.outcome == "fail" else "rgba(148, 163, 184, 0.22)")
        )
        fig.add_trace(
            go.Scatter(
                x=p.curve_days,
                y=p.curve_balances,
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=grid_days,
            y=p50,
            mode="lines",
            line=dict(color="#A5B4FC", width=2.5),
            name="Median equity path",
        )
    )

    init_bal = rules.account_size
    loss_floor = init_bal * (1.0 - rules.max_loss_pct / 100.0)
    fig.add_hline(
        y=loss_floor,
        line_dash="dash",
        line_color=RED,
        annotation_text=(
            f"Max Loss Floor (${loss_floor:,.0f})"
            if rules.max_loss_type == "static"
            else f"Initial Trailing Floor (${loss_floor:,.0f})"
        ),
        annotation_position="bottom right",
        annotation_font_color=RED,
    )
    for ph in rules.phases:
        tgt = init_bal * (1.0 + ph.target_pct / 100.0)
        fig.add_hline(
            y=tgt,
            line_dash="dot",
            line_color=GREEN,
            annotation_text=f"{ph.name} Target (+{ph.target_pct:.1f}% = ${tgt:,.0f})",
            annotation_position="top right",
            annotation_font_color=GREEN,
        )

    _apply_dark_theme(fig, title)
    fig.update_xaxes(title_text="Days Since Challenge Start")
    fig.update_yaxes(title_text="Account Balance (USD)", tickprefix="$")
    return fig


def plot_failure_breakdown(
    summary: BatchChallengeSummary,
    title: str = "Non-Pass Outcome Breakdown",
) -> go.Figure:
    """Bar chart of hard breaches vs policy restarts."""
    counts = dict(summary.fail_reason_counts)
    labels_map = {
        "daily_loss": "Daily Loss Breach",
        "max_loss": "Max Loss Breach",
        "max_days_timeout": "Phase Max-Days Timeout",
        "restart_timeout": "Policy Restart Cutoff",
    }
    colors_map = {
        "daily_loss": RED,
        "max_loss": "#F97316",
        "max_days_timeout": AMBER,
        "restart_timeout": CYAN,
    }

    fig = go.Figure()
    if not counts:
        fig.add_annotation(
            text="Zero failures or restarts in this run.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=GREEN, size=14),
        )
        return _apply_dark_theme(fig, title, height=320)

    keys = list(counts.keys())
    fig.add_trace(
        go.Bar(
            x=[labels_map.get(k, k) for k in keys],
            y=[counts[k] for k in keys],
            marker_color=[colors_map.get(k, ACCENT) for k in keys],
            text=[counts[k] for k in keys],
            textposition="auto",
        )
    )
    _apply_dark_theme(fig, title, height=320)
    fig.update_yaxes(title_text="Count")
    return fig


def plot_pass_rate_heatmap(
    grid_records: Sequence[GridCandidateRecord],
    title: str = "Pass-Rate Heatmap: Risk/Trade x Near-Target Multiplier",
) -> go.Figure:
    """2D heatmap of best pass rate across risk_pct x near_mult from the optimisation grid."""
    fig = go.Figure()
    if not grid_records:
        return _apply_dark_theme(fig, title)

    risks = sorted({r.policy.risk_pct for r in grid_records})
    mults = sorted({r.policy.near_mult for r in grid_records})

    z = np.full((len(mults), len(risks)), np.nan, dtype=np.float64)
    text_grid = [["" for _ in risks] for _ in mults]

    for yi, m in enumerate(mults):
        for xi, rsk in enumerate(risks):
            matching = [
                rec.summary.pass_rate
                for rec in grid_records
                if abs(rec.policy.risk_pct - rsk) < 1e-9 and abs(rec.policy.near_mult - m) < 1e-9
            ]
            if matching:
                best_pr = max(matching) * 100.0
                z[yi, xi] = best_pr
                text_grid[yi][xi] = f"{best_pr:.1f}%"

    fig.add_trace(
        go.Heatmap(
            x=[f"{r:.2g}%" for r in risks],
            y=[f"{m:.2g}x" if m < 1.0 else "1.0x (off)" for m in mults],
            z=z,
            text=text_grid,
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0.0,
            zmax=100.0,
            colorbar=dict(title="Pass %"),
        )
    )
    _apply_dark_theme(fig, title)
    fig.update_xaxes(title_text="Base Risk per Trade")
    fig.update_yaxes(title_text="Near-Target Size Multiplier")
    return fig


def plot_funded_payout_distribution(
    funded: FundedSummary,
    title: str = "12-Month Funded Monthly Payout Distribution",
) -> go.Figure:
    """Histogram of average monthly net payouts across simulated 12-month funded paths."""
    fig = go.Figure()
    vals = funded.path_monthly_avg_usd
    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=min(40, max(12, int(np.sqrt(len(vals))))),
            marker=dict(color=GREEN, line=dict(color="#34D399", width=1)),
            opacity=0.85,
            name="Monthly payout",
        )
    )
    if math.isfinite(funded.monthly_payout_mean_usd):
        fig.add_vline(
            x=funded.monthly_payout_mean_usd,
            line_color=CYAN,
            line_width=2,
            annotation_text=f"Mean ${funded.monthly_payout_mean_usd:,.0f}/mo",
            annotation_position="top right",
            annotation_font_color=CYAN,
        )
    if math.isfinite(funded.monthly_payout_median_usd):
        fig.add_vline(
            x=funded.monthly_payout_median_usd,
            line_color=AMBER,
            line_dash="dash",
            line_width=2,
            annotation_text=f"Median ${funded.monthly_payout_median_usd:,.0f}/mo",
            annotation_position="top left",
            annotation_font_color=AMBER,
        )
    p_lose_str = f"{funded.p_lose_account * 100:.1f}%" if math.isfinite(funded.p_lose_account) else "N/A"
    p10_str = f"${funded.monthly_payout_p10_usd:,.0f}" if math.isfinite(funded.monthly_payout_p10_usd) else "N/A"
    p90_str = f"${funded.monthly_payout_p90_usd:,.0f}" if math.isfinite(funded.monthly_payout_p90_usd) else "N/A"
    subtitle = (
        f"P(lose account in 12m): {p_lose_str} | "
        f"P10: {p10_str} | P90: {p90_str}"
    )
    _apply_dark_theme(fig, f"{title}<br><sup>{subtitle}</sup>")
    fig.update_xaxes(title_text="Average Monthly Net Payout (USD)", tickprefix="$")
    fig.update_yaxes(title_text="Simulated Paths")
    return fig


def plot_funded_monthly_curve(
    funded: FundedSummary,
    title: str = "Expected Net Payout by Funded Month (1-12)",
) -> go.Figure:
    """Bar chart of average payout earned in each funded month 1..12."""
    fig = go.Figure()
    months = [f"M{i}" for i in range(1, len(funded.per_month_mean_usd) + 1)]
    fig.add_trace(
        go.Bar(
            x=months,
            y=funded.per_month_mean_usd,
            marker_color=ACCENT,
            text=[f"${v:,.0f}" for v in funded.per_month_mean_usd],
            textposition="auto",
        )
    )
    _apply_dark_theme(fig, title, height=340)
    fig.update_xaxes(title_text="Funded Month")
    fig.update_yaxes(title_text="Mean Net Payout (USD)", tickprefix="$")
    return fig
