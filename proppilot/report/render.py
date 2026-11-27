"""Single-file HTML (Jinja2; Plotly.js loaded from its CDN) and JSON report renderer."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from proppilot import __version__
from proppilot.ingest import TradeDataset
from proppilot.optimize import OptimizationResult
from proppilot.report.charts import (
    plot_days_to_pass_histogram,
    plot_equity_fan_chart,
    plot_failure_breakdown,
    plot_funded_monthly_curve,
    plot_funded_payout_distribution,
    plot_pass_rate_heatmap,
)
from proppilot.sim import (
    BatchChallengeSummary,
    format_days,
    format_expected_attempts,
    format_expected_cost,
    format_pct,
    sample_challenge_fan_paths,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent


def _format_summary_row(label: str, sm: BatchChallengeSummary) -> dict[str, str]:
    return {
        "label": label,
        "n_finished": f"{sm.n_finished:,}",
        "n_unfinished": f"{sm.n_unfinished:,}",
        "pass_rate": format_pct(sm.pass_rate),
        "fail_rate": format_pct(sm.fail_rate),
        "restart_rate": format_pct(sm.restart_rate),
        "days_median": format_days(sm.days_median),
        "expected_attempts": format_expected_attempts(sm.expected_attempts),
        "expected_fees": format_expected_cost(sm.expected_fees),
    }


def render_reports(
    dataset: TradeDataset,
    result: OptimizationResult,
    explanation_text: str,
    used_ai: bool,
    out_dir: str | Path = "reports",
    stem: str = "proppilot_plan",
) -> tuple[Path, Path]:
    """Render both self-contained HTML and JSON reports to `out_dir`."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    html_path = target_dir / f"{stem}.html"
    json_path = target_dir / f"{stem}.json"

    _, val_ds = dataset.split_chronological(0.6)
    fan_paths = sample_challenge_fan_paths(val_ds, result.rules, result.best_policy, max_paths=80)

    fig_days = plot_days_to_pass_histogram(result.val_summary, title="Validation Days-to-Pass Distribution")
    fig_heatmap = plot_pass_rate_heatmap(result.grid_records)
    fig_fan = plot_equity_fan_chart(fan_paths, result.rules, title="Validation Challenge Equity Paths (Fan Chart)")
    fig_failures = plot_failure_breakdown(result.val_summary, title="Validation Non-Pass Outcomes")
    fig_funded_dist = plot_funded_payout_distribution(result.funded_summary)
    fig_funded_monthly = plot_funded_monthly_curve(result.funded_summary)

    # Include Plotly JS from CDN once in the first chart; embed subsequent chart divs cleanly.
    chart_days_html = fig_days.to_html(full_html=False, include_plotlyjs="cdn")
    chart_heatmap_html = fig_heatmap.to_html(full_html=False, include_plotlyjs=False)
    chart_fan_html = fig_fan.to_html(full_html=False, include_plotlyjs=False)
    chart_failures_html = fig_failures.to_html(full_html=False, include_plotlyjs=False)
    chart_funded_dist_html = fig_funded_dist.to_html(full_html=False, include_plotlyjs=False)
    chart_funded_monthly_html = fig_funded_monthly.to_html(full_html=False, include_plotlyjs=False)

    comparison_rows = [
        _format_summary_row(f"Optimise (First 60%, {result.opt_trade_count:,} trades)", result.opt_summary),
        _format_summary_row(f"Validate (Last 40%, {result.val_trade_count:,} trades)", result.val_summary),
        _format_summary_row(f"Bootstrap ({result.boot_summary.n_total:,} x 1y paths)", result.boot_summary),
    ]

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("template.html.j2")
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    trade_stats = dataset.summary_stats()

    html_content = template.render(
        version=__version__,
        generated_utc=generated_utc,
        rules=result.rules,
        result=result,
        trade_stats=trade_stats,
        comparison_rows=comparison_rows,
        val_cost_finite=bool(math.isfinite(result.val_summary.expected_fees)),
        chart_days=chart_days_html,
        chart_heatmap=chart_heatmap_html,
        chart_fan=chart_fan_html,
        chart_failures=chart_failures_html,
        chart_funded_dist=chart_funded_dist_html,
        chart_funded_monthly=chart_funded_monthly_html,
        explanation_text=explanation_text,
        used_ai=used_ai,
    )
    html_path.write_text(html_content, encoding="utf-8")

    payload: dict[str, Any] = result.to_dict()
    payload["generated_utc"] = generated_utc
    payload["trade_stats"] = trade_stats
    payload["used_ai"] = used_ai
    payload["explanation"] = explanation_text
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return html_path, json_path
