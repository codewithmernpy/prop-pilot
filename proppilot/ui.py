"""Shared Streamlit UI state, cached simulation wrappers, sidebar controls, and suite theme."""

from __future__ import annotations

import hashlib
import html
import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from proppilot.ingest import TradeDataset, load_trades
from proppilot.optimize import (
    DEFAULT_MAX_LOSE_FUNDED,
    ObjectiveName,
    OptimizationResult,
    run_honest_optimisation,
)
from proppilot.policy import SizingPolicy, build_default_grid
from proppilot.rules import PropRules, list_presets, load_preset, parse_rules_yaml_text
from proppilot.sim import (
    BatchChallengeSummary,
    ChallengeRunResult,
    FailureAttribution,
    FundedSummary,
    analyze_failures,
    format_days,
    format_expected_attempts,
    format_expected_cost,
    format_pct,
    sample_challenge_fan_paths,
    simulate_bootstrap_challenge,
    simulate_funded_bootstrap,
    simulate_rolling_starts,
)
from proppilot.synthetic import write_sample_trades

SUITE_FOOTER_MD = "Part of the Crypto Quant Suite: QuantCritic | PropPilot | SqueezeRadar | RiskGuard"

_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "trades_sample.csv"

_TRADES_MODE_OPTIONS = ("Sample (4y Synthetic Crypto)", "Upload CSV / JSON")
_TRADES_MODE_DEFAULT = _TRADES_MODE_OPTIONS[0]
_ACCOUNT_SIZE_OPTIONS = (10_000, 25_000, 50_000, 100_000, 200_000)
_ACCOUNT_SIZE_DEFAULT = 100_000
_RULES_MODE_DEFAULT = "Built-in Preset"
_PRESET_DEFAULT = "two_step_classic"

_CUSTOM_CSS = """
<style>
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 2.2vw, 1.8rem) !important;
}
.pp-plan-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.20), rgba(16, 185, 129, 0.12));
    border: 1px solid rgba(99, 102, 241, 0.45);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0 18px 0;
}
.pp-plan-card-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #A5B4FC;
    font-weight: 700;
    margin-bottom: 6px;
}
.pp-plan-risk {
    font-size: 1.0rem;
    font-weight: 700;
    border: 1px solid;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 10px;
}
.pp-plan-card-body {
    font-size: 1.08rem;
    font-weight: 600;
    color: #E6E9F2;
    line-height: 1.5;
}
.pp-footer {
    margin-top: 28px;
    padding-top: 12px;
    border-top: 1px solid rgba(230, 233, 242, 0.12);
    color: #9AA4BF;
    font-size: 0.85rem;
}
</style>
"""


def _ensure_sample_exists() -> Path:
    if _SAMPLE_PATH.exists():
        return _SAMPLE_PATH
    return write_sample_trades(_SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def load_cached_trades(source_key: str, raw_bytes: bytes | None, file_ext: str) -> TradeDataset:
    """Load and cache TradeDataset from sample CSV or uploaded bytes."""
    if source_key == "sample" or raw_bytes is None:
        return load_trades(_ensure_sample_exists())
    buf = io.BytesIO(raw_bytes)
    buf.name = f"uploaded.{file_ext}"
    return load_trades(buf, fmt=file_ext)


@st.cache_data(show_spinner=False)
def cached_simulate_challenge(
    source_key: str,
    raw_bytes: bytes | None,
    file_ext: str,
    rules_yaml: str,
    policy_json: str,
    n_bootstrap: int = 300,
) -> tuple[BatchChallengeSummary, BatchChallengeSummary, list[ChallengeRunResult], FailureAttribution]:
    """Cached single-policy evaluation (rolling starts, bootstrap, fan curves, failures)."""
    ds = load_cached_trades(source_key, raw_bytes, file_ext)
    rules = parse_rules_yaml_text(rules_yaml)
    pol = SizingPolicy.from_json(policy_json)
    rolling = simulate_rolling_starts(ds, rules, pol)
    boot = simulate_bootstrap_challenge(ds, rules, pol, n_paths=n_bootstrap, seed=42)
    fan_paths = sample_challenge_fan_paths(ds, rules, pol, max_paths=35)
    failures = analyze_failures(ds, rules, pol, max_detail_runs=35, batch_summary=rolling)
    return rolling, boot, fan_paths, failures


def _build_interactive_fast_grid(rules: PropRules) -> list[SizingPolicy]:
    """Compact ~20-policy grid for sub-4s cold interactive Streamlit runs."""
    risk_candidates = [0.3, 0.5, 0.7, 1.0, 1.25]
    if rules.max_risk_per_trade_pct is not None:
        risk_candidates = [r for r in risk_candidates if r <= rules.max_risk_per_trade_pct + 1e-9]
        if not risk_candidates:
            risk_candidates = [rules.max_risk_per_trade_pct]

    near_mults = [0.5, 1.0]
    dlim = rules.daily_loss_pct
    daily_stops = sorted(
        {
            round(max(0.4, dlim * 0.5), 2),
            round(max(0.5, min(dlim - 0.25, dlim * 0.75)), 2),
        }
    )
    phase_max_days = [p.max_days for p in rules.phases if p.max_days is not None]
    if phase_max_days:
        min_md = min(phase_max_days)
        restart_default: int | None = max(15, min_md // 2)
    elif not rules.phases:
        restart_default = None
    else:
        restart_default = 90

    grid: list[SizingPolicy] = []
    for risk in risk_candidates:
        for nm in near_mults:
            for ds_pct in daily_stops:
                if risk > ds_pct + 1e-9:
                    continue
                f_risk = round(max(0.2, min(risk * 0.45, dlim * 0.08)), 2)
                if rules.max_risk_per_trade_pct is not None:
                    f_risk = min(f_risk, rules.max_risk_per_trade_pct)
                grid.append(
                    SizingPolicy(
                        risk_pct=risk,
                        near_frac=0.75 if nm < 1.0 else None,
                        near_mult=nm,
                        own_daily_stop_pct=ds_pct,
                        max_open_risk_pct=ds_pct,
                        restart_after_days=restart_default,
                        funded_risk_pct=f_risk,
                        keep_buffer_pct=4.0,
                        cppi_cap=1.5,
                        cppi_c0=float(rules.max_loss_pct),
                    )
                )
    return grid


@st.cache_data(show_spinner=False)
def cached_optimise(
    source_key: str,
    raw_bytes: bytes | None,
    file_ext: str,
    rules_yaml: str,
    objective: ObjectiveName,
    max_fail: float,
    full_precision: bool = False,
    max_lose_funded: float = DEFAULT_MAX_LOSE_FUNDED,
) -> OptimizationResult:
    """Cached 60/40 optimisation + validation + bootstrap + funded bundle."""
    ds = load_cached_trades(source_key, raw_bytes, file_ext)
    rules = parse_rules_yaml_text(rules_yaml)
    grid = build_default_grid(rules) if full_precision else _build_interactive_fast_grid(rules)
    n_boot = 2000 if full_precision else 250
    return run_honest_optimisation(
        dataset=ds,
        rules=rules,
        objective=objective,
        max_fail=max_fail,
        n_bootstrap=n_boot,
        seed=42,
        grid=grid,
        max_lose_funded=max_lose_funded,
        funded_paths=300 if full_precision else 120,
    )


def grid_sizes(rules: PropRules) -> tuple[int, int]:
    """Return (fast interactive grid size, full grid size) for labels."""
    return len(_build_interactive_fast_grid(rules)), len(build_default_grid(rules))


def render_plan_card(res: OptimizationResult) -> None:
    """Render the plan card with the funded-account risk shown prominently."""
    risk_colour = "#10B981" if res.funded_risk_ok else "#EF4444"
    st.markdown(
        f"""
        <div class="pp-plan-card">
          <div class="pp-plan-card-title">Recommended Plan Card</div>
          <div class="pp-plan-risk" style="border-color:{risk_colour};color:{risk_colour};">
            FUNDED RISK &mdash; {html.escape(res.funded_risk_line())}
          </div>
          <div class="pp-plan-card-body">{html.escape(res.plan_card_summary(include_funded_risk=False))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_funded_projection(
    source_key: str,
    raw_bytes: bytes | None,
    file_ext: str,
    rules_yaml: str,
    policy_json: str,
    n_paths: int = 600,
) -> FundedSummary:
    """Cached 12-month funded account Monte Carlo projection."""
    ds = load_cached_trades(source_key, raw_bytes, file_ext)
    rules = parse_rules_yaml_text(rules_yaml)
    pol = SizingPolicy.from_json(policy_json)
    return simulate_funded_bootstrap(ds, rules, pol, n_paths=n_paths, seed=123)


def _sync_widget_store(widget_key: str, store_key: str) -> None:
    """Copy a widget's session_state value into its persistent non-widget store key."""
    if widget_key in st.session_state:
        st.session_state[store_key] = st.session_state[widget_key]


def _restore_widget_from_store(
    widget_key: str,
    store_key: str,
    default: Any,
    valid_options: Sequence[Any] | None = None,
) -> None:
    """Hydrate a widget key from its persistent non-widget store key before rendering."""
    val = st.session_state.get(store_key, default)
    if valid_options is not None and val not in valid_options:
        val = default
    st.session_state[store_key] = val
    st.session_state[widget_key] = val


def _store_uploaded_trades(uploaded_file: Any) -> None:
    """Persist uploaded trade bytes and metadata into non-widget session_state keys."""
    raw_bytes = uploaded_file.getvalue()
    file_ext = Path(uploaded_file.name).suffix.lstrip(".").lower() or "csv"
    source_key = hashlib.sha256(raw_bytes).hexdigest()[:16]
    st.session_state["pp_uploaded_bytes"] = raw_bytes
    st.session_state["pp_uploaded_ext"] = file_ext
    st.session_state["pp_uploaded_key"] = source_key
    st.session_state["pp_uploaded_name"] = uploaded_file.name


def _on_trades_upload_change() -> None:
    uploaded_file = st.session_state.get("pp_trades_uploader")
    if uploaded_file is not None:
        _store_uploaded_trades(uploaded_file)


def _clear_uploaded_trades() -> None:
    for key in ("pp_uploaded_bytes", "pp_uploaded_ext", "pp_uploaded_key", "pp_uploaded_name"):
        st.session_state.pop(key, None)
    st.session_state.pop("pp_trades_uploader", None)


def _store_uploaded_yaml(uploaded_file: Any) -> None:
    """Persist uploaded rules YAML text and filename into non-widget session_state keys."""
    st.session_state["pp_uploaded_yaml_text"] = uploaded_file.getvalue().decode("utf-8")
    st.session_state["pp_uploaded_yaml_name"] = uploaded_file.name


def _on_yaml_upload_change() -> None:
    uploaded_file = st.session_state.get("pp_yaml_up")
    if uploaded_file is not None:
        _store_uploaded_yaml(uploaded_file)


def _clear_uploaded_yaml() -> None:
    for key in ("pp_uploaded_yaml_text", "pp_uploaded_yaml_name"):
        st.session_state.pop(key, None)
    st.session_state.pop("pp_yaml_up", None)


def init_page_and_sidebar(page_title: str) -> dict[str, Any]:
    """Configure wide page layout, render shared Crypto Quant Suite sidebar, and return active context."""
    st.set_page_config(page_title=f"PropPilot — {page_title}", page_icon="🧭", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    if "pp_pending_rules_mode" in st.session_state:
        st.session_state["_pp_rules_mode_store"] = st.session_state.pop("pp_pending_rules_mode")

    presets = list_presets()
    rules_options = [_RULES_MODE_DEFAULT, "Upload YAML"]
    if "pp_custom_rules_yaml" in st.session_state:
        rules_options.append("Custom (Rules Lab)")

    _restore_widget_from_store(
        "pp_trades_mode",
        "_pp_trades_mode_store",
        _TRADES_MODE_DEFAULT,
        valid_options=_TRADES_MODE_OPTIONS,
    )
    _restore_widget_from_store(
        "pp_rules_mode",
        "_pp_rules_mode_store",
        _RULES_MODE_DEFAULT,
        valid_options=rules_options,
    )
    _restore_widget_from_store(
        "pp_account_size",
        "_pp_account_size_store",
        _ACCOUNT_SIZE_DEFAULT,
        valid_options=_ACCOUNT_SIZE_OPTIONS,
    )
    preset_default = _PRESET_DEFAULT if _PRESET_DEFAULT in presets else presets[0]
    _restore_widget_from_store(
        "pp_preset_name",
        "_pp_preset_name_store",
        preset_default,
        valid_options=presets,
    )
    _restore_widget_from_store("pp_api_key", "_pp_api_key_store", "")

    with st.sidebar:
        st.markdown("## 🧭 PropPilot")
        st.caption("Turn any crypto strategy into a prop-firm pass plan.")
        st.divider()

        trades_mode = st.radio(
            "Trades Source",
            options=list(_TRADES_MODE_OPTIONS),
            key="pp_trades_mode",
            on_change=_sync_widget_store,
            args=("pp_trades_mode", "_pp_trades_mode_store"),
        )
        raw_bytes: bytes | None = None
        file_ext = "csv"
        source_key = "sample"
        uploaded_dataset: TradeDataset | None = None

        if trades_mode == "Upload CSV / JSON":
            uploaded = st.file_uploader(
                "Upload Trades CSV or JSON",
                type=["csv", "json"],
                key="pp_trades_uploader",
                on_change=_on_trades_upload_change,
            )
            if uploaded is not None:
                _store_uploaded_trades(uploaded)
            if "pp_uploaded_bytes" in st.session_state:
                raw_bytes = st.session_state["pp_uploaded_bytes"]
                file_ext = st.session_state.get("pp_uploaded_ext", "csv")
                source_key = st.session_state.get("pp_uploaded_key") or hashlib.sha256(raw_bytes).hexdigest()[:16]
                uploaded_name = st.session_state.get("pp_uploaded_name", f"uploaded.{file_ext}")
                try:
                    uploaded_dataset = load_cached_trades(source_key, raw_bytes, file_ext)
                    st.caption(f"Using uploaded file: {uploaded_name} ({len(uploaded_dataset):,} trades)")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not parse uploaded file '{uploaded_name}': {exc}")
                    raw_bytes = None
                    file_ext = "csv"
                    source_key = "sample"
                    uploaded_dataset = None
                st.button(
                    "Clear upload",
                    key="pp_clear_trades_upload",
                    on_click=_clear_uploaded_trades,
                )
            else:
                st.info("No file uploaded yet — falling back to sample dataset.")

        rules_mode = st.radio(
            "Rules Source",
            options=rules_options,
            key="pp_rules_mode",
            on_change=_sync_widget_store,
            args=("pp_rules_mode", "_pp_rules_mode_store"),
        )
        account_size = float(
            st.selectbox(
                "Account Size (USD)",
                options=list(_ACCOUNT_SIZE_OPTIONS),
                key="pp_account_size",
                format_func=lambda v: f"${v:,.0f}",
                on_change=_sync_widget_store,
                args=("pp_account_size", "_pp_account_size_store"),
            )
        )

        if rules_mode == "Built-in Preset":
            preset_name = st.selectbox(
                "Preset Rulebook",
                options=presets,
                key="pp_preset_name",
                on_change=_sync_widget_store,
                args=("pp_preset_name", "_pp_preset_name_store"),
            )
            rules = load_preset(preset_name, account_size=account_size)
        elif rules_mode == "Upload YAML":
            yaml_up = st.file_uploader(
                "Upload Rules YAML",
                type=["yaml", "yml"],
                key="pp_yaml_up",
                on_change=_on_yaml_upload_change,
            )
            if yaml_up is not None:
                _store_uploaded_yaml(yaml_up)
            if "pp_uploaded_yaml_text" in st.session_state:
                yaml_name = st.session_state.get("pp_uploaded_yaml_name", "rules.yaml")
                try:
                    rules = parse_rules_yaml_text(
                        st.session_state["pp_uploaded_yaml_text"],
                        account_size=account_size,
                    )
                    st.caption(f"Using uploaded rules: {yaml_name}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not parse uploaded rules '{yaml_name}': {exc}")
                    rules = load_preset("two_step_classic", account_size=account_size)
                st.button(
                    "Clear rules upload",
                    key="pp_clear_yaml_upload",
                    on_click=_clear_uploaded_yaml,
                )
            else:
                st.info("Upload a YAML rulebook or switch to Preset — using Two-Step Classic.")
                rules = load_preset("two_step_classic", account_size=account_size)
        else:
            rules = parse_rules_yaml_text(
                st.session_state["pp_custom_rules_yaml"],
                account_size=account_size,
            )

        api_key = st.text_input(
            "Anthropic API Key (session only)",
            type="password",
            key="pp_api_key",
            on_change=_sync_widget_store,
            args=("pp_api_key", "_pp_api_key_store"),
            help="Stored in browser session state only; never written to disk.",
        )

        st.divider()
        st.caption(SUITE_FOOTER_MD)

    dataset = uploaded_dataset if uploaded_dataset is not None else load_cached_trades(source_key, raw_bytes, file_ext)
    return {
        "dataset": dataset,
        "source_key": source_key,
        "raw_bytes": raw_bytes,
        "file_ext": file_ext,
        "rules": rules,
        "rules_yaml": rules.to_yaml(),
        "api_key": api_key.strip() or None,
    }


def render_suite_footer() -> None:
    """Render shared Crypto Quant Suite footer on page bottom."""
    st.markdown(f'<div class="pp-footer">{SUITE_FOOTER_MD}</div>', unsafe_allow_html=True)


def format_opt_comparison_rows(res: OptimizationResult) -> list[dict[str, Any]]:
    """Build comparison table rows for Optimise vs Validate vs Bootstrap."""
    rows = []
    for label, sm in [
        (f"Optimise (first 60%, {res.opt_trade_count:,} trades)", res.opt_summary),
        (f"Validate (last 40%, {res.val_trade_count:,} trades)", res.val_summary),
        (f"Bootstrap ({res.boot_summary.n_total:,} x 1y paths)", res.boot_summary),
    ]:
        rows.append(
            {
                "Cohort": label,
                "Finished": sm.n_finished,
                "Unfinished": sm.n_unfinished,
                "Pass %": round(sm.pass_rate * 100.0, 1) if math.isfinite(sm.pass_rate) else None,
                "Breach %": round(sm.fail_rate * 100.0, 1) if math.isfinite(sm.fail_rate) else None,
                "Restart %": round(sm.restart_rate * 100.0, 1) if math.isfinite(sm.restart_rate) else None,
                "Median Days": (
                    round(sm.days_median, 1)
                    if sm.days_median is not None and math.isfinite(sm.days_median)
                    else None
                ),
                "Exp. Attempts": round(sm.expected_attempts, 2)
                if math.isfinite(sm.expected_attempts)
                else None,
                "Exp. Fees ($)": round(sm.expected_fees, 0)
                if math.isfinite(sm.expected_fees)
                else None,
            }
        )
    return rows
