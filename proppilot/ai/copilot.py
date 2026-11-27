"""Claude AI Copilot agent via Anthropic beta tool_runner plus deterministic offline fallback."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from anthropic import beta_tool
import numpy as np

from proppilot.ai.session import PlanSession, compact_json
from proppilot.optimize import ObjectiveName
from proppilot.sim import (
    format_days,
    format_expected_attempts,
    format_expected_cost,
    format_pct,
    format_usd,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
BETA_FLAGS = ["server-side-fallback-2026-07-01"]

COPILOT_PROMPT = """You are PropPilot Copilot, an institutional quantitative risk analyst helping crypto traders translate historical strategy trades into an honest prop-firm evaluation and funded plan.

Rules of conduct:
1. ALWAYS call the provided tools to inspect trade stats, rules, optimisation results, failures, and funded projections before citing ANY number. Never invent or guess figures.
2. Explain the plan in plain, actionable English: base risk per trade, near-target de-risking, own daily stop vs firm limit, restart horizon, expected attempts/fees, and funded payout/CPPI mechanics.
3. Highlight the exact failure modes observed in the data (e.g., correlated same-day crypto stop-outs across BTC/ETH/SOL triggering daily loss limits, or trailing drawdown ratcheting).
4. Explicitly distinguish between the optimisation split (first 60% of trades), out-of-sample validation split (last 40%), and weekly block-bootstrap Monte Carlo. Call out any overfit or regime-shift warnings directly. Choose or tune policies only on optimisation-split numbers; validation numbers are a one-way holdout check and must never be used to pick a policy.
5. Always state the probability of losing the funded account within 12 months next to any funded payout figure, and never recommend a funded plan whose P(lose) exceeds the configured limit without saying so prominently.
6. State risks honestly in every plan summary:
   - Historical trade performance does not guarantee future results.
   - Strategy edge decays over time and crypto volatility regimes shift.
   - Real execution involves slippage, funding rates, exchange fees, and intraday mark-to-market drawdown not visible in trade-level exit logs.
"""


@dataclass
class CopilotReply:
    """Response container returned by AI or offline copilot."""

    text: str
    used_ai: bool
    fallback_reason: str | None = None
    tool_calls_count: int = 0


def build_copilot_tools(session: PlanSession) -> list[Any]:
    """Create the 7 @beta_tool closures bound to `session`."""

    @beta_tool
    def get_trade_stats() -> str:
        """Return summary statistics of the loaded crypto trade history and 60/40 split.

        Returns compact JSON with trade count, symbols, win rate, expectancy in R, and span.
        """
        try:
            return session.tool_get_trade_stats()
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    @beta_tool
    def get_rules() -> str:
        """Return the active prop-firm evaluation and funded account rules.

        Returns compact JSON with phases, daily/max loss limits, reset hour, fee, and notes.
        """
        try:
            return session.tool_get_rules()
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    @beta_tool
    def simulate_policy(
        policy_json: str,
        period: Literal["optimise", "validate", "bootstrap"] = "validate",
    ) -> str:
        """Simulate a sizing policy on one evaluation split or weekly block bootstrap.

        Args:
            policy_json: JSON object of SizingPolicy fields (empty '{}' uses active policy).
            period: Data window to test: 'optimise' (first 60%), 'validate' (last 40%), or 'bootstrap'.
        """
        try:
            return session.tool_simulate_policy(policy_json=policy_json, period=period)
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    @beta_tool
    def optimise(
        objective: Literal["fastest", "cheapest", "max_value"] = "fastest",
        max_fail: float = 0.2,
    ) -> str:
        """Run honest 60/40 grid optimisation for the requested objective and max failure rate.

        Args:
            objective: Optimisation goal: 'fastest', 'cheapest', or 'max_value'.
            max_fail: Maximum allowed non-pass probability on optimisation split (0.0 to 1.0).
        """
        try:
            return session.tool_optimise(objective=objective, max_fail=max_fail)
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    @beta_tool
    def compare_policies(policies_json: str) -> str:
        """Compare up to 6 sizing policies on the optimisation split (first 60%) and the holdout validation split (last 40%).

        Rank policies on the optimisation numbers; the validation numbers are a holdout check only.

        Args:
            policies_json: JSON array of up to 6 SizingPolicy parameter objects (missing fields use the active policy).
        """
        try:
            return session.tool_compare_policies(policies_json=policies_json)
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    @beta_tool
    def funded_projection(policy_json: str) -> str:
        """Project 12-month funded payouts, CPPI sizing, and P(lose account) via block bootstrap.

        Args:
            policy_json: JSON object of SizingPolicy/funded fields ('{}' uses active policy).
        """
        try:
            return session.tool_funded_projection(policy_json=policy_json)
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    @beta_tool
    def explain_failures(policy_json: str) -> str:
        """Inspect which specific UTC days, symbols, and correlated losers caused rule breaches.

        Args:
            policy_json: JSON object of SizingPolicy fields ('{}' uses active policy).
        """
        try:
            return session.tool_explain_failures(policy_json=policy_json)
        except Exception as exc:  # noqa: BLE001
            return compact_json({"error": str(exc)})

    return [
        get_trade_stats,
        get_rules,
        simulate_policy,
        optimise,
        compare_policies,
        funded_projection,
        explain_failures,
    ]


def _extract_text_from_message(msg: Any) -> str:
    blocks = getattr(msg, "content", None) or []
    parts: list[str] = []
    for blk in blocks:
        if getattr(blk, "type", None) == "text":
            txt = getattr(blk, "text", "")
            if txt:
                parts.append(txt)
    return "\n\n".join(parts).strip()


def run_copilot_query(
    session: PlanSession,
    prompt: str,
    history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> CopilotReply:
    """Run one copilot turn using Anthropic beta.messages.tool_runner, with offline fallback."""
    tools = build_copilot_tools(session)

    resolved_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
    if client is None and not resolved_key:
        return CopilotReply(
            text=generate_offline_plan_explanation(session, user_question=prompt),
            used_ai=False,
            fallback_reason="No ANTHROPIC_API_KEY configured; generated deterministic offline analysis.",
            tool_calls_count=0,
        )

    if client is None:
        client = anthropic.Anthropic(api_key=resolved_key) if api_key else anthropic.Anthropic()

    messages: list[dict[str, Any]] = []
    if history:
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    start_log_len = len(session.tool_log)
    model_name = os.getenv("PP_MODEL", DEFAULT_MODEL)

    try:
        runner = client.beta.messages.tool_runner(
            model=model_name,
            max_tokens=16000,
            max_iterations=20,
            system=COPILOT_PROMPT,
            output_config={"effort": "high"},
            betas=BETA_FLAGS,
            fallbacks="default",
            tools=tools,
            messages=messages,
        )
        final_msg = None
        for message in runner:
            final_msg = message
            if getattr(message, "stop_reason", None) == "refusal":
                return CopilotReply(
                    text=generate_offline_plan_explanation(session, user_question=prompt),
                    used_ai=False,
                    fallback_reason="Model stopped with stop_reason='refusal'; switched to offline explainer.",
                    tool_calls_count=len(session.tool_log) - start_log_len,
                )

        text = _extract_text_from_message(final_msg) if final_msg is not None else ""
        if not text:
            return CopilotReply(
                text=generate_offline_plan_explanation(session, user_question=prompt),
                used_ai=False,
                fallback_reason="Empty AI response text; switched to offline explainer.",
                tool_calls_count=len(session.tool_log) - start_log_len,
            )

        return CopilotReply(
            text=text,
            used_ai=True,
            fallback_reason=None,
            tool_calls_count=len(session.tool_log) - start_log_len,
        )
    except anthropic.APIError as exc:
        logger.warning("Anthropic APIError in copilot; using offline fallback: %s", exc)
        return CopilotReply(
            text=generate_offline_plan_explanation(session, user_question=prompt),
            used_ai=False,
            fallback_reason=f"Anthropic API error ({exc}); switched to offline explainer.",
            tool_calls_count=len(session.tool_log) - start_log_len,
        )


def explain_plan(
    session: PlanSession,
    use_ai: bool = False,
    api_key: str | None = None,
    client: Any | None = None,
) -> CopilotReply:
    """Generate a full plan explanation via AI (when enabled and keyed) or offline deterministically."""
    res = session.ensure_optimised()
    if not use_ai:
        return CopilotReply(
            text=generate_offline_plan_explanation(session),
            used_ai=False,
            fallback_reason=None,
            tool_calls_count=0,
        )

    prompt = (
        f"Explain the recommended prop-firm pass plan for rules '{res.rules.name}' "
        f"(account size ${res.rules.account_size:,.0f}, objective='{res.objective}', "
        f"max_fail={res.max_fail:.2f}). Call get_trade_stats, get_rules, "
        "simulate_policy on 'validate' and 'bootstrap', explain_failures, and funded_projection "
        "using the current recommended policy "
        f"{json.dumps(res.best_policy.model_dump(mode='json'))}. "
        "Cite exact figures from the tools, highlight failure causes, and state risks honestly."
    )
    return run_copilot_query(session, prompt=prompt, api_key=api_key, client=client)


def generate_offline_plan_explanation(
    session: PlanSession,
    user_question: str | None = None,
) -> str:
    """Deterministic, data-grounded plan explanation requiring zero external API calls."""
    res = session.ensure_optimised()
    stats = session.dataset.summary_stats()
    pol = res.best_policy
    opt = res.opt_summary
    val = res.val_summary
    boot = res.boot_summary
    fn = res.funded_summary
    fail_attr = res.failure_attribution

    val_days = format_days(val.days_median, suffix=" days")
    opt_days = format_days(opt.days_median, suffix=" days")
    boot_days = format_days(boot.days_median, suffix=" days")
    val_cost = format_expected_cost(val.expected_fees)
    val_cost_phrase = f"**{val_cost}** per funded account" if math.isfinite(val.expected_fees) else f"**{val_cost}**"
    val_attempts = format_expected_attempts(val.expected_attempts)

    lines: list[str] = []

    if user_question:
        q_lower = user_question.lower()
        if "fail" in q_lower or "breach" in q_lower or "loss" in q_lower:
            lines.append("### Failure & Breach Diagnostics (Offline Analysis)")
            hard = {k: v for k, v in fail_attr.fail_reason_counts.items() if k != "restart_timeout"}
            restarts = fail_attr.fail_reason_counts.get("restart_timeout", 0)
            lines.append(
                f"Across **{fail_attr.n_total_starts}** out-of-sample rolling starts (last 40% of trades), "
                f"**{fail_attr.n_breaches}** starts breached hard firm rules ("
                + (", ".join(f"`{k}`={v}" for k, v in hard.items()) if hard else "none")
                + f"); **{restarts}** more were abandoned by the policy's own restart cutoff."
            )
            if fail_attr.top_breach_days:
                top_d = fail_attr.top_breach_days[0]
                lines.append(
                    f"The worst single day was **{top_d['day_utc']}** ({top_d['breach_count']} rolling breaches, "
                    f"symbols {', '.join(top_d['symbols_involved'])}, mean day sum {top_d['mean_day_r_sum']}R)."
                )
            lines.append("")

    lines.extend(
        [
            f"### Recommended Pass Plan ({res.rules.name} — ${res.rules.account_size:,.0f})",
            f"- **Strategy baseline**: {stats['n_trades']:,} trades across {stats['n_symbols']} symbols "
            f"({stats['trades_per_month']} trades/mo, win rate {stats['win_rate'] * 100:.1f}%, "
            f"expectancy {stats['expectancy_r']:+.3f}R/trade).",
            f"- **Sizing rule**: Risk **{pol.risk_pct:.2g}%** per trade (`{pol.risk_basis}`). "
            + (
                f"Cut risk to **{pol.near_mult:.2g}x** once phase progress reaches **{int((pol.near_frac or 0) * 100)}%** of target. "
                if pol.near_frac is not None and pol.near_mult != 1.0
                else ""
            )
            + f"Enforce an internal daily stop of **{pol.own_daily_stop_pct:.2g}%** (below the firm's "
            f"**{res.rules.daily_loss_pct:.2g}%** daily limit). "
            + (
                f"Abandon and restart unfinished challenges after **{pol.restart_after_days} days**."
                if pol.restart_after_days is not None
                else "No early restart cutoff."
            ),
            "",
            f"### Honest Out-of-Sample & Bootstrap Evidence",
            f"- **Optimise (first 60%, {res.opt_trade_count:,} trades)**: "
            f"pass rate **{format_pct(opt.pass_rate)}**, median **{opt_days}**.",
            f"- **Validate (last 40%, {res.val_trade_count:,} trades — unseen)**: "
            f"pass rate **{format_pct(val.pass_rate)}**, median **{val_days}**, "
            f"expected attempts **{val_attempts}**, expected challenge fee {val_cost_phrase}.",
            f"- **Block Bootstrap ({boot.n_total:,} x 1-yr synthetic paths)**: "
            f"pass rate **{format_pct(boot.pass_rate)}**, median **{boot_days}**.",
        ]
    )

    if fail_attr.top_breach_days:
        top_day_summary = "; ".join(
            f"{d['day_utc']} ({d['breach_count']} breaches across {', '.join(d['symbols_involved'][:3])})"
            for d in fail_attr.top_breach_days[:3]
        )
        lines.append(f"- **Primary failure clusters**: {top_day_summary}.")
    else:
        lines.append("- **Primary failure clusters**: No hard rule breaches observed in validation starts.")

    lines.extend(
        [
            "",
            "### 12-Month Funded Projection",
            f"- **Funded risk**: {res.funded_risk_line()}.",
            f"- Funded settings: {pol.funded_label()} (chosen on the optimisation slice as the highest expected "
            f"payout with P(lose) <= {res.max_lose_funded * 100:.0f}%).",
            f"- Projected mean payout: **{format_usd(fn.monthly_payout_mean_usd, suffix='/month')}** "
            f"(**{f'{fn.monthly_payout_mean_pct:.2f}%/mo' if math.isfinite(fn.monthly_payout_mean_pct) else 'N/A'}**, "
            f"lost accounts included; median **{format_usd(fn.monthly_payout_median_usd, suffix='/mo')}**).",
        ]
    )

    if res.warnings:
        lines.append("")
        lines.append("### Diagnostic Warnings")
        for w in res.warnings:
            lines.append(f"- {w}")

    lines.extend(
        [
            "",
            "### Honest Risk Disclosures",
            "- Past historical trades do not guarantee future prop-firm pass rates or funded income.",
            "- Strategy edge can decay and crypto cross-asset correlations spike during market cascades.",
            "- Trade-level simulations approximate intraday mark-to-market equity between entry and exit, "
            "and real execution introduces slippage, spread, and funding costs not fully captured in R multiples.",
        ]
    )

    return "\n".join(lines)
