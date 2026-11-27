"""Prop-firm rules text extraction into PropRules.

Two paths:
- `extract_rules_from_text`: Claude structured output via `client.beta.messages.parse`
  (`output_format=PropRules`), with an explicit `ambiguities` list.
- `extract_rules_offline`: deterministic keyword parser used when no API key is configured or
  the AI call fails / is refused. It never guesses silently: every field it could not find in
  the text is filled with a stated default AND listed in `ambiguities`.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import anthropic
import pydantic

from proppilot.rules import PhaseRule, PropRules

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
BETA_FLAGS = ["server-side-fallback-2026-07-01"]

EXTRACT_PROMPT = """You are a strict compliance parser extracting prop-trading evaluation and funded account rules into a structured PropRules specification.

Critical instructions:
1. Extract ONLY what the provided rules text states explicitly.
2. NEVER invent, guess, or silently assume missing or ambiguous parameters.
3. Whenever ANY rule detail is missing, vague, or open to interpretation (for example: whether daily loss is based on day-start balance vs equity, the exact UTC daily reset hour, static vs trailing drawdown basis, closed vs equity trailing, whether a trailing drawdown stops trailing (locks) at the initial balance, consistency percentage, minimum trading days, time limits, payout frequency, fee, profit split, or max risk per trade), add a clear, specific entry to `ambiguities` explaining what was not stated clearly and what placeholder value was used so the user can review and confirm before running simulations.
4. Use generic rule names only; do not introduce third-party brand assumptions beyond what is in the input text.
5. For instant-funded programs with no evaluation phases, set `phases` to an empty list `[]`.
"""


class RuleExtractionError(RuntimeError):
    """Raised when AI rule extraction cannot complete (missing key, refusal, invalid output, or API error)."""


def extract_rules_from_text(
    text: str,
    api_key: str | None = None,
    client: Any | None = None,
) -> PropRules:
    """Parse unstructured prop-firm rules text into a validated PropRules model with Claude.

    Raises RuleExtractionError if no API key is available, the model refuses, the output does
    not validate against PropRules, or the API fails. Callers fall back to
    `extract_rules_offline`.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input rules text cannot be empty.")

    if client is None:
        resolved_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuleExtractionError(
                "No ANTHROPIC_API_KEY configured. Provide a key or edit rules YAML manually offline."
            )
        client = anthropic.Anthropic(api_key=resolved_key) if api_key else anthropic.Anthropic()

    model_name = os.getenv("PP_MODEL", DEFAULT_MODEL)
    try:
        resp = client.beta.messages.parse(
            model=model_name,
            max_tokens=16000,
            betas=BETA_FLAGS,
            fallbacks="default",
            output_config={"effort": "high"},
            system=EXTRACT_PROMPT,
            messages=[{"role": "user", "content": cleaned}],
            output_format=PropRules,
        )
    except anthropic.APIError as exc:
        logger.warning("Anthropic API error during rule extraction: %s", exc)
        raise RuleExtractionError(f"Anthropic API error: {exc}") from exc
    except pydantic.ValidationError as exc:
        # The SDK validates the JSON text against PropRules (including its range validators).
        raise RuleExtractionError(f"Model output did not validate as PropRules: {exc}") from exc

    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason == "refusal":
        raise RuleExtractionError("Model returned stop_reason='refusal' while parsing rules text.")
    if stop_reason == "max_tokens":
        raise RuleExtractionError("Model output was truncated (stop_reason='max_tokens').")

    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        raise RuleExtractionError("Model response did not include a parsed PropRules output.")

    if isinstance(parsed, PropRules):
        return parsed
    try:
        return PropRules.model_validate(parsed)
    except pydantic.ValidationError as exc:
        raise RuleExtractionError(f"Model output did not validate as PropRules: {exc}") from exc


# ---------------------------------------------------------------------------
# Deterministic offline keyword parser
# ---------------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_PCT = _NUM + r"\s*%"


def _search(pattern: str, text: str) -> Optional[re.Match[str]]:
    return re.search(pattern, text, flags=re.IGNORECASE)


def _clause(pattern: str, text: str) -> str:
    """Return the text from the first match of `pattern` to the end of its line/sentence."""
    m = _search(pattern + r"(?:[^\n.;]|\.\d)*", text)
    return m.group(0) if m else ""


def _money(raw: str) -> float:
    return float(raw.replace(",", ""))


def extract_rules_offline(text: str) -> PropRules:
    """Parse rules text with deterministic keyword rules (no network, no API key).

    Recognises common phrasings of profit targets per phase, minimum trading days, time
    limits, daily loss (and its balance/equity basis and reset hour), max loss (static or
    trailing, lock at initial balance), consistency, fee, account size, profit split, payout
    frequency, and max risk per trade. Anything not found is defaulted and flagged.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input rules text cannot be empty.")
    t = re.sub(r"[ \t]+", " ", cleaned)
    amb: list[str] = ["Parsed offline with keyword rules (no AI); confirm every value before relying on it."]

    # --- phases ---------------------------------------------------------------------------
    phase_targets: list[float] = []
    for idx in (1, 2, 3):
        m = _search(rf"(?:phase|step|stage)\s*{idx}\b[^\n;%]*?{_PCT}", t)
        if not m:
            break
        phase_targets.append(float(m.group(1)))
    instant = bool(_search(r"\binstant(?:ly)?\s+fund|\bno\s+evaluation\b", t))
    if not phase_targets and not instant:
        m = _search(rf"profit\s+target[^\n;%]*?{_PCT}", t)
        if m:
            phase_targets.append(float(m.group(1)))
        else:
            amb.append("No profit target found; assumed an instant-funded program (no evaluation phases).")

    min_days = 0
    m = _search(r"min(?:imum)?\.?\s+(?:of\s+)?(\d+)\s+(?:trading\s+)?days", t) or _search(
        r"min(?:imum)?\.?\s+trading\s+days\s*:?\s*(\d+)", t
    )
    if m:
        min_days = int(m.group(1))
    elif phase_targets:
        amb.append("Minimum trading days not stated; used 0.")

    max_days: Optional[int] = None
    if _search(r"\b(?:unlimited|no)\s+(?:time\s+limit|trading\s+period|deadline)|time\s+limit\s*:?\s*(?:none|unlimited)", t):
        max_days = None
    else:
        m = _search(r"time\s+limit\s*:?\s*(\d+)\s*(?:calendar\s+)?days", t) or _search(
            r"(\d+)\s*(?:calendar\s+)?days?\s+(?:time\s+limit|to\s+(?:complete|pass))", t
        )
        if m:
            max_days = int(m.group(1))
        elif phase_targets:
            amb.append("Phase time limit not stated; assumed unlimited.")

    phases = [
        PhaseRule(name=f"Phase {i + 1}", target_pct=tgt, min_days=min_days, max_days=max_days)
        for i, tgt in enumerate(phase_targets)
    ]

    # --- daily loss -------------------------------------------------------------------------
    daily_loss = 5.0
    daily_basis = "day_start_balance"
    daily_pattern = r"daily\s+(?:loss|drawdown)"
    daily_clause = _clause(daily_pattern, t)
    m = _search(rf"{daily_pattern}[^\n%]*?{_PCT}", t)
    if m:
        daily_loss = float(m.group(1))
    else:
        amb.append("Daily loss limit not found; used 5%.")
    if _search(r"equity", daily_clause):
        daily_basis = "day_start_max_balance_equity"
        amb.append("Daily loss references equity; modelled with day-start closed balance (trade-level data).")
    elif not _search(r"balance", daily_clause):
        amb.append("Daily loss basis (day-start balance vs equity) not stated; used day-start balance.")

    daily_reset = 0
    m = _search(r"(?:reset|resets|resetting|midnight)[^\n;]*?(\d{1,2})[:.](\d{2})\s*(utc|gmt|cet|cest|est|edt|et)?", t)
    zone = (m.group(3) or "").lower() if m else ""
    if m and zone in ("utc", "gmt") and 0 <= int(m.group(1)) <= 23:
        daily_reset = int(m.group(1))
    elif m and zone:
        amb.append(f"Daily reset given as {m.group(1)}:{m.group(2)} {zone.upper()}; convert it to a UTC hour (used 0).")
    else:
        amb.append("Daily reset time (UTC hour) not stated; used 00:00 UTC.")

    # --- max loss ---------------------------------------------------------------------------
    max_loss = 10.0
    max_type = "static"
    trailing_lock = False
    ml_pattern = (
        r"(?:max(?:imum)?\.?\s+(?:overall|total|trailing|static)\s+(?:loss|drawdown)"
        r"|overall\s+(?:static\s+|trailing\s+)?(?:loss|drawdown)|total\s+(?:loss|drawdown)"
        r"|max(?:imum)?\.?\s+(?:loss|drawdown)|trailing\s+drawdown)"
    )
    m = _search(ml_pattern + rf"[^\n%]*?{_PCT}", t)
    ml_clause = _clause(ml_pattern, t)
    if m:
        max_loss = float(m.group(1))
    else:
        amb.append("Maximum overall loss not found; used 10%.")
    if _search(r"trailing", ml_clause):
        max_type = "trailing_equity" if _search(r"equity", ml_clause) else "trailing_closed"
        if max_type == "trailing_equity":
            amb.append("Equity-based trailing drawdown is approximated with closed-trade balances.")
        if _search(r"\block|stops?\s+trailing|no\s+longer\s+trail", t):
            trailing_lock = True
        else:
            amb.append("Not stated whether the trailing drawdown locks at the initial balance; assumed it never locks.")
    elif not _search(r"static", ml_clause):
        amb.append("Max loss type (static vs trailing) not stated; used static from the initial balance.")

    # --- consistency, fee, size, split, payouts, per-trade cap ------------------------------
    consistency: Optional[float] = None
    m = _search(rf"consisten\w*[^\n%]*?{_PCT}", t)
    if m:
        consistency = float(m.group(1))
    elif _search(r"consisten", t):
        amb.append("A consistency rule is mentioned without a percentage; ignored.")

    account_size = 100_000.0
    m = _search(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k\b)?\s*(?:account|funded|balance)", t) or _search(
        r"account\s+(?:size|balance)\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?", t
    )
    if m:
        account_size = _money(m.group(1)) * (1000.0 if m.group(2) else 1.0)
    else:
        amb.append("Account size not stated; used $100,000.")

    fee = 0.0
    m = _search(r"fee[^\n$]*?\$\s*([\d,]+(?:\.\d+)?)", t) or _search(
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:evaluation\s+|challenge\s+)?fee", t
    )
    if m:
        fee = _money(m.group(1))
    else:
        amb.append("Challenge fee not stated; used $0 (expected-cost figures will be zero).")

    split = 80.0
    m = _search(rf"split[^\n%]*?{_PCT}", t) or _search(rf"{_PCT}\s*(?:profit\s+)?split", t)
    if m:
        split = float(m.group(1))
    else:
        amb.append("Profit split not stated; used 80%.")

    payout_days = 14
    m = _search(r"(?:payouts?|withdrawals?)[^\n;]*?every\s+(\d+)\s*(?:calendar\s+)?days", t)
    if m:
        payout_days = int(m.group(1))
    elif _search(r"bi-?weekly", t):
        payout_days = 14
    elif _search(r"(?:payouts?|withdrawals?)[^\n;]*?\bweekly", t):
        payout_days = 7
    elif _search(r"(?:payouts?|withdrawals?)[^\n;]*?\bmonthly", t):
        payout_days = 30
    else:
        amb.append("Payout frequency not stated; used every 14 days.")

    max_risk: Optional[float] = None
    m = _search(rf"risk\s+per\s+trade[^\n%]*?{_PCT}", t)
    if m:
        max_risk = float(m.group(1))

    try:
        return PropRules(
            name="Offline-parsed rules",
            phases=phases,
            daily_loss_pct=daily_loss,
            daily_basis=daily_basis,  # type: ignore[arg-type]
            daily_reset=daily_reset,
            max_loss_pct=max_loss,
            max_loss_type=max_type,  # type: ignore[arg-type]
            trailing_lock_at_initial=trailing_lock,
            consistency_pct=consistency,
            fee=fee,
            profit_split=split,
            payout_frequency_days=payout_days,
            account_size=account_size,
            max_risk_per_trade_pct=max_risk,
            ambiguities=amb,
        )
    except pydantic.ValidationError as exc:
        raise ValueError(f"Offline parser found values outside the PropRules limits: {exc}") from exc
