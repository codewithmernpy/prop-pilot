"""Tests for Anthropic AI schema/tool builders, mocked SDK calls, offline fallbacks, and reports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # the anthropic 2.x SDK ships httpx2 (plain httpx is not installed with it)
import pytest
from anthropic.lib._parse._transform import transform_schema

from proppilot.ai import (
    PlanSession,
    RuleExtractionError,
    build_copilot_tools,
    explain_plan,
    extract_rules_from_text,
    run_copilot_query,
)
from proppilot.ingest import trades_from_dataframe
from proppilot.optimize import run_honest_optimisation
from proppilot.report import render_reports
from proppilot.rules import PhaseRule, PropRules, load_preset
from proppilot.synthetic import SyntheticConfig, generate_synthetic_trades


def test_proprules_schema_and_tools_build_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # PropRules transforms cleanly into Anthropic structured-output JSON schema
    schema = transform_schema(PropRules)
    assert schema["type"] == "object"
    assert "ambiguities" in schema["properties"]
    assert "phases" in schema["properties"]

    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=120, seed=5)))
    rules = load_preset("two_step_classic")
    session = PlanSession(dataset=ds, rules=rules, n_bootstrap=50, seed=42)

    tools = build_copilot_tools(session)
    assert len(tools) == 7
    names = {t.name for t in tools}
    assert names == {
        "get_trade_stats",
        "get_rules",
        "simulate_policy",
        "optimise",
        "compare_policies",
        "funded_projection",
        "explain_failures",
    }
    for t in tools:
        defn = t.to_dict()
        assert "input_schema" in defn

    # Invoke every tool directly and assert compact JSON strictly < 3000 chars
    outputs = [
        tools[0](),  # get_trade_stats
        tools[1](),  # get_rules
        tools[2](policy_json="{}", period="validate"),  # simulate_policy
        tools[3](objective="fastest", max_fail=0.35),  # optimise
        tools[4](policies_json=json.dumps([{"risk_pct": 0.5}, {"risk_pct": 0.8}])),  # compare_policies
        tools[5](policy_json="{}"),  # funded_projection
        tools[6](policy_json="{}"),  # explain_failures
    ]
    for out in outputs:
        assert isinstance(out, str)
        assert len(out) < 3000
        json.loads(out)


def test_rule_extractor_sdk_call_and_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuleExtractionError):
        extract_rules_from_text("Max daily loss 5%.")

    captured: dict[str, object] = {}

    class _FakeParseClient:
        class beta:
            class messages:
                @staticmethod
                def parse(**kwargs: object) -> object:
                    captured.update(kwargs)
                    return SimpleNamespace(
                        stop_reason="end_turn",
                        parsed_output=PropRules(
                            name="Extracted Generic Rules",
                            phases=[PhaseRule(name="Step 1", target_pct=8.0, min_days=3)],
                            daily_loss_pct=4.0,
                            daily_basis="day_start_balance",
                            daily_reset=0,
                            max_loss_pct=8.0,
                            max_loss_type="static",
                            fee=400.0,
                            profit_split=80.0,
                            payout_frequency_days=14,
                            account_size=100_000.0,
                            ambiguities=["Daily loss basis (balance vs equity) was not stated."],
                        ),
                    )

    parsed = extract_rules_from_text("8% target, 4% daily, 8% max loss.", client=_FakeParseClient())
    assert parsed.ambiguities == ["Daily loss basis (balance vs equity) was not stated."]
    assert captured["betas"] == ["server-side-fallback-2026-07-01"]
    assert captured["fallbacks"] == "default"
    assert captured["output_config"] == {"effort": "high"}
    assert captured["output_format"] is PropRules

    class _RefusalClient:
        class beta:
            class messages:
                @staticmethod
                def parse(**kwargs: object) -> object:
                    return SimpleNamespace(stop_reason="refusal", parsed_output=None)

    with pytest.raises(RuleExtractionError, match="refusal"):
        extract_rules_from_text("test", client=_RefusalClient())


def test_copilot_tool_runner_and_offline_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=120, seed=9)))
    rules = load_preset("two_step_classic")
    session = PlanSession(dataset=ds, rules=rules, n_bootstrap=50, seed=42)

    # Without API key: deterministic offline fallback works seamlessly
    reply_off = run_copilot_query(session, "Why do challenges fail?")
    assert not reply_off.used_ai
    assert "Recommended Pass Plan" in reply_off.text

    # Mocked tool_runner verifying exact parameters and refusal fallback
    captured: dict[str, object] = {}

    class _FakeRunnerClient:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs: object) -> list[object]:
                    captured.update(kwargs)
                    return [
                        SimpleNamespace(
                            stop_reason="end_turn",
                            content=[SimpleNamespace(type="text", text="Validated plan summary.")],
                        )
                    ]

    reply_ai = run_copilot_query(session, "Explain my plan", client=_FakeRunnerClient())
    assert reply_ai.used_ai
    assert reply_ai.text == "Validated plan summary."
    assert captured["betas"] == ["server-side-fallback-2026-07-01"]
    assert captured["fallbacks"] == "default"
    assert captured["max_iterations"] == 20
    assert captured["output_config"] == {"effort": "high"}

    class _RefusalRunnerClient:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs: object) -> list[object]:
                    return [SimpleNamespace(stop_reason="refusal", content=[])]

    reply_refusal = run_copilot_query(session, "Explain my plan", client=_RefusalRunnerClient())
    assert not reply_refusal.used_ai
    assert "refusal" in (reply_refusal.fallback_reason or "").lower()

    class _ErrorRunnerClient:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs: object) -> list[object]:
                    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                    raise anthropic.APIConnectionError(request=req)

    reply_err = run_copilot_query(session, "Explain my plan", client=_ErrorRunnerClient())
    assert not reply_err.used_ai
    assert "Recommended Pass Plan" in reply_err.text


def test_html_and_json_report_rendering(tmp_path: Path) -> None:
    ds = trades_from_dataframe(generate_synthetic_trades(SyntheticConfig(days=180, seed=13)))
    rules = load_preset("two_step_classic")
    res = run_honest_optimisation(ds, rules, objective="fastest", max_fail=0.35, n_bootstrap=80, seed=42)
    session = PlanSession(dataset=ds, rules=rules, n_bootstrap=80, opt_result=res)
    reply = explain_plan(session, use_ai=False)

    html_p, json_p = render_reports(
        dataset=ds,
        result=res,
        explanation_text=reply.text,
        used_ai=reply.used_ai,
        out_dir=tmp_path,
        stem="test_plan",
    )
    assert html_p.exists()
    assert json_p.exists()
    html_txt = html_p.read_text(encoding="utf-8")
    assert "Recommended Execution Plan Card" in html_txt
    assert "cdn.plot.ly" in html_txt
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    assert payload["objective"] == "fastest"
    assert "validate" in payload and "optimise" in payload and "funded" in payload
