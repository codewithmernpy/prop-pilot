"""Stateful PlanSession bridging simulation/optimisation outputs to AI tools and offline explainers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from proppilot.ingest import TradeDataset
from proppilot.optimize import (
    DEFAULT_MAX_LOSE_FUNDED,
    ObjectiveName,
    OptimizationResult,
    run_honest_optimisation,
)
from proppilot.policy import SizingPolicy, default_policy_for_rules
from proppilot.rules import PropRules
from proppilot.sim import (
    analyze_failures,
    simulate_bootstrap_challenge,
    simulate_funded_bootstrap,
    simulate_rolling_starts,
)

MAX_TOOL_JSON_CHARS = 2850


def compact_json(payload: Any, max_chars: int = MAX_TOOL_JSON_CHARS) -> str:
    """Serialize payload to compact JSON guaranteed strictly below `max_chars` (< 3000)."""
    text = json.dumps(payload, separators=(",", ":"), default=_json_default)
    if len(text) <= max_chars:
        return text

    # Iteratively trim large list fields if needed so output stays valid JSON < max_chars.
    if isinstance(payload, dict):
        trimmed = dict(payload)
        for key in ("sample_breaches", "top_breach_days", "results", "grid", "warnings"):
            if key in trimmed and isinstance(trimmed[key], list):
                trimmed[key] = trimmed[key][:2]
                text = json.dumps(trimmed, separators=(",", ":"), default=_json_default)
                if len(text) <= max_chars:
                    return text
        for key in list(trimmed.keys()):
            if isinstance(trimmed[key], list) and len(trimmed[key]) > 1:
                trimmed[key] = trimmed[key][:1]
                text = json.dumps(trimmed, separators=(",", ":"), default=_json_default)
                if len(text) <= max_chars:
                    return text

    # Last resort: embed a prefix of the serialized payload as a string. JSON escaping can grow
    # the text, so shrink the prefix until the final document is below the limit.
    keep = max_chars - 64
    while keep > 0:
        out = json.dumps({"truncated": True, "summary": text[:keep]}, separators=(",", ":"))
        if len(out) <= max_chars:
            return out
        keep -= max(1, len(out) - max_chars)
    return json.dumps({"truncated": True, "summary": ""}, separators=(",", ":"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


@dataclass
class PlanSession:
    """Session container holding strategy trades, prop rules, results, and tool-call audit log."""

    dataset: TradeDataset
    rules: PropRules
    objective: ObjectiveName = "fastest"
    max_fail: float = 0.20
    n_bootstrap: int = 2000
    seed: int = 42
    max_lose_funded: float = DEFAULT_MAX_LOSE_FUNDED
    opt_result: OptimizationResult | None = None
    tool_log: list[dict[str, Any]] = field(default_factory=list)

    def ensure_optimised(self) -> OptimizationResult:
        """Return cached OptimizationResult or run honest 60/40 optimisation lazily."""
        if self.opt_result is None:
            self.opt_result = run_honest_optimisation(
                dataset=self.dataset,
                rules=self.rules,
                objective=self.objective,
                max_fail=self.max_fail,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed,
                max_lose_funded=self.max_lose_funded,
            )
        return self.opt_result

    def active_policy(self) -> SizingPolicy:
        """Return current best policy if optimised, otherwise a baseline rule-aware policy."""
        if self.opt_result is not None:
            return self.opt_result.best_policy
        return default_policy_for_rules(self.rules)

    def _record_tool(self, name: str, args: dict[str, Any], result_json: str) -> str:
        self.tool_log.append(
            {
                "tool": name,
                "args": args,
                "chars": len(result_json),
                "result_preview": result_json[:240],
            }
        )
        return result_json

    def tool_get_trade_stats(self) -> str:
        opt_ds, val_ds = self.dataset.split_chronological(0.6)
        payload = {
            "full": self.dataset.summary_stats(),
            "optimise_60pct": {
                "n_trades": len(opt_ds),
                "expectancy_r": round(float(np.mean(opt_ds.r_vals)), 4),
                "win_rate": round(float(np.mean(opt_ds.r_vals > 0)), 4),
            },
            "validate_40pct": {
                "n_trades": len(val_ds),
                "expectancy_r": round(float(np.mean(val_ds.r_vals)), 4),
                "win_rate": round(float(np.mean(val_ds.r_vals > 0)), 4),
            },
        }
        return self._record_tool("get_trade_stats", {}, compact_json(payload))

    def tool_get_rules(self) -> str:
        payload = self.rules.model_dump(mode="json")
        payload["approximation_notes"] = self.rules.approximation_notes
        return self._record_tool("get_rules", {}, compact_json(payload))

    def tool_simulate_policy(
        self,
        policy_json: str,
        period: Literal["optimise", "validate", "bootstrap"] = "validate",
    ) -> str:
        pol = _parse_policy_json(policy_json, self.active_policy())
        opt_ds, val_ds = self.dataset.split_chronological(0.6)
        if period == "optimise":
            sm = simulate_rolling_starts(opt_ds, self.rules, pol)
        elif period == "validate":
            sm = simulate_rolling_starts(val_ds, self.rules, pol)
        elif period == "bootstrap":
            sm = simulate_bootstrap_challenge(
                opt_ds,
                self.rules,
                pol,
                n_paths=min(self.n_bootstrap, 1000),
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unsupported period '{period}'.")
        payload = {
            "period": period,
            "policy": pol.label(),
            "metrics": sm.to_dict(),
        }
        return self._record_tool(
            "simulate_policy",
            {"policy_json": policy_json, "period": period},
            compact_json(payload),
        )

    def tool_optimise(
        self,
        objective: ObjectiveName = "fastest",
        max_fail: float = 0.20,
    ) -> str:
        self.objective = objective
        self.max_fail = max_fail
        self.opt_result = run_honest_optimisation(
            dataset=self.dataset,
            rules=self.rules,
            objective=objective,
            max_fail=max_fail,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed,
            max_lose_funded=self.max_lose_funded,
        )
        res = self.opt_result
        payload = {
            "objective": res.objective,
            "max_fail": res.max_fail,
            "feasible_found": res.feasible_found,
            "best_policy": res.best_policy.label(),
            "funded": res.best_policy.funded_label(),
            "funded_risk": res.funded_risk_line(),
            "optimise": _brief(res.opt_summary.to_dict()),
            "validate": _brief(res.val_summary.to_dict()),
            "bootstrap": _brief(res.boot_summary.to_dict()),
            "funded_projection": {
                "p_lose_opt": round(res.funded_summary.p_lose_account, 4),
                "p_lose_val": round(res.funded_val_summary.p_lose_account, 4)
                if res.funded_val_summary is not None
                else None,
                "monthly_payout_mean_usd": round(res.funded_summary.monthly_payout_mean_usd, 2),
            },
            "warnings": res.warnings,
        }
        return self._record_tool(
            "optimise",
            {"objective": objective, "max_fail": max_fail},
            compact_json(payload),
        )

    def tool_compare_policies(self, policies_json: str) -> str:
        raw = json.loads(policies_json)
        if not isinstance(raw, list) or not raw:
            raise ValueError("policies_json must be a non-empty JSON array of policy objects.")
        opt_ds, val_ds = self.dataset.split_chronological(0.6)
        base = self.active_policy()
        rows = []
        for idx, item in enumerate(raw[:6]):
            if not isinstance(item, dict):
                raise ValueError("Each policy must be a JSON object of SizingPolicy fields.")
            pol = _parse_policy_json(json.dumps(item), base)
            row: dict[str, Any] = {"idx": idx, "label": pol.label()}
            for split_name, ds in (("optimise", opt_ds), ("validate", val_ds)):
                sm = simulate_rolling_starts(ds, self.rules, pol)
                row[split_name] = {
                    "pass": round(sm.pass_rate, 3),
                    "breach": round(sm.fail_rate, 3),
                    "restart": round(sm.restart_rate, 3),
                    "days_med": round(sm.days_median, 1) if sm.days_median is not None else None,
                    "fees": round(sm.expected_fees, 0) if np.isfinite(sm.expected_fees) else None,
                }
            rows.append(row)
        return self._record_tool(
            "compare_policies",
            {"n_policies": len(rows)},
            compact_json(
                {
                    "note": "Rank on 'optimise' (first 60%); 'validate' (last 40%) is a holdout check only.",
                    "results": rows,
                }
            ),
        )

    def tool_funded_projection(self, policy_json: str) -> str:
        pol = _parse_policy_json(policy_json, self.active_policy())
        opt_ds, _ = self.dataset.split_chronological(0.6)
        fn = simulate_funded_bootstrap(
            opt_ds,
            self.rules,
            pol,
            n_paths=min(self.n_bootstrap, 1000),
            seed=self.seed + 1,
        )
        payload = {
            "policy": pol.label(),
            "funded_risk_pct": pol.funded_risk_pct,
            "keep_buffer_pct": pol.keep_buffer_pct,
            "cppi_cap": pol.cppi_cap,
            "cppi_c0": pol.cppi_c0,
            "projection": fn.to_dict(),
        }
        return self._record_tool(
            "funded_projection",
            {"policy_json": policy_json},
            compact_json(payload),
        )

    def tool_explain_failures(self, policy_json: str) -> str:
        pol = _parse_policy_json(policy_json, self.active_policy())
        _, val_ds = self.dataset.split_chronological(0.6)
        attr = analyze_failures(val_ds, self.rules, pol)
        payload = {
            "period": "validate_40pct",
            "policy": pol.label(),
            **attr.to_dict(),
        }
        return self._record_tool(
            "explain_failures",
            {"policy_json": policy_json},
            compact_json(payload),
        )


def _brief(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = ("n_finished", "pass_rate", "fail_rate", "restart_rate", "days_median", "expected_fees")
    return {k: metrics.get(k) for k in keep}


def _parse_policy_json(policy_json: str, default_policy: SizingPolicy) -> SizingPolicy:
    text = policy_json.strip()
    if not text or text == "{}":
        return default_policy
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("policy_json must decode to a JSON object.")
    merged = default_policy.model_dump(mode="json")
    merged.update(raw)
    return SizingPolicy.model_validate(merged)
