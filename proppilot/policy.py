"""Position sizing, daily risk guard, restart, and funded payout policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from proppilot.rules import PropRules


class FundedPolicy(BaseModel):
    """Funded-phase sizing, buffer retention, and CPPI multiplier configuration."""

    risk_pct: float = Field(default=0.5, description="Base risk per trade in % during funded phase.")
    keep_buffer_pct: float = Field(
        default=4.0,
        description="Profit buffer (% above initial balance) kept in the account before paying out.",
    )
    cppi_cap: float = Field(
        default=1.5,
        description="Maximum CPPI risk multiplier (1.0 = CPPI may only cut risk, never raise it).",
    )
    cppi_c0: Optional[float] = Field(
        default=None,
        description=(
            "CPPI cushion (% of initial balance above the max-loss floor) at which the multiplier "
            "is 1.0: multiplier = min(cppi_cap, cushion_pct / cppi_c0). Null disables CPPI."
        ),
    )

    @model_validator(mode="after")
    def _validate_funded(self) -> "FundedPolicy":
        if self.risk_pct <= 0:
            raise ValueError("funded risk_pct must be > 0.")
        if self.keep_buffer_pct < 0:
            raise ValueError("keep_buffer_pct must be >= 0.")
        if self.cppi_cap < 1.0:
            raise ValueError("cppi_cap must be >= 1.0.")
        if self.cppi_c0 is not None and self.cppi_c0 <= 0:
            raise ValueError("cppi_c0 must be > 0 when set.")
        return self


class SizingPolicy(BaseModel):
    """Complete evaluation + funded position-sizing and guard policy."""

    risk_pct: float = Field(default=0.7, description="Base evaluation risk per trade (%).")
    risk_basis: Literal["current_balance", "initial_balance"] = Field(
        default="current_balance",
        description="Balance reference used to compute trade dollar risk.",
    )
    near_frac: Optional[float] = Field(
        default=0.75,
        description="Fraction of phase target profit where near-target risk reduction engages.",
    )
    near_mult: float = Field(
        default=0.5,
        description="Risk multiplier applied once progress >= near_frac.",
    )
    behind_dd_pct: Optional[float] = Field(
        default=None,
        description="Optional drawdown (%) from HWM at which behind_mult engages.",
    )
    behind_mult: float = Field(
        default=0.5,
        description="Risk multiplier applied when drawdown >= behind_dd_pct.",
    )
    own_daily_stop_pct: float = Field(
        default=3.0,
        description="Trader's own daily loss+open-risk cap (%) strictly tighter than firm limit.",
    )
    per_symbol_daily_cap_pct: Optional[float] = Field(
        default=None,
        description="Optional daily reserved risk cap (%) per individual symbol.",
    )
    max_open_risk_pct: Optional[float] = Field(
        default=None,
        description="Optional maximum total concurrent open risk (%) across all symbols.",
    )
    restart_after_days: Optional[int] = Field(
        default=90,
        description="Abandon unfinished evaluation after N days and purchase a new attempt.",
    )
    funded_risk_pct: float = Field(
        default=0.4,
        description="Base risk per trade (%) in the funded phase.",
    )
    keep_buffer_pct: float = Field(
        default=4.0,
        description="Kept profit buffer (%) above initial balance in the funded phase.",
    )
    cppi_cap: float = Field(
        default=1.5,
        description="Maximum CPPI multiplier in the funded phase.",
    )
    cppi_c0: Optional[float] = Field(
        default=None,
        description=(
            "Funded CPPI: cushion % above the max-loss floor at which the multiplier is 1.0 "
            "(multiplier = min(cppi_cap, cushion_pct / cppi_c0)); null disables CPPI."
        ),
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> "SizingPolicy":
        if self.risk_pct <= 0:
            raise ValueError("risk_pct must be > 0.")
        if self.near_frac is not None and not (0.0 < self.near_frac <= 1.0):
            raise ValueError("near_frac must be in (0, 1].")
        if self.near_mult <= 0:
            raise ValueError("near_mult must be > 0.")
        if self.behind_dd_pct is not None and self.behind_dd_pct <= 0:
            raise ValueError("behind_dd_pct must be > 0 when set.")
        if self.behind_mult <= 0:
            raise ValueError("behind_mult must be > 0.")
        if self.own_daily_stop_pct <= 0:
            raise ValueError("own_daily_stop_pct must be > 0.")
        if self.per_symbol_daily_cap_pct is not None and self.per_symbol_daily_cap_pct <= 0:
            raise ValueError("per_symbol_daily_cap_pct must be > 0 when set.")
        if self.max_open_risk_pct is not None and self.max_open_risk_pct <= 0:
            raise ValueError("max_open_risk_pct must be > 0 when set.")
        if self.restart_after_days is not None and self.restart_after_days <= 0:
            raise ValueError("restart_after_days must be > 0 when set.")
        if self.funded_risk_pct <= 0:
            raise ValueError("funded_risk_pct must be > 0.")
        if self.keep_buffer_pct < 0:
            raise ValueError("keep_buffer_pct must be >= 0.")
        if self.cppi_cap < 1.0:
            raise ValueError("cppi_cap must be >= 1.0.")
        if self.cppi_c0 is not None and self.cppi_c0 <= 0:
            raise ValueError("cppi_c0 must be > 0 when set.")
        return self

    def funded_config(self) -> FundedPolicy:
        """Extract the funded-phase portion as a FundedPolicy."""
        return FundedPolicy(
            risk_pct=self.funded_risk_pct,
            keep_buffer_pct=self.keep_buffer_pct,
            cppi_cap=self.cppi_cap,
            cppi_c0=self.cppi_c0,
        )

    def label(self) -> str:
        """Return a compact, human-readable summary of the sizing policy."""
        parts = [f"risk {self.risk_pct:.2g}%/trade"]
        if self.near_frac is not None and abs(self.near_mult - 1.0) > 1e-6:
            parts.append(f"{self.near_mult:.2g}x at {int(round(self.near_frac * 100))}% of target")
        if self.behind_dd_pct is not None and abs(self.behind_mult - 1.0) > 1e-6:
            parts.append(f"{self.behind_mult:.2g}x at -{self.behind_dd_pct:.2g}% DD")
        parts.append(f"daily stop {self.own_daily_stop_pct:.2g}%")
        if self.restart_after_days is not None:
            parts.append(f"restart {self.restart_after_days}d")
        else:
            parts.append("no restart")
        return ", ".join(parts)

    def funded_label(self) -> str:
        """Return a compact summary of the funded-phase settings."""
        cppi = (
            f"CPPI 1x at {self.cppi_c0:.2g}% cushion, cap {self.cppi_cap:.2g}x"
            if self.cppi_c0 is not None
            else "no CPPI"
        )
        return f"funded {self.funded_risk_pct:.2g}%/trade, {cppi}, keep {self.keep_buffer_pct:.2g}% buffer"

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SizingPolicy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Policy YAML must decode to a mapping: {path}")
        return cls.model_validate(raw)

    @classmethod
    def from_json(cls, text: str) -> "SizingPolicy":
        return cls.model_validate_json(text)


def default_policy_for_rules(rules: PropRules) -> SizingPolicy:
    """Build a sensible baseline SizingPolicy respecting firm daily/max limits."""
    daily_stop = round(min(rules.daily_loss_pct * 0.65, max(0.5, rules.daily_loss_pct - 0.5)), 2)
    base_risk = round(min(0.7, daily_stop * 0.35), 2)
    if rules.max_risk_per_trade_pct is not None:
        base_risk = min(base_risk, rules.max_risk_per_trade_pct)
    funded_risk = round(min(0.4, base_risk * 0.7), 2)

    phase_max_days = [p.max_days for p in rules.phases if p.max_days is not None]
    restart_days = None if phase_max_days else 90

    return SizingPolicy(
        risk_pct=base_risk,
        near_frac=0.75,
        near_mult=0.5,
        own_daily_stop_pct=daily_stop,
        max_open_risk_pct=daily_stop,
        restart_after_days=restart_days,
        funded_risk_pct=funded_risk,
        keep_buffer_pct=4.0,
        cppi_cap=1.5,
        cppi_c0=float(rules.max_loss_pct),
    )


def build_default_grid(rules: PropRules) -> list[SizingPolicy]:
    """Generate a sensible ~200-policy search grid calibrated to PropRules."""
    risk_candidates = [0.3, 0.5, 0.7, 1.0, 1.25, 1.5]
    if rules.max_risk_per_trade_pct is not None:
        risk_candidates = [r for r in risk_candidates if r <= rules.max_risk_per_trade_pct + 1e-9]
        if not risk_candidates:
            risk_candidates = [rules.max_risk_per_trade_pct]

    near_mult_candidates = [0.35, 0.5, 0.75, 1.0]

    dlim = rules.daily_loss_pct
    daily_stops = sorted(
        {
            round(max(0.4, dlim * 0.5), 2),
            round(max(0.5, dlim * 0.7), 2),
            round(max(0.6, min(dlim - 0.25, dlim * 0.85)), 2),
        }
    )

    phase_max_days = [p.max_days for p in rules.phases if p.max_days is not None]
    if phase_max_days:
        min_max_days = min(phase_max_days)
        restart_candidates: list[Optional[int]] = [
            d for d in (max(15, min_max_days // 2), None) if d is None or d <= min_max_days
        ]
    elif not rules.phases:
        # Instant funded: no evaluation timeout needed.
        restart_candidates = [None]
    else:
        restart_candidates = [60, 90, 120]

    grid: list[SizingPolicy] = []
    for risk in risk_candidates:
        for near_mult in near_mult_candidates:
            for daily_stop in daily_stops:
                if risk > daily_stop + 1e-9:
                    continue
                for restart in restart_candidates:
                    funded_risk = round(max(0.2, min(risk * 0.45, dlim * 0.08)), 2)
                    if rules.max_risk_per_trade_pct is not None:
                        funded_risk = min(funded_risk, rules.max_risk_per_trade_pct)
                    grid.append(
                        SizingPolicy(
                            risk_pct=risk,
                            near_frac=0.75 if near_mult < 1.0 else None,
                            near_mult=near_mult,
                            own_daily_stop_pct=daily_stop,
                            max_open_risk_pct=daily_stop,
                            restart_after_days=restart,
                            funded_risk_pct=funded_risk,
                            keep_buffer_pct=4.0,
                            cppi_cap=1.5,
                            cppi_c0=float(rules.max_loss_pct),
                        )
                    )
    return grid


FUNDED_RISK_CANDIDATES = (0.25, 0.5, 0.75, 1.0)


def build_funded_grid(rules: PropRules, base: SizingPolicy) -> list[SizingPolicy]:
    """Funded-phase variants of `base` (same evaluation parameters and entry guards).

    Varies funded risk per trade (0.25-1.0%, capped by the firm per-trade cap and the policy's
    own daily stop) with CPPI off / on (c0 = max_loss_pct, i.e. 1x at the starting cushion,
    cap 1.5x). Keep buffer stays at 4% above the initial balance.
    """
    variants: list[SizingPolicy] = []
    for funded_risk in FUNDED_RISK_CANDIDATES:
        if rules.max_risk_per_trade_pct is not None and funded_risk > rules.max_risk_per_trade_pct + 1e-9:
            continue
        if funded_risk > base.own_daily_stop_pct + 1e-9:
            continue
        for c0 in (None, float(rules.max_loss_pct)):
            variants.append(
                base.model_copy(
                    update={
                        "funded_risk_pct": funded_risk,
                        "keep_buffer_pct": 4.0,
                        "cppi_cap": 1.5,
                        "cppi_c0": c0,
                    }
                )
            )
    if not variants:
        variants.append(base.model_copy())
    return variants
