"""Prop-firm evaluation and funded account rules schema, presets, and YAML I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


class PhaseRule(BaseModel):
    """Single evaluation phase definition."""

    name: str = Field(default="Phase", description="Human-readable phase label.")
    target_pct: float = Field(
        ..., description="Profit target percentage required to pass this phase (e.g. 8.0 for +8%)."
    )
    min_days: int = Field(
        default=0,
        description="Minimum distinct trading days (days with >= 1 entry) required in this phase.",
    )
    max_days: Optional[int] = Field(
        default=None,
        description="Maximum calendar days allowed for this phase, or null if unlimited.",
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PhaseRule":
        if self.target_pct <= 0:
            raise ValueError("Phase target_pct must be > 0.")
        if self.min_days < 0:
            raise ValueError("Phase min_days must be >= 0.")
        if self.max_days is not None and self.max_days <= 0:
            raise ValueError("Phase max_days must be > 0 when set.")
        return self


class PropRules(BaseModel):
    """Generic prop-trading challenge and funded account rulebook."""

    name: str = Field(default="Custom Rules", description="Rulebook identifier.")
    phases: list[PhaseRule] = Field(
        default_factory=list,
        description="Ordered evaluation phases; empty list indicates instant funding.",
    )
    daily_loss_pct: float = Field(
        ...,
        description="Maximum allowed daily loss percentage relative to day-start reference.",
    )
    daily_basis: Literal["day_start_balance", "day_start_max_balance_equity"] = Field(
        default="day_start_balance",
        description="Reference used at daily reset hour to anchor the daily loss limit.",
    )
    daily_reset: int = Field(
        default=0,
        description="UTC hour (0-23) when the daily loss counter resets.",
    )
    max_loss_pct: float = Field(
        ...,
        description="Maximum overall drawdown limit percentage.",
    )
    max_loss_type: Literal["static", "trailing_closed", "trailing_equity"] = Field(
        default="static",
        description="Drawdown floor mechanics: static, trailing_closed, or trailing_equity.",
    )
    trailing_lock_at_initial: bool = Field(
        default=False,
        description=(
            "Trailing max-loss only: true if the trailing floor stops rising once it reaches the "
            "initial balance (floor = min(initial_balance, HWM * (1 - max_loss_pct/100))); false if "
            "it keeps trailing the high-water mark forever. Ignored for static max loss."
        ),
    )
    consistency_pct: Optional[float] = Field(
        default=None,
        description="Optional max share (%) a single best day may represent of total profit before pass/payout.",
    )
    fee: float = Field(
        ...,
        description="Challenge / account evaluation fee in USD for the configured account_size.",
    )
    profit_split: float = Field(
        default=80.0,
        description="Trader share of funded payouts in percent (e.g. 80.0 for 80%).",
    )
    payout_frequency_days: int = Field(
        default=14,
        description="Calendar days between eligible payout checks in the funded phase.",
    )
    account_size: float = Field(
        default=100_000.0,
        description="Nominal account balance in USD.",
    )
    max_risk_per_trade_pct: Optional[float] = Field(
        default=None,
        description="Optional firm cap on risk per single trade in percent, or null.",
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Unclear, missing, or ambiguous clauses flagged during rule parsing.",
    )

    @model_validator(mode="after")
    def _validate_rules(self) -> "PropRules":
        if self.daily_loss_pct <= 0:
            raise ValueError("daily_loss_pct must be > 0.")
        if not 0 <= self.daily_reset <= 23:
            raise ValueError("daily_reset must be an integer UTC hour in 0..23.")
        if self.max_loss_pct <= 0:
            raise ValueError("max_loss_pct must be > 0.")
        if self.consistency_pct is not None and not (0 < self.consistency_pct <= 100):
            raise ValueError("consistency_pct must be in (0, 100] when set.")
        if self.fee < 0:
            raise ValueError("fee must be >= 0.")
        if not (0 < self.profit_split <= 100):
            raise ValueError("profit_split must be in (0, 100].")
        if self.payout_frequency_days <= 0:
            raise ValueError("payout_frequency_days must be > 0.")
        if self.account_size <= 0:
            raise ValueError("account_size must be > 0.")
        if self.max_risk_per_trade_pct is not None and self.max_risk_per_trade_pct <= 0:
            raise ValueError("max_risk_per_trade_pct must be > 0 when set.")
        return self

    def with_account_size(self, account_size: float) -> "PropRules":
        """Return a copy scaled proportionally to a new account size."""
        if account_size <= 0:
            raise ValueError("account_size must be > 0.")
        if abs(account_size - self.account_size) < 1e-9:
            return self.model_copy(deep=True)
        ratio = account_size / self.account_size
        return self.model_copy(
            update={
                "account_size": float(account_size),
                "fee": round(self.fee * ratio, 2),
            },
            deep=True,
        )

    def to_yaml(self) -> str:
        """Serialize rules cleanly to YAML."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def save_yaml(self, path: str | Path) -> Path:
        """Write rules YAML to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml(), encoding="utf-8")
        return target

    @property
    def approximation_notes(self) -> list[str]:
        """Return methodological notes when tick-level rule variants are approximated."""
        notes: list[str] = []
        if self.daily_basis == "day_start_max_balance_equity":
            notes.append(
                "daily_basis='day_start_max_balance_equity' is approximated using closed balance "
                "at reset because trade-level logs lack intraday mark-to-market equity."
            )
        if self.max_loss_type == "trailing_equity":
            notes.append(
                "max_loss_type='trailing_equity' is approximated using closed-trade high-water "
                "mark at exits because trade-level logs lack intraday equity ticks."
            )
        return notes


def list_presets() -> list[str]:
    """Return available built-in rule preset names."""
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def load_preset(name: str, account_size: float | None = None) -> PropRules:
    """Load a built-in preset by name (e.g. 'two_step_classic')."""
    stem = name[:-5] if name.endswith(".yaml") else name
    path = PRESETS_DIR / f"{stem}.yaml"
    if not path.exists():
        available = ", ".join(list_presets())
        raise FileNotFoundError(f"Unknown preset '{name}'. Available presets: {available}")
    return load_rules(path, account_size=account_size)


def load_rules(source: str | Path, account_size: float | None = None) -> PropRules:
    """Load PropRules from a preset name or a YAML file path."""
    path = Path(source)
    if not path.exists():
        if isinstance(source, str) and (PRESETS_DIR / f"{source}.yaml").exists():
            path = PRESETS_DIR / f"{source}.yaml"
        else:
            raise FileNotFoundError(f"Rules file or preset not found: {source}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Rules YAML must decode to a mapping: {path}")
    rules = PropRules.model_validate(raw)
    if account_size is not None:
        rules = rules.with_account_size(account_size)
    return rules


def parse_rules_yaml_text(yaml_text: str, account_size: float | None = None) -> PropRules:
    """Parse PropRules directly from a YAML string."""
    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("Rules YAML must decode to a mapping.")
    rules = PropRules.model_validate(raw)
    if account_size is not None:
        rules = rules.with_account_size(account_size)
    return rules
