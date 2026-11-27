"""Reproducible SYNTHETIC multi-symbol crypto trade history generator.

The generated history imitates a *decent but ordinary* crypto perpetuals system, not a
money printer (seed 42, 4 years: ~64 trades/month, ~39% win rate, ~+0.09R expectancy):

- 6 symbols (BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT, ARBUSDT)
- 3 persistent Markov regimes (trend, chop, high-volatility cascade) that create multi-week
  good and bad spells and realistic losing streaks (high-teens consecutive losers occur)
- a daily market factor that correlates same-day outcomes across symbols (much stronger on
  cascade days, which is what breaks daily-loss limits)
- winners average ~1.7R (0.45R partial exits up to 5R+ runners), losers mostly a full -1R stop
  with fee/slippage scatter plus some early exits between -0.8R and -0.1R
- a small win-rate decay in the last 40% of the history (edge decay)

Every file produced here is SYNTHETIC and only meant for demos and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfinv

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "ARBUSDT")
_SYMBOL_WEIGHTS = (0.28, 0.24, 0.18, 0.12, 0.10, 0.08)

# Daily regime transition matrix: 0 = trend, 1 = chop, 2 = high-volatility cascade.
_REGIME_TRANSITIONS = np.array(
    [
        [0.92, 0.06, 0.02],
        [0.07, 0.89, 0.04],
        [0.08, 0.17, 0.75],
    ],
    dtype=np.float64,
)

# Per-regime parameters: (win_probability, mean_trades_per_day, same_day_correlation)
_REGIME_PARAMS = {
    0: (0.44, 2.2, 0.35),
    1: (0.34, 1.9, 0.30),
    2: (0.29, 2.6, 0.65),
}

_LATE_WIN_PROB_DECAY = 0.01


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for deterministic synthetic crypto trade generation."""

    start_date: str = "2022-01-01"
    days: int = 1461  # ~4 years (includes the 2024 leap day)
    seed: int = 42


def generate_synthetic_trades(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a reproducible multi-symbol SYNTHETIC crypto trade DataFrame."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    start_ts = pd.Timestamp(cfg.start_date, tz="UTC")
    rows: list[dict[str, object]] = []
    decay_from_day = int(cfg.days * 0.6)

    regime = 0
    for day_offset in range(cfg.days):
        regime = int(rng.choice(3, p=_REGIME_TRANSITIONS[regime]))
        win_prob, mean_trades, corr = _REGIME_PARAMS[regime]
        if day_offset > decay_from_day:
            win_prob -= _LATE_WIN_PROB_DECAY

        n_today = int(rng.poisson(mean_trades))
        if n_today == 0:
            continue

        # One market factor per day drives correlated outcomes across symbols.
        market_shock = float(rng.normal(0.0, 1.0))
        # P(latent > threshold) = win_prob for a standard normal latent variable.
        threshold = float(-np.sqrt(2.0) * erfinv(2.0 * win_prob - 1.0))

        day_start = start_ts + pd.Timedelta(days=day_offset)
        entry_minutes = np.sort(rng.uniform(0.0, 24 * 60 - 1.0, size=n_today))
        chosen_symbols = rng.choice(SYMBOLS, size=n_today, p=_SYMBOL_WEIGHTS)
        day_side = "LONG" if rng.random() < 0.55 else "SHORT"

        for k in range(n_today):
            side = day_side if rng.random() < 0.75 else ("SHORT" if day_side == "LONG" else "LONG")
            entry_dt = day_start + pd.Timedelta(minutes=float(entry_minutes[k]))
            # Holding time: median ~9 hours, from 30 minutes up to 5 days (overlapping positions).
            hold_minutes = float(np.clip(rng.lognormal(np.log(540.0), 0.85), 30.0, 5 * 1440.0))
            exit_dt = entry_dt + pd.Timedelta(minutes=hold_minutes)

            latent = corr * market_shock + np.sqrt(1.0 - corr**2) * float(rng.normal(0.0, 1.0))
            if latent > threshold:
                r_val = 0.45 + float(rng.gamma(2.2, 0.55))
            elif rng.random() < 0.15:
                r_val = float(rng.uniform(-0.8, -0.1))  # early exit before the stop
            else:
                r_val = float(np.clip(rng.normal(-1.02, 0.04), -1.25, -0.9))  # stop + costs

            rows.append(
                {
                    "entry_time": entry_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "exit_time": exit_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symbol": str(chosen_symbols[k]),
                    "side": side,
                    "R": round(r_val, 4),
                }
            )

    df = pd.DataFrame(rows, columns=["entry_time", "exit_time", "symbol", "side", "R"])
    return df.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)


def write_sample_trades(
    output_path: str | Path = "examples/trades_sample.csv",
    config: SyntheticConfig | None = None,
) -> Path:
    """Generate and save the default SYNTHETIC sample trades CSV."""
    df = generate_synthetic_trades(config)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, lineterminator="\n")
    return target


if __name__ == "__main__":
    out = write_sample_trades()
    print(f"Wrote SYNTHETIC trades to {out}")
