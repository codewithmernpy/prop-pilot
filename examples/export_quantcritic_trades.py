"""Export REAL backtest trades from the sister project QuantCritic into PropPilot's CSV format.

Runs QuantCritic's `donchian_breakout` strategy with its default parameters and default
backtest configuration (next-bar-open fills, 0.05% taker fee, 2 bps slippage, perpetual
funding) on QuantCritic's bundled Binance BTCUSDT / ETHUSDT / SOLUSDT history resampled to 4h,
and writes `entry_time, exit_time, symbol, side, R` (R is net of fees, slippage and funding).

Usage (QuantCritic is only read and executed, never modified):

    python examples/export_quantcritic_trades.py --quantcritic ../quant-critic \
        --out examples/trades_quantcritic_donchian_4h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quantcritic", required=True, help="Path to the QuantCritic project root.")
    parser.add_argument("--out", default="examples/trades_quantcritic_donchian_4h.csv")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="4h")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.quantcritic).resolve()))
    from quantcritic.config import BacktestConfig
    from quantcritic.data.loader import load_market_data
    from quantcritic.engine.backtest import run_backtest
    from quantcritic.strategies.registry import get_strategy

    data = load_market_data(args.symbols, interval=args.interval)
    strategy = get_strategy("donchian_breakout")
    params = strategy.merge_params({})
    result = run_backtest(data, strategy, params, BacktestConfig(interval=args.interval))

    trades = result.trades[["entry_time", "exit_time", "symbol", "side", "R"]].copy()
    trades = trades.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)
    trades["entry_time"] = trades["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    trades["exit_time"] = trades["exit_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    trades["R"] = trades["R"].round(4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False, lineterminator="\n")
    m = result.metrics
    print(
        f"Wrote {len(trades)} REAL QuantCritic trades to {out} "
        f"(params {params}; win rate {m['win_rate'] * 100:.1f}%, avg R {m['avg_r']:+.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
