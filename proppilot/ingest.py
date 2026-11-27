"""Trade history ingestion, validation, chronological splitting, and event indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
HOUR_MS = 3_600_000

REQUIRED_COLUMNS = ("entry_time", "exit_time", "symbol", "R")

EV_EXIT_NORMAL = 0
EV_ENTRY = 1
EV_EXIT_INSTANT = 2


def _parse_utc_ms(series: pd.Series, column_name: str) -> np.ndarray:
    """Parse timestamps (ISO8601 strings, datetimes, or epoch values) into UTC int64 ms."""
    if pd.api.types.is_numeric_dtype(series):
        vals = series.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column '{column_name}' contains non-finite numeric timestamps.")
        # Heuristic: values < 1e11 are epoch seconds; >= 1e11 are epoch milliseconds.
        if np.median(np.abs(vals)) < 1e11:
            return np.rint(vals * 1000.0).astype(np.int64)
        return np.rint(vals).astype(np.int64)

    try:
        dt = pd.to_datetime(series, utc=True, format="ISO8601")
    except (ValueError, TypeError):
        try:
            dt = pd.to_datetime(series, utc=True, format="mixed")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Failed to parse UTC timestamps in column '{column_name}': {exc}") from exc

    if dt.isna().any():
        raise ValueError(f"Column '{column_name}' contains unparseable/null timestamps.")

    return dt.dt.as_unit("ms").astype("int64").to_numpy(copy=True)


@dataclass(frozen=True)
class TradeDataset:
    """Immutable, sorted, precomputed numpy representation of historical trades."""

    entry_ms: np.ndarray
    exit_ms: np.ndarray
    r_vals: np.ndarray
    symbol_ids: np.ndarray
    symbols: tuple[str, ...]
    sides: tuple[str, ...]
    _ev_ms: np.ndarray = field(init=False, repr=False)
    _ev_kind: np.ndarray = field(init=False, repr=False)
    _ev_trade_idx: np.ndarray = field(init=False, repr=False)
    _entry_to_ev: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.entry_ms)
        if n == 0:
            raise ValueError("TradeDataset cannot be empty.")
        if not (len(self.exit_ms) == len(self.r_vals) == len(self.symbol_ids) == len(self.sides) == n):
            raise ValueError("All trade arrays must have identical length.")
        if np.any(self.exit_ms < self.entry_ms):
            raise ValueError("Found trade(s) with exit_time < entry_time.")
        if not np.all(np.isfinite(self.r_vals)):
            raise ValueError("Found non-finite R values in trades.")
        if n > 1 and np.any(np.diff(self.entry_ms) < 0):
            raise ValueError("TradeDataset entry_ms must be sorted ascending.")

        trade_idx = np.arange(n, dtype=np.int32)
        is_instant = self.exit_ms == self.entry_ms

        ev_ms = np.concatenate([self.entry_ms, self.exit_ms])
        ev_kind = np.concatenate(
            [
                np.full(n, EV_ENTRY, dtype=np.uint8),
                np.where(is_instant, EV_EXIT_INSTANT, EV_EXIT_NORMAL).astype(np.uint8),
            ]
        )
        ev_trade = np.concatenate([trade_idx, trade_idx])

        # Sort chronologically: timestamp -> event priority (normal exit=0, entry=1, instant exit=2) -> trade_idx
        order = np.lexsort((ev_trade, ev_kind, ev_ms))
        ev_ms_sorted = np.ascontiguousarray(ev_ms[order], dtype=np.int64)
        ev_kind_sorted = np.ascontiguousarray(ev_kind[order], dtype=np.uint8)
        ev_trade_sorted = np.ascontiguousarray(ev_trade[order], dtype=np.int32)

        entry_to_ev = np.empty(n, dtype=np.int32)
        entry_mask = ev_kind_sorted == EV_ENTRY
        entry_to_ev[ev_trade_sorted[entry_mask]] = np.flatnonzero(entry_mask).astype(np.int32)

        object.__setattr__(self, "_ev_ms", ev_ms_sorted)
        object.__setattr__(self, "_ev_kind", ev_kind_sorted)
        object.__setattr__(self, "_ev_trade_idx", ev_trade_sorted)
        object.__setattr__(self, "_entry_to_ev", entry_to_ev)

    def __len__(self) -> int:
        return int(len(self.entry_ms))

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def start_ms(self) -> int:
        return int(self.entry_ms[0])

    @property
    def end_ms(self) -> int:
        return int(self.exit_ms.max())

    @property
    def events(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pre-sorted (ev_ms, ev_kind, ev_trade_idx) arrays."""
        return self._ev_ms, self._ev_kind, self._ev_trade_idx

    def rolling_start_trade_indices(self) -> np.ndarray:
        """Return the first trade index for every distinct UTC calendar day in entry_ms."""
        days = self.entry_ms // DAY_MS
        _, first_indices = np.unique(days, return_index=True)
        return np.sort(first_indices).astype(np.int32)

    def slice_trades(self, start: int, end: int) -> "TradeDataset":
        """Slice trades by positional index [start:end], preserving symbol dictionary."""
        if start < 0 or end > len(self) or start >= end:
            raise ValueError(f"Invalid slice [{start}:{end}] for {len(self)} trades.")
        return TradeDataset(
            entry_ms=np.ascontiguousarray(self.entry_ms[start:end], dtype=np.int64),
            exit_ms=np.ascontiguousarray(self.exit_ms[start:end], dtype=np.int64),
            r_vals=np.ascontiguousarray(self.r_vals[start:end], dtype=np.float64),
            symbol_ids=np.ascontiguousarray(self.symbol_ids[start:end], dtype=np.int32),
            symbols=self.symbols,
            sides=self.sides[start:end],
        )

    def split_chronological(self, train_frac: float = 0.6) -> tuple["TradeDataset", "TradeDataset"]:
        """Split dataset strictly chronologically into (optimise_slice, validate_slice)."""
        if not (0.1 <= train_frac <= 0.9):
            raise ValueError("train_frac must be between 0.1 and 0.9.")
        n = len(self)
        if n < 2:
            raise ValueError("Need at least 2 trades to split into train/validation.")
        split_idx = max(1, min(n - 1, int(round(n * train_frac))))
        # Avoid splitting mid-timestamp if multiple trades share the exact boundary entry_ms.
        boundary_ms = self.entry_ms[split_idx]
        while split_idx < n - 1 and self.entry_ms[split_idx - 1] == boundary_ms:
            split_idx += 1
        return self.slice_trades(0, split_idx), self.slice_trades(split_idx, n)

    def max_concurrent_open(self) -> int:
        """Compute maximum number of overlapping open trades at any instant."""
        delta = np.where(self._ev_kind == EV_ENTRY, 1, -1)
        return int(np.max(np.cumsum(delta)))

    def summary_stats(self) -> dict[str, Any]:
        """Compute descriptive strategy statistics without assuming position sizing."""
        n = len(self)
        r = self.r_vals
        wins = r[r > 0]
        losses = r[r < 0]
        win_rate = float(np.mean(r > 0))
        avg_win_r = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss_r = float(np.mean(losses)) if len(losses) else 0.0
        expectancy_r = float(np.mean(r))
        span_days = max(1.0, (self.end_ms - self.start_ms) / DAY_MS)
        trades_per_month = float(n / (span_days / 30.4375))
        start_iso = pd.Timestamp(self.start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
        end_iso = pd.Timestamp(self.end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
        profit_factor = (
            float(np.sum(wins) / abs(np.sum(losses))) if len(losses) and np.sum(losses) != 0 else float("inf")
        )
        return {
            "n_trades": n,
            "n_symbols": self.n_symbols,
            "symbols": list(self.symbols),
            "start_utc": start_iso,
            "end_utc": end_iso,
            "span_days": round(span_days, 1),
            "trades_per_month": round(trades_per_month, 1),
            "win_rate": round(win_rate, 4),
            "avg_win_r": round(avg_win_r, 4),
            "avg_loss_r": round(avg_loss_r, 4),
            "expectancy_r": round(expectancy_r, 4),
            "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else None,
            "total_r": round(float(np.sum(r)), 2),
            "max_concurrent_open": self.max_concurrent_open(),
            "n_calendar_days_with_entries": int(len(self.rolling_start_trade_indices())),
        }


def trades_from_dataframe(df: pd.DataFrame) -> TradeDataset:
    """Validate and convert a raw trades DataFrame (including QuantCritic exports) to TradeDataset."""
    if df.empty:
        raise ValueError("Trades input contains zero rows.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Trades input missing required column(s): {', '.join(missing)}")

    entry_ms = _parse_utc_ms(df["entry_time"], "entry_time")
    exit_ms = _parse_utc_ms(df["exit_time"], "exit_time")
    r_vals = pd.to_numeric(df["R"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(r_vals)):
        raise ValueError("Column 'R' contains null or non-numeric values.")

    symbols_raw = df["symbol"].astype(str).str.strip()
    if (symbols_raw == "").any() or df["symbol"].isna().any():
        raise ValueError("Column 'symbol' contains empty or null values.")

    if "side" in df.columns:
        sides_raw = df["side"].fillna("").astype(str).str.strip().str.upper().tolist()
    else:
        sides_raw = [""] * len(df)

    # Sort stably by entry_time, then exit_time, preserving row order on exact ties.
    order = np.lexsort((exit_ms, entry_ms))
    entry_sorted = np.ascontiguousarray(entry_ms[order], dtype=np.int64)
    exit_sorted = np.ascontiguousarray(exit_ms[order], dtype=np.int64)
    r_sorted = np.ascontiguousarray(r_vals[order], dtype=np.float64)
    symbols_sorted = symbols_raw.to_numpy()[order]
    sides_sorted = tuple(sides_raw[i] for i in order)

    unique_symbols = tuple(sorted(set(symbols_sorted.tolist())))
    sym_to_id = {sym: idx for idx, sym in enumerate(unique_symbols)}
    symbol_ids = np.array([sym_to_id[sym] for sym in symbols_sorted], dtype=np.int32)

    return TradeDataset(
        entry_ms=entry_sorted,
        exit_ms=exit_sorted,
        r_vals=r_sorted,
        symbol_ids=symbol_ids,
        symbols=unique_symbols,
        sides=sides_sorted,
    )


def load_trades(
    source: str | Path | BinaryIO | TextIO,
    fmt: str | None = None,
) -> TradeDataset:
    """Load trade history from CSV or JSON (including QuantCritic exports)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Trades file not found: {path}")
        resolved_fmt = (fmt or path.suffix.lstrip(".")).lower()
        if resolved_fmt == "csv":
            df = pd.read_csv(path)
        elif resolved_fmt == "json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            df = _json_obj_to_dataframe(raw)
        else:
            raise ValueError(f"Unsupported trades format '{resolved_fmt}'. Expected csv or json.")
        return trades_from_dataframe(df)

    # File-like object (e.g. Streamlit UploadedFile)
    name = getattr(source, "name", "") or ""
    resolved_fmt = (fmt or Path(name).suffix.lstrip(".") or "csv").lower()
    if resolved_fmt == "csv":
        df = pd.read_csv(source)
    elif resolved_fmt == "json":
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        df = _json_obj_to_dataframe(json.loads(content))
    else:
        raise ValueError(f"Unsupported trades format '{resolved_fmt}'. Expected csv or json.")
    return trades_from_dataframe(df)


def _json_obj_to_dataframe(raw: Any) -> pd.DataFrame:
    """Convert JSON list or QuantCritic-style dict payload to a DataFrame."""
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict):
        for key in ("trades", "data", "records"):
            if key in raw and isinstance(raw[key], list):
                return pd.DataFrame(raw[key])
    raise ValueError("JSON trades payload must be a list of trade objects or a dict containing 'trades'.")
