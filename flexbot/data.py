from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

from . import context


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame | None:
    try:
        raw = context.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as exc:
        logging.error("Erro fetch_ohlcv %s %s: %s", symbol, timeframe, exc)
        return None
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df


def fetch_ohlcv_paginated(symbol: str, timeframe: str, bars_needed: int) -> pd.DataFrame | None:
    try:
        tf_ms = int(context.exchange.parse_timeframe(timeframe) * 1000)
    except Exception:
        tf_ms = 60_000
    since = max(0, context.exchange.milliseconds() - (bars_needed + 5) * tf_ms)
    all_rows = []
    attempts = 0
    while len(all_rows) < bars_needed and attempts < 50:
        limit = min(1000, bars_needed - len(all_rows))
        try:
            batch = context.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as exc:
            logging.error("Erro fetch_ohlcv_paginated %s %s: %s", symbol, timeframe, exc)
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + tf_ms
        attempts += 1
        if len(batch) < limit:
            break
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="ts", inplace=True)
    df.sort_values("ts", inplace=True)
    df = df.tail(bars_needed)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df


def fetch_backtest_series(symbol: str, timeframe: str, candles_needed: int) -> Tuple[pd.DataFrame | None, Dict[str, object]]:
    candles_needed = min(candles_needed, context.max_backtest_candles_per_tf)
    df = fetch_ohlcv_paginated(symbol, timeframe, candles_needed)
    if df is None or df.empty:
        return None, {
            "candles": 0,
            "start": None,
            "end": None,
            "requested": candles_needed,
        }
    start_ts = df.index[0]
    end_ts = df.index[-1]
    return df, {
        "candles": len(df),
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "requested": candles_needed,
    }


def timeframe_to_minutes(tf: str) -> int:
    if tf in context.tf_minutes_map:
        return context.tf_minutes_map[tf]
    try:
        seconds = context.exchange.parse_timeframe(tf)
        return max(1, int(seconds / 60))
    except Exception as exc:
        raise ValueError(f"Timeframe não suportado: {tf}") from exc
