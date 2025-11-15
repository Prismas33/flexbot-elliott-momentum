from __future__ import annotations

from typing import Dict

import pandas as pd

from . import context
from .data import fetch_ohlcv


def determine_trend_direction(df: pd.DataFrame | None, fast: int = context.trend_fast_span, slow: int = context.trend_slow_span) -> str:
    if df is None or len(df) < slow + 5:
        return "unknown"
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
    if diff > context.trend_flat_tolerance:
        return "up"
    if diff < -context.trend_flat_tolerance:
        return "down"
    return "flat"


def compute_symbol_bias(symbol: str) -> Dict[str, str]:
    bias = {}
    parent_cache: Dict[str, pd.DataFrame | None] = {}
    for tf in context.timeframes:
        parent = context.get_bias_parent(tf)
        if parent is None:
            bias[tf] = "both"
            continue
        if parent not in parent_cache:
            parent_cache[parent] = fetch_ohlcv(symbol, parent, limit=context.lookback)
        bias[tf] = determine_trend_direction(parent_cache[parent])
    return bias


def bias_allows_long(tf: str, bias_map: Dict[str, str]) -> bool:
    direction = bias_map.get(tf, "both")
    return direction != "down"


def bias_allows_short(tf: str, bias_map: Dict[str, str]) -> bool:
    direction = bias_map.get(tf, "both")
    return direction != "up"


def bias_from_slices(tf: str, ts, slice_store):
    parent = context.get_bias_parent(tf)
    if parent is None:
        return "both"
    parent_df = slice_store.get(parent)
    if parent_df is None or parent_df.empty:
        return "unknown"
    df = parent_df[parent_df.index <= ts].tail(context.lookback)
    if df is None or df.empty:
        return "unknown"
    return determine_trend_direction(df)
