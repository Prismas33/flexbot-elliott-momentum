from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from . import context


def compute_rsi(df: pd.DataFrame, period: int = context.rsi_period) -> pd.Series:
    return RSIIndicator(df["close"], period).rsi()


def compute_atr(df: pd.DataFrame, period: int = context.atr_period) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def is_local_max(series: pd.Series, idx: int, order: int) -> bool:
    start = max(0, idx - order)
    end = min(len(series) - 1, idx + order)
    center = series.iloc[idx]
    window = series.iloc[start : end + 1]
    return center >= window.max()


def is_local_min(series: pd.Series, idx: int, order: int) -> bool:
    start = max(0, idx - order)
    end = min(len(series) - 1, idx + order)
    center = series.iloc[idx]
    window = series.iloc[start : end + 1]
    return center <= window.min()


def find_turning_points(df_close: pd.Series, order: int = context.local_extrema_order):
    pts = []
    n = len(df_close)
    for i in range(order, n - order):
        if is_local_max(df_close, i, order):
            pts.append((i, "max"))
        elif is_local_min(df_close, i, order):
            pts.append((i, "min"))
    return pts
