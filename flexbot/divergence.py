from __future__ import annotations

import numpy as np
import pandas as pd

from . import context
from .indicators import compute_rsi, find_turning_points


def has_bullish_rsi_divergence(df: pd.DataFrame, order: int | None = None) -> bool:
    if df is None or len(df) < 40:
        return False
    order = order or context.local_extrema_order
    close = df["close"]
    rsi = compute_rsi(df, context.rsi_period)
    if rsi.isna().all():
        return False
    pts = find_turning_points(close, order=order)
    lows = [idx for idx, typ in pts if typ == "min"]
    if len(lows) < 2:
        return False
    idx1, idx2 = lows[-2], lows[-1]
    if (idx2 - idx1) < 8:
        return False
    price1 = close.iloc[idx1]
    price2 = close.iloc[idx2]
    rsi1 = rsi.iloc[idx1]
    rsi2 = rsi.iloc[idx2]
    if np.isnan(price1) or np.isnan(price2) or np.isnan(rsi1) or np.isnan(rsi2):
        return False
    threshold = max(0.0, context.divergence_min_drop_pct)
    price_drop_pct = abs(price2 - price1) / price1 if price1 != 0 else 0.0
    if price_drop_pct < threshold:
        return False
    return price2 < price1 and rsi2 > rsi1


def has_bearish_rsi_divergence(df: pd.DataFrame, order: int | None = None) -> bool:
    if df is None or len(df) < 40:
        return False
    order = order or context.local_extrema_order
    close = df["close"]
    rsi = compute_rsi(df, context.rsi_period)
    if rsi.isna().all():
        return False
    pts = find_turning_points(close, order=order)
    highs = [idx for idx, typ in pts if typ == "max"]
    if len(highs) < 2:
        return False
    idx1, idx2 = highs[-2], highs[-1]
    if (idx2 - idx1) < 8:
        return False
    price1 = close.iloc[idx1]
    price2 = close.iloc[idx2]
    rsi1 = rsi.iloc[idx1]
    rsi2 = rsi.iloc[idx2]
    if np.isnan(price1) or np.isnan(price2) or np.isnan(rsi1) or np.isnan(rsi2):
        return False
    threshold = max(0.0, context.divergence_min_drop_pct)
    price_push_pct = abs(price2 - price1) / price1 if price1 != 0 else 0.0
    if price_push_pct < threshold:
        return False
    return price2 > price1 and rsi2 < rsi1
