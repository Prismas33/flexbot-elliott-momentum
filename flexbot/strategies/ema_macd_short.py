"""OMDs short-side confirmation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ema_macd_confirm_short"]


def ema_macd_confirm_short(
    df: pd.DataFrame,
    *,
    ema_fast_period: int,
    ema_slow_period: int,
    ema_macd_fast: int,
    ema_macd_slow: int,
    ema_macd_signal: int,
    atr_period: int,
    rsi_period: int,
    compute_atr,
    compute_rsi,
    compute_macd,
    has_bearish_rsi_divergence,
    cross_lookback: int = 8,
    require_divergence: bool = True,
    require_rsi_zone: bool = False,
    rsi_zone_short_min: float = 70.0,
):
    """Analisa condições de venda para OMDs (versão short)."""
    if df is None or len(df) < max(ema_slow_period, ema_macd_slow) + 5:
        return False, {}

    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=ema_fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow_period, adjust=False).mean()
    macd_line, signal_line, hist = compute_macd(df['close'], ema_macd_fast, ema_macd_slow, ema_macd_signal)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['atr'] = compute_atr(df, atr_period)
    df['rsi'] = compute_rsi(df, rsi_period)

    def crossed_down(series_a, series_b, lookback):
        length = len(series_a)
        for offset in range(lookback):
            idx = length - 1 - offset
            prev_idx = idx - 1
            if prev_idx < 0:
                break
            if series_a.iloc[idx] < series_b.iloc[idx] and series_a.iloc[prev_idx] >= series_b.iloc[prev_idx]:
                return True
        return False

    ema_cross_recent = crossed_down(df['ema_fast'], df['ema_slow'], cross_lookback)
    macd_cross_recent = crossed_down(df['macd'], df['macd_signal'], cross_lookback)

    ema_aligned = df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]
    macd_aligned = df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]
    histogram_negative = df['macd_hist'].iloc[-1] < 0
    atr_available = df['atr'].iloc[-1] > 0

    recent_hist = df['macd_hist'].iloc[-5:]
    hist_now = recent_hist.iloc[-1]
    hist_trough = recent_hist.min()
    hist_ok = abs(hist_now) <= abs(hist_trough) * 1.1 if hist_trough != 0 else True

    ema_fast_recent = df['ema_fast'].iloc[-5:]
    ema_slow_recent = df['ema_slow'].iloc[-5:]
    slope_fast = ema_fast_recent.iloc[-1] - ema_fast_recent.iloc[0]
    slope_slow = ema_slow_recent.iloc[-1] - ema_slow_recent.iloc[0]
    slope_ok = slope_fast < 0 and slope_slow <= 0.0001

    has_divergence = has_bearish_rsi_divergence(df)
    divergence_ok = has_divergence or not require_divergence

    rsi_val = float(df['rsi'].iloc[-1]) if not np.isnan(df['rsi'].iloc[-1]) else None
    rsi_ok = True
    rsi_zone_ok = True
    if rsi_val is None:
        if require_rsi_zone:
            rsi_zone_ok = False
            rsi_ok = False
    else:
        if require_rsi_zone:
            rsi_zone_ok = rsi_val >= rsi_zone_short_min
            rsi_ok = rsi_zone_ok
        else:
            if rsi_val < 18 or rsi_val > 82:
                rsi_ok = False

    confluence = (
        (ema_cross_recent and macd_aligned) or
        (macd_cross_recent and ema_aligned)
    )

    confirmed = (
        ema_aligned and
        macd_aligned and
        histogram_negative and
        confluence and
        divergence_ok and
        atr_available and
        slope_ok and
        hist_ok and
        rsi_ok and
        rsi_zone_ok
    )

    score = 3 if confirmed else 0

    details = {
        "ema_fast": float(df['ema_fast'].iloc[-1]),
        "ema_slow": float(df['ema_slow'].iloc[-1]),
        "macd": float(df['macd'].iloc[-1]),
        "macd_signal": float(df['macd_signal'].iloc[-1]),
        "macd_hist": float(df['macd_hist'].iloc[-1]),
        "atr": float(df['atr'].iloc[-1]),
        "last_close": float(df['close'].iloc[-1]),
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
        "rsi_zone_ok": rsi_zone_ok,
        "hist_ok": hist_ok,
        "slope_fast": float(slope_fast),
        "slope_slow": float(slope_slow),
        "ema_cross_recent": ema_cross_recent,
        "macd_cross_recent": macd_cross_recent,
        "has_divergence": has_divergence,
        "require_divergence": require_divergence,
        "divergence_ok": divergence_ok,
        "require_rsi_zone": require_rsi_zone,
        "rsi_zone_short_min": rsi_zone_short_min,
        "score": score,
    }
    if confirmed:
        details["trigger"] = "ema_macd" if ema_cross_recent and macd_cross_recent else ("ema" if ema_cross_recent else "macd")
    else:
        details["trigger"] = None

    return confirmed, details
