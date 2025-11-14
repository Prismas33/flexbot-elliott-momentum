import numpy as np
import pandas as pd


def ema_macd_confirm_long(
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
    has_bullish_rsi_divergence,
    cross_lookback: int = 8,
    require_divergence: bool = True,
):
    """Confirma setup de compra para EMA + MACD."""
    if df is None or len(df) < max(ema_slow_period, ema_macd_slow) + 5:
        return False, {}

    working = df.copy()
    working['ema_fast'] = working['close'].ewm(span=ema_fast_period, adjust=False).mean()
    working['ema_slow'] = working['close'].ewm(span=ema_slow_period, adjust=False).mean()
    macd_line, signal_line, hist = compute_macd(working['close'], ema_macd_fast, ema_macd_slow, ema_macd_signal)
    working['macd'] = macd_line
    working['macd_signal'] = signal_line
    working['macd_hist'] = hist
    working['atr'] = compute_atr(working, atr_period)
    working['rsi'] = compute_rsi(working, rsi_period)

    def crossed_up(series_a, series_b, lookback):
        length = len(series_a)
        for offset in range(lookback):
            idx = length - 1 - offset
            prev_idx = idx - 1
            if prev_idx < 0:
                break
            if series_a.iloc[idx] > series_b.iloc[idx] and series_a.iloc[prev_idx] <= series_b.iloc[prev_idx]:
                return True
        return False

    ema_cross_recent = crossed_up(working['ema_fast'], working['ema_slow'], cross_lookback)
    macd_cross_recent = crossed_up(working['macd'], working['macd_signal'], cross_lookback)

    ema_aligned = working['ema_fast'].iloc[-1] > working['ema_slow'].iloc[-1]
    macd_aligned = working['macd'].iloc[-1] > working['macd_signal'].iloc[-1]
    histogram_positive = working['macd_hist'].iloc[-1] > 0
    atr_available = working['atr'].iloc[-1] > 0

    recent_hist = working['macd_hist'].iloc[-5:]
    hist_now = recent_hist.iloc[-1]
    hist_peak = recent_hist.max()
    hist_ok = abs(hist_now) <= abs(hist_peak) * 1.1

    ema_fast_recent = working['ema_fast'].iloc[-5:]
    ema_slow_recent = working['ema_slow'].iloc[-5:]
    slope_fast = ema_fast_recent.iloc[-1] - ema_fast_recent.iloc[0]
    slope_slow = ema_slow_recent.iloc[-1] - ema_slow_recent.iloc[0]
    slope_ok = slope_fast > 0 and slope_slow >= -0.0001

    has_divergence = has_bullish_rsi_divergence(working)
    divergence_ok = has_divergence or not require_divergence

    rsi_val = float(working['rsi'].iloc[-1]) if not np.isnan(working['rsi'].iloc[-1]) else None
    rsi_ok = True
    if rsi_val is not None and (rsi_val < 28 or rsi_val > 78):
        rsi_ok = False

    confluence = (
        (ema_cross_recent and macd_aligned) or
        (macd_cross_recent and ema_aligned)
    )

    confirmed = (
        ema_aligned and
        macd_aligned and
        histogram_positive and
        confluence and
        divergence_ok and
        atr_available and
        slope_ok and
        hist_ok and
        rsi_ok
    )

    score = 3 if confirmed else 0

    details = {
        "ema_fast": float(working['ema_fast'].iloc[-1]),
        "ema_slow": float(working['ema_slow'].iloc[-1]),
        "macd": float(working['macd'].iloc[-1]),
        "macd_signal": float(working['macd_signal'].iloc[-1]),
        "macd_hist": float(working['macd_hist'].iloc[-1]),
        "atr": float(working['atr'].iloc[-1]),
        "last_close": float(working['close'].iloc[-1]),
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
        "hist_ok": hist_ok,
        "slope_fast": float(slope_fast),
        "slope_slow": float(slope_slow),
        "ema_cross_recent": ema_cross_recent,
        "macd_cross_recent": macd_cross_recent,
        "has_divergence": has_divergence,
        "require_divergence": require_divergence,
        "divergence_ok": divergence_ok,
        "score": score,
    }
    if confirmed:
        details["trigger"] = "ema_macd" if ema_cross_recent and macd_cross_recent else ("ema" if ema_cross_recent else "macd")
    else:
        details["trigger"] = None

    return confirmed, details
