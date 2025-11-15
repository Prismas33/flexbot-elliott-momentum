"""Momentum strategy helpers."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from .. import bias, context, data, divergence, indicators, state
from ..risk import get_risk_multiplier, loss_streaks

ctx = context

fetch_ohlcv = data.fetch_ohlcv
compute_rsi = indicators.compute_rsi
compute_atr = indicators.compute_atr
has_bullish_rsi_divergence = divergence.has_bullish_rsi_divergence
compute_symbol_bias = bias.compute_symbol_bias


def momentum_confirm(df):
    if df is None or len(df) < 50:
        return False, {}
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["atr"] = compute_atr(df, ctx.atr_period)
    df["vol_ma"] = df["volume"].rolling(ctx.volume_ma_period, min_periods=1).mean()
    df["rsi"] = compute_rsi(df, ctx.rsi_period)

    ema_cross = df["ema20"].iloc[-1] > df["ema50"].iloc[-1]
    atr_expansion = df["atr"].iloc[-1] > df["atr"].iloc[-5:].mean() * 1.05
    vol_spike = df["volume"].iloc[-1] > df["vol_ma"].iloc[-1] * 1.1

    rsi_val = float(df["rsi"].iloc[-1]) if not np.isnan(df["rsi"].iloc[-1]) else None
    rsi_ok = True
    if rsi_val is not None and rsi_val > 80:
        rsi_ok = False

    score = 0.0
    if ema_cross:
        score += 1.5
    if atr_expansion:
        score += 1.0
    if vol_spike:
        score += 0.5

    confirmed = score >= 1.5 and rsi_ok
    return confirmed, {
        "ema_cross": ema_cross,
        "atr_expansion": atr_expansion,
        "vol_spike": vol_spike,
        "score": score,
        "atr": df["atr"].iloc[-1],
        "last_close": df["close"].iloc[-1],
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
    }


def analyze_momentum_and_maybe_trade(symbol: str):
    summary: Dict[str, object] = {
        "symbol": symbol,
        "status": "ok",
        "strategy": "momentum",
        "should_enter": False,
        "reference_tf": None,
        "score": None,
    }
    if ctx.trade_bias == "short":
        summary["status"] = "bias_not_supported"
        summary["note"] = "Momentum strategy opera apenas comprado."
        return None, summary

    bias_map = compute_symbol_bias(symbol)
    summary["bias"] = bias_map
    confirmations: Dict[str, Dict[str, object]] = {}
    allowed_entry_tfs = ctx.entry_timeframes if ctx.entry_timeframes else list(ctx.timeframes)
    core_tfs: Tuple[str, ...] = tuple(allowed_entry_tfs) if allowed_entry_tfs else ("15m", "1h")

    for tf in ctx.timeframes:
        bias_dir = bias_map.get(tf, "both")
        df = fetch_ohlcv(symbol, tf, limit=ctx.lookback)
        if df is None or len(df) < 50:
            confirmations[tf] = {"confirmed": False}
            continue

        has_div = has_bullish_rsi_divergence(df)
        confirmed, details = momentum_confirm(df)
        if has_div and confirmed:
            details["score"] = details.get("score", 0) + 1
        if bias_dir == "down" and confirmed and details.get("score", 0) < ctx.momentum_bias_override_score:
            confirmations[tf] = {"confirmed": False, "bias_blocked": True}
            continue

        confirmations[tf] = {
            "confirmed": confirmed,
            "details": details,
            "price": details.get("last_close"),
            "atr": details.get("atr"),
            "bias": bias_dir,
            "bar_time": df.index[-1] if len(df.index) else None,
        }

    summary["signals"] = {tf: data.get("confirmed") for tf, data in confirmations.items()}
    ready = [
        (tf, data)
        for tf, data in confirmations.items()
        if data.get("confirmed") and data.get("details") and (not allowed_entry_tfs or tf in allowed_entry_tfs)
    ]
    required = min(ctx.momentum_min_tf_agree, len(ready))
    if len(ready) < max(1, required):
        return None, summary

    ready.sort(
        key=lambda item: (
            1 if item[0] in core_tfs else 0,
            item[1]["details"].get("score", 0),
        ),
        reverse=True,
    )
    ref_tf, ref_data = ready[0]
    summary["reference_tf"] = ref_tf
    summary["score"] = ref_data["details"].get("score")
    entry_price = ref_data["price"]
    atr_val = ref_data.get("atr") or 0
    if atr_val == 0:
        summary["status"] = "no_atr"
        return None, summary

    stop_price = entry_price - atr_val * ctx.momentum_stop_atr
    tp_price = entry_price + atr_val * ctx.momentum_tp_atr
    rr = state.compute_rr(entry_price, stop_price, tp_price)
    summary["rr"] = rr
    if rr is None or rr < ctx.min_rr_required:
        summary["status"] = "min_rr_not_met"
        return None, summary

    account_balance = state.fetch_account_balance()
    risk_mult = get_risk_multiplier(symbol, "momentum_long")
    if risk_mult < 1.0:
        streak = loss_streaks[(symbol, "momentum_long")]
        base_pct = ctx.risk_percent * 100
        logging.info(
            "Risco reduzido em %s (Momentum) — streak=%d risco=%.2f%%",
            symbol,
            streak,
            base_pct * risk_mult,
        )

    if state.has_open_position(symbol):
        summary["status"] = "position_open"
        summary["note"] = "Já existe posição aberta; aguardando encerramento antes de novo trade."
        return None, summary

    bar_time = ref_data.get("bar_time")
    key = (symbol, "momentum_long", ref_tf)
    if bar_time is not None:
        last_bar = state.last_entry_tracker.get(key)
        if last_bar is not None and bar_time <= last_bar:
            summary["status"] = "already_traded_bar"
            summary["note"] = f"Entrada já executada na vela {bar_time}."
            return None, summary

    pos = state.enter_position(
        symbol,
        entry_price,
        stop_price,
        tp_price,
        account_balance,
        side="buy",
        strategy_label="momentum_long",
        risk_multiplier=None,
        bar_time=bar_time,
        timeframe=ref_tf,
    )
    summary["should_enter"] = pos is not None
    summary["entry_price"] = entry_price
    summary["direction"] = "long"
    return pos, summary

__all__ = ["momentum_confirm", "analyze_momentum_and_maybe_trade"]
