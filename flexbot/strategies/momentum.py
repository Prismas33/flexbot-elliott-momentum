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
has_bearish_rsi_divergence = divergence.has_bearish_rsi_divergence
compute_symbol_bias = bias.compute_symbol_bias


def momentum_confirm_long(df):
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
    if rsi_val is not None and rsi_val > ctx.momentum_rsi_long_max:
        rsi_ok = False

    score = 0.0
    if ema_cross:
        score += 1.5
    if atr_expansion:
        score += 1.0
    if vol_spike:
        score += 0.5

    confirmed = score >= 1.5 and rsi_ok
    details = {
        "ema_cross": ema_cross,
        "atr_expansion": atr_expansion,
        "vol_spike": vol_spike,
        "score": score,
        "atr": df["atr"].iloc[-1],
        "last_close": df["close"].iloc[-1],
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
        "rsi_max_allowed": ctx.momentum_rsi_long_max,
    }
    if confirmed:
        details["trigger"] = "momentum"
    return confirmed, details


def momentum_confirm_short(df):
    if df is None or len(df) < 50:
        return False, {}
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["atr"] = compute_atr(df, ctx.atr_period)
    df["vol_ma"] = df["volume"].rolling(ctx.volume_ma_period, min_periods=1).mean()
    df["rsi"] = compute_rsi(df, ctx.rsi_period)

    ema_cross = df["ema20"].iloc[-1] < df["ema50"].iloc[-1]
    atr_expansion = df["atr"].iloc[-1] > df["atr"].iloc[-5:].mean() * 1.05
    vol_spike = df["volume"].iloc[-1] > df["vol_ma"].iloc[-1] * 1.1

    rsi_val = float(df["rsi"].iloc[-1]) if not np.isnan(df["rsi"].iloc[-1]) else None
    rsi_ok = True
    if rsi_val is not None and rsi_val < ctx.momentum_rsi_short_min:
        rsi_ok = False

    score = 0.0
    if ema_cross:
        score += 1.5
    if atr_expansion:
        score += 1.0
    if vol_spike:
        score += 0.5

    confirmed = score >= 1.5 and rsi_ok
    details = {
        "ema_cross": ema_cross,
        "atr_expansion": atr_expansion,
        "vol_spike": vol_spike,
        "score": score,
        "atr": df["atr"].iloc[-1],
        "last_close": df["close"].iloc[-1],
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
        "rsi_min_allowed": ctx.momentum_rsi_short_min,
    }
    if confirmed:
        details["trigger"] = "momentum"
    return confirmed, details


def momentum_confirm(df):
    return momentum_confirm_long(df)


def analyze_momentum_and_maybe_trade(symbol: str):
    summary: Dict[str, object] = {
        "symbol": symbol,
        "status": "ok",
        "strategy": "momentum",
        "should_enter": False,
        "reference_tf": None,
        "score": None,
    }

    bias_map = compute_symbol_bias(symbol)
    summary["bias"] = bias_map
    confirmations: Dict[str, Dict[str, object]] = {}
    allowed_entry_tfs = ctx.entry_timeframes if ctx.entry_timeframes else list(ctx.timeframes)
    core_tfs: Tuple[str, ...] = tuple(allowed_entry_tfs) if allowed_entry_tfs else ("15m", "1h")

    for tf in ctx.timeframes:
        bias_dir = bias_map.get(tf, "both")
        df = fetch_ohlcv(symbol, tf, limit=ctx.lookback)
        tf_entry: Dict[str, object] = {"bias": bias_dir}
        if df is None or len(df) < 50:
            tf_entry["long"] = {"confirmed": False, "bias": bias_dir}
            tf_entry["short"] = {"confirmed": False, "bias": bias_dir}
            confirmations[tf] = tf_entry
            continue

        if ctx.trade_bias in ("long", "both"):
            has_div_long = has_bullish_rsi_divergence(df)
            confirmed_long, details_long = momentum_confirm_long(df)
            if details_long is None:
                details_long = {}
            details_long["has_divergence"] = has_div_long
            details_long["require_divergence"] = ctx.momentum_require_divergence
            divergence_ok = has_div_long or not ctx.momentum_require_divergence
            details_long["divergence_ok"] = divergence_ok
            bonus_applied = False
            if confirmed_long and ctx.momentum_use_divergence_bonus and has_div_long:
                details_long["score"] = details_long.get("score", 0) + 1.0
                bonus_applied = True
            details_long["divergence_bonus_applied"] = bonus_applied
            if confirmed_long and not divergence_ok:
                confirmed_long = False
                details_long["divergence_blocked"] = True
            long_info = {
                "confirmed": confirmed_long,
                "details": details_long,
                "price": details_long.get("last_close") if details_long else None,
                "atr": details_long.get("atr") if details_long else None,
                "bias": bias_dir,
                "bar_time": df.index[-1] if len(df.index) else None,
            }
            if confirmed_long and bias_dir == "down" and details_long.get("score", 0) < ctx.momentum_bias_override_score:
                long_info["confirmed"] = False
                long_info["bias_blocked"] = True
            tf_entry["long"] = long_info
        else:
            tf_entry["long"] = {"confirmed": False, "bias": bias_dir}

        if ctx.trade_bias in ("short", "both"):
            has_div_short = has_bearish_rsi_divergence(df)
            confirmed_short, details_short = momentum_confirm_short(df)
            if details_short is None:
                details_short = {}
            details_short["has_divergence"] = has_div_short
            details_short["require_divergence"] = ctx.momentum_require_divergence
            divergence_ok_short = has_div_short or not ctx.momentum_require_divergence
            details_short["divergence_ok"] = divergence_ok_short
            bonus_applied_short = False
            if confirmed_short and ctx.momentum_use_divergence_bonus and has_div_short:
                details_short["score"] = details_short.get("score", 0) + 1.0
                bonus_applied_short = True
            details_short["divergence_bonus_applied"] = bonus_applied_short
            if confirmed_short and not divergence_ok_short:
                confirmed_short = False
                details_short["divergence_blocked"] = True
            short_info = {
                "confirmed": confirmed_short,
                "details": details_short,
                "price": details_short.get("last_close") if details_short else None,
                "atr": details_short.get("atr") if details_short else None,
                "bias": bias_dir,
                "bar_time": df.index[-1] if len(df.index) else None,
            }
            if confirmed_short and bias_dir == "up" and details_short.get("score", 0) < ctx.momentum_bias_override_score:
                short_info["confirmed"] = False
                short_info["bias_blocked"] = True
            tf_entry["short"] = short_info
        else:
            tf_entry["short"] = {"confirmed": False, "bias": bias_dir}

        confirmations[tf] = tf_entry

    summary["signals"] = {
        tf: {
            "long": entry.get("long", {}).get("confirmed", False),
            "short": entry.get("short", {}).get("confirmed", False),
        }
        for tf, entry in confirmations.items()
    }

    candidates = []
    if ctx.trade_bias in ("long", "both"):
        long_ready = [
            (tf, entry.get("long"))
            for tf, entry in confirmations.items()
            if entry.get("long", {}).get("confirmed") and entry.get("long", {}).get("details") and (not allowed_entry_tfs or tf in allowed_entry_tfs)
        ]
        if long_ready:
            long_ready.sort(
                key=lambda item: (
                    1 if item[0] in core_tfs else 0,
                    item[1]["details"].get("score", 0),
                ),
                reverse=True,
            )
            ref_tf, ref_data = long_ready[0]
            entry_price = ref_data.get("price")
            atr_val = ref_data.get("atr") or 0
            if entry_price is not None and atr_val > 0:
                stop_price = entry_price - atr_val * ctx.momentum_stop_atr
                tp_price = entry_price + atr_val * ctx.momentum_tp_atr
                rr = state.compute_rr(entry_price, stop_price, tp_price, side="buy")
                candidates.append({
                    "tf": ref_tf,
                    "details": ref_data.get("details", {}),
                    "entry_price": entry_price,
                    "atr": atr_val,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "rr": rr,
                    "score": ref_data.get("details", {}).get("score", 0),
                    "risk_key": "momentum_long",
                    "side": "buy",
                    "direction": "long",
                    "bar_time": ref_data.get("bar_time"),
                })

    if ctx.trade_bias in ("short", "both"):
        short_ready = [
            (tf, entry.get("short"))
            for tf, entry in confirmations.items()
            if entry.get("short", {}).get("confirmed") and entry.get("short", {}).get("details") and (not allowed_entry_tfs or tf in allowed_entry_tfs)
        ]
        if short_ready:
            short_ready.sort(
                key=lambda item: (
                    1 if item[0] in core_tfs else 0,
                    item[1]["details"].get("score", 0),
                ),
                reverse=True,
            )
            ref_tf, ref_data = short_ready[0]
            entry_price = ref_data.get("price")
            atr_val = ref_data.get("atr") or 0
            if entry_price is not None and atr_val > 0:
                stop_price = entry_price + atr_val * ctx.momentum_stop_atr
                tp_price = entry_price - atr_val * ctx.momentum_tp_atr
                rr = state.compute_rr(entry_price, stop_price, tp_price, side="sell")
                candidates.append({
                    "tf": ref_tf,
                    "details": ref_data.get("details", {}),
                    "entry_price": entry_price,
                    "atr": atr_val,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "rr": rr,
                    "score": ref_data.get("details", {}).get("score", 0),
                    "risk_key": "momentum_short",
                    "side": "sell",
                    "direction": "short",
                    "bar_time": ref_data.get("bar_time"),
                })

    if not candidates:
        summary["status"] = "no_confirmed"
        return None, summary

    valid_candidates = [c for c in candidates if c.get("rr") is not None and c.get("rr") >= ctx.min_rr_required]
    if not valid_candidates:
        summary["status"] = "min_rr_not_met"
        summary["rr"] = max((c.get("rr") or 0) for c in candidates)
        return None, summary

    valid_candidates.sort(key=lambda c: (c.get("score", 0), c.get("rr", 0)), reverse=True)
    selected = valid_candidates[0]

    summary["reference_tf"] = selected["tf"]
    summary["score"] = selected.get("score")
    summary["rr"] = selected.get("rr")
    summary["entry_price"] = selected.get("entry_price")
    summary["direction"] = selected.get("direction")
    summary["trigger"] = selected.get("details", {}).get("trigger")

    if state.has_open_position(symbol):
        summary["status"] = "position_open"
        summary["note"] = "Já existe posição aberta; aguardando encerramento antes de novo trade."
        return None, summary

    bar_time = selected.get("bar_time")
    tf_key = selected.get("tf")
    risk_key = selected["risk_key"]
    if bar_time is not None and tf_key is not None:
        last_bar = state.last_entry_tracker.get((symbol, risk_key, tf_key))
        if last_bar is not None and bar_time <= last_bar:
            summary["status"] = "already_traded_bar"
            summary["note"] = f"Entrada já executada na vela {bar_time}."
            return None, summary

    account_balance = state.fetch_account_balance()
    risk_mult = get_risk_multiplier(symbol, risk_key)
    if risk_mult < 1.0:
        streak = loss_streaks[(symbol, risk_key)]
        base_pct = ctx.risk_percent * 100
        logging.info(
            "Risco reduzido em %s (%s) — streak=%d risco=%.2f%%",
            symbol,
            risk_key,
            streak,
            base_pct * risk_mult,
        )

    pos = state.enter_position(
        symbol,
        selected["entry_price"],
        selected["stop_price"],
        selected["tp_price"],
        account_balance,
        side=selected["side"],
        strategy_label=risk_key,
        risk_multiplier=None,
        bar_time=selected.get("bar_time"),
        timeframe=selected.get("tf"),
        trailing_enabled=ctx.momentum_use_trailing,
        trailing_rr=ctx.momentum_trailing_rr,
        trailing_activate_rr=ctx.momentum_trailing_activate_rr,
        trailing_strategy="momentum",
    )
    summary["should_enter"] = pos is not None
    return pos, summary


__all__ = [
    "momentum_confirm",
    "momentum_confirm_long",
    "momentum_confirm_short",
    "analyze_momentum_and_maybe_trade",
]
