"""Signal analysis helpers used by the CLI and dashboard entrypoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from . import bias, context as ctx, data, indicators, state, divergence
from .risk import get_risk_multiplier, loss_streaks
from .strategies import analyze_momentum_and_maybe_trade
from .strategies.ema_macd import ema_macd_confirm
from .strategies.ema_macd_short import ema_macd_confirm_short

__all__ = ["analyze_ema_macd_and_maybe_trade", "analyze_and_maybe_trade"]

compute_symbol_bias = bias.compute_symbol_bias
bias_allows_long = bias.bias_allows_long
bias_allows_short = bias.bias_allows_short
fetch_ohlcv = data.fetch_ohlcv
compute_rr = state.compute_rr
enter_position = state.enter_position
has_open_position = state.has_open_position
fetch_account_balance = state.fetch_account_balance
compute_atr = indicators.compute_atr
compute_rsi = indicators.compute_rsi
compute_macd = indicators.compute_macd
has_bearish_rsi_divergence = divergence.has_bearish_rsi_divergence


def analyze_ema_macd_and_maybe_trade(symbol: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Run the OMDs confirmation stack for a single symbol and maybe open a trade."""
    summary: Dict[str, Any] = {
        "symbol": symbol,
        "status": "ok",
        "strategy": "ema_macd",
        "should_enter": False,
        "reference_tf": None,
        "score": None,
    }
    bias_map = compute_symbol_bias(symbol)
    summary["bias"] = bias_map
    confirmations: Dict[str, Dict[str, Any]] = {}
    allowed_entry_tfs = ctx.entry_timeframes if ctx.entry_timeframes else list(ctx.timeframes)
    for tf in ctx.timeframes:
        bias_dir = bias_map.get(tf, "both")
        df = fetch_ohlcv(symbol, tf, limit=ctx.lookback)
        tf_entry: Dict[str, Any] = {"bias": bias_dir}
        if df is None or len(df) < ctx.ema_slow_period + 10:
            tf_entry["long"] = {"confirmed": False, "bias": bias_dir}
            tf_entry["short"] = {"confirmed": False, "bias": bias_dir}
            confirmations[tf] = tf_entry
            continue
        current_bar = df.index[-1] if len(df.index) else None

        if ctx.trade_bias in ("long", "both"):
            confirmed_long, details_long = ema_macd_confirm(
                df,
                cross_lookback=ctx.ema_macd_cross_lookback,
                require_divergence=ctx.ema_macd_require_divergence,
                require_rsi_zone=ctx.ema_require_rsi_zone,
                rsi_zone_long_max=ctx.ema_rsi_zone_long_max,
            )
            long_info = {
                "confirmed": confirmed_long,
                "details": details_long,
                "price": details_long.get("last_close") if details_long else None,
                "atr": details_long.get("atr") if details_long else None,
                "bias": bias_dir,
                "bar_time": current_bar,
            }
            if confirmed_long and not bias_allows_long(tf, bias_map) and details_long.get("score", 0) < ctx.ema_macd_bias_override_score:
                long_info["confirmed"] = False
                long_info["bias_blocked"] = True
            tf_entry["long"] = long_info
        else:
            tf_entry["long"] = {"confirmed": False, "bias": bias_dir}

        if ctx.trade_bias in ("short", "both"):
            confirmed_short, details_short = ema_macd_confirm_short(
                df,
                ema_fast_period=ctx.ema_fast_period,
                ema_slow_period=ctx.ema_slow_period,
                ema_macd_fast=ctx.ema_macd_fast,
                ema_macd_slow=ctx.ema_macd_slow,
                ema_macd_signal=ctx.ema_macd_signal,
                atr_period=ctx.atr_period,
                rsi_period=ctx.rsi_period,
                compute_atr=compute_atr,
                compute_rsi=compute_rsi,
                compute_macd=compute_macd,
                has_bearish_rsi_divergence=has_bearish_rsi_divergence,
                cross_lookback=ctx.ema_macd_cross_lookback,
                require_divergence=ctx.ema_macd_require_divergence,
                require_rsi_zone=ctx.ema_require_rsi_zone,
                rsi_zone_short_min=ctx.ema_rsi_zone_short_min,
            )
            if not confirmed_short and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "OMDs short rejeitado (live) %s %s — detalhes: %s",
                    symbol,
                    tf,
                    details_short,
                )
            short_info = {
                "confirmed": confirmed_short,
                "details": details_short,
                "price": details_short.get("last_close") if details_short else None,
                "atr": details_short.get("atr") if details_short else None,
                "bias": bias_dir,
                "bar_time": current_bar,
            }
            if confirmed_short and not bias_allows_short(tf, bias_map) and details_short.get("score", 0) < ctx.ema_macd_bias_override_score:
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
            long_ready.sort(key=lambda item: item[1]["details"].get("score", 0), reverse=True)
            ref_tf, ref_data = long_ready[0]
            entry_price = ref_data.get("price")
            atr_val = ref_data.get("atr") or 0
            if entry_price is not None and atr_val > 0:
                stop_price = entry_price - atr_val * ctx.ema_macd_stop_atr
                tp_price = entry_price + atr_val * ctx.ema_macd_tp_atr
                rr = compute_rr(entry_price, stop_price, tp_price, side='buy')
                candidates.append({
                    "tf": ref_tf,
                    "details": ref_data.get("details", {}),
                    "entry_price": entry_price,
                    "atr": atr_val,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "rr": rr,
                    "score": ref_data.get("details", {}).get("score", 0),
                    "side": 'buy',
                    "direction": "long",
                    "risk_key": "ema_macd_long",
                    "bar_time": ref_data.get("bar_time"),
                })

    if ctx.trade_bias in ("short", "both"):
        short_ready = [
            (tf, entry.get("short"))
            for tf, entry in confirmations.items()
            if entry.get("short", {}).get("confirmed") and entry.get("short", {}).get("details") and (not allowed_entry_tfs or tf in allowed_entry_tfs)
        ]
        if short_ready:
            short_ready.sort(key=lambda item: item[1]["details"].get("score", 0), reverse=True)
            ref_tf, ref_data = short_ready[0]
            entry_price = ref_data.get("price")
            atr_val = ref_data.get("atr") or 0
            if entry_price is not None and atr_val > 0:
                stop_price = entry_price + atr_val * ctx.ema_macd_stop_atr
                tp_price = entry_price - atr_val * ctx.ema_macd_tp_atr
                rr = compute_rr(entry_price, stop_price, tp_price, side='sell')
                candidates.append({
                    "tf": ref_tf,
                    "details": ref_data.get("details", {}),
                    "entry_price": entry_price,
                    "atr": atr_val,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "rr": rr,
                    "score": ref_data.get("details", {}).get("score", 0),
                    "side": 'sell',
                    "direction": "short",
                    "risk_key": "ema_macd_short",
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

    if has_open_position(symbol):
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

    account_balance = fetch_account_balance()
    risk_mult = get_risk_multiplier(symbol, selected["risk_key"])
    if risk_mult < 1.0:
        streak = loss_streaks[(symbol, selected["risk_key"])]
        base_pct = ctx.risk_percent * 100
        logging.info(
            "Risco reduzido em %s (%s) — streak=%d risco=%.2f%%",
            symbol,
            selected["risk_key"],
            streak,
            base_pct * risk_mult,
        )

    trailing_kwargs = {}
    if risk_key.startswith("ema"):
        trailing_kwargs = {
            "trailing_enabled": ctx.ema_macd_use_trailing,
            "trailing_rr": ctx.ema_macd_trailing_rr,
            "trailing_activate_rr": ctx.ema_macd_trailing_activate_rr,
            "trailing_strategy": "ema_macd",
        }
    elif risk_key.startswith("momentum"):
        trailing_kwargs = {
            "trailing_enabled": ctx.momentum_use_trailing,
            "trailing_rr": ctx.momentum_trailing_rr,
            "trailing_activate_rr": ctx.momentum_trailing_activate_rr,
            "trailing_strategy": "momentum",
        }

    pos = enter_position(
        symbol,
        selected["entry_price"],
        selected["stop_price"],
        selected["tp_price"],
        account_balance,
        side=selected["side"],
        strategy_label=selected["risk_key"],
        risk_multiplier=None,
        bar_time=selected.get("bar_time"),
        timeframe=selected.get("tf"),
        **trailing_kwargs,
    )
    summary["should_enter"] = pos is not None
    return pos, summary


def analyze_and_maybe_trade(symbol: str):
    """Dispatches to the active strategy implementation."""
    if ctx.strategy_mode == "ema_macd":
        return analyze_ema_macd_and_maybe_trade(symbol)
    return analyze_momentum_and_maybe_trade(symbol)
