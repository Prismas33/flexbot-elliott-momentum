"""
elliott_momentum_breakout_bot.py

Sistema Momentum / EMA+MACD Breakout
- Multi-par (ETH/BTC/SOL) + Multi-timeframe (5m,15m,1h)
- SL/TP automáticos, trailing stop opcional, position sizing por risco
- Backtester simples para simular sinais e métricas

INSTRUÇÕES RÁPIDAS
1) pip install ccxt pandas numpy ta
2) Preenche API_KEY / API_SECRET
3) Mantém paper=True para testar
4) Para correr backtest: executar o ficheiro; por defeito faz backtest ETH 15m
5) Para correr em live: set paper=False e chama main_loop()

Nota: testa em paper e faz backtests antes de usar capital real.
"""

import sys
import argparse
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

from strategies.ema_macd_long import ema_macd_confirm_long
from strategies.ema_macd_short import ema_macd_confirm_short

from state_store import (
    write_runtime_state,
    write_backtest_summary,
    update_user_config,
)

from flexbot import bias, context, data, divergence, indicators, state
from flexbot.risk import get_risk_multiplier, register_trade_outcome, loss_streaks
from flexbot.strategies import analyze_momentum_and_maybe_trade, momentum_confirm


ctx = context

fetch_ohlcv = data.fetch_ohlcv
fetch_ohlcv_paginated = data.fetch_ohlcv_paginated
fetch_backtest_series = data.fetch_backtest_series
timeframe_to_minutes = data.timeframe_to_minutes

compute_rsi = indicators.compute_rsi
compute_atr = indicators.compute_atr
compute_macd = indicators.compute_macd
find_turning_points = indicators.find_turning_points

has_bullish_rsi_divergence = divergence.has_bullish_rsi_divergence
has_bearish_rsi_divergence = divergence.has_bearish_rsi_divergence

determine_trend_direction = bias.determine_trend_direction
compute_symbol_bias = bias.compute_symbol_bias
bias_allows_long = bias.bias_allows_long
bias_allows_short = bias.bias_allows_short
bias_from_slices = bias.bias_from_slices

has_open_position = state.has_open_position
serialize_open_positions = state.serialize_open_positions
enter_position = state.enter_position
monitor_and_close_positions = state.monitor_and_close_positions
compute_size = state.compute_size
compute_rr = state.compute_rr
fetch_account_balance = state.fetch_account_balance
get_environment_mode = state.get_environment_mode
validate_symbol = state.validate_symbol
reset_runtime_state = state.reset_runtime_state

# Dependency check — helpful message if a module is missing
try:
    import pandas as _pd  # quick check for common libs
    import numpy as _np
    from ta.momentum import RSIIndicator as _rsi
except ModuleNotFoundError as e:
    logging.error("Missing Python package: %s. Install dependencies with:\n  py -3 -m pip install -r requirements.txt\nOr: py -3 -m pip install ccxt pandas numpy ta", e.name)
    raise

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

exchange = ctx.exchange
exchange_id = ctx.exchange_id

# ---------- Momentum & Breakout filters ----------
def ema_macd_confirm(df, cross_lookback=None, require_divergence=None):
    cross_lb = cross_lookback if cross_lookback is not None else ctx.ema_macd_cross_lookback
    divergence_required = ctx.ema_macd_require_divergence if require_divergence is None else require_divergence
    return ema_macd_confirm_long(
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
        has_bullish_rsi_divergence=has_bullish_rsi_divergence,
        cross_lookback=cross_lb,
        require_divergence=divergence_required,
    )

# ---------- Live analyze & execute ----------
def analyze_ema_macd_and_maybe_trade(symbol):
    summary = {
        "symbol": symbol,
        "status": "ok",
        "strategy": "ema_macd",
        "should_enter": False,
        "reference_tf": None,
        "score": None,
    }
    bias_map = compute_symbol_bias(symbol)
    summary["bias"] = bias_map
    confirmations = {}
    allowed_entry_tfs = ctx.entry_timeframes if ctx.entry_timeframes else list(ctx.timeframes)
    for tf in ctx.timeframes:
        bias_dir = bias_map.get(tf, "both")
        df = fetch_ohlcv(symbol, tf, limit=ctx.lookback)
        tf_entry = {"bias": bias_dir}
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
            )
            if not confirmed_short and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "EMA+MACD short rejeitado (live) %s %s — detalhes: %s",
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
            "long": data.get("long", {}).get("confirmed", False),
            "short": data.get("short", {}).get("confirmed", False),
        }
        for tf, data in confirmations.items()
    }
    candidates = []
    if ctx.trade_bias in ("long", "both"):
        long_ready = [
            (tf, data.get("long"))
            for tf, data in confirmations.items()
            if data.get("long", {}).get("confirmed") and data.get("long", {}).get("details") and (not allowed_entry_tfs or tf in allowed_entry_tfs)
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
            (tf, data.get("short"))
            for tf, data in confirmations.items()
            if data.get("short", {}).get("confirmed") and data.get("short", {}).get("details") and (not allowed_entry_tfs or tf in allowed_entry_tfs)
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
    )
    summary["should_enter"] = pos is not None
    return pos, summary

def analyze_and_maybe_trade(symbol):
    if ctx.strategy_mode == "ema_macd":
        return analyze_ema_macd_and_maybe_trade(symbol)
    return analyze_momentum_and_maybe_trade(symbol)

# ---------- BACKTESTER ----------
def simulate_trade_on_series(df, entry_idx, entry_side, sl_price, tp_price, entry_price=None):
    n = len(df)
    entry_ref_price = entry_price if entry_price is not None else df['close'].iloc[entry_idx]

    # risco por unidade
    risk_per_unit = entry_ref_price - sl_price if entry_side == 'buy' else sl_price - entry_ref_price
    if risk_per_unit <= 0:
        risk_per_unit = 1e-8

    partial_done = False
    be_moved = False
    current_sl = sl_price

    for i in range(entry_idx+1, n):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if entry_side == 'buy':
            max_price = high
            rr_reached = (max_price - entry_ref_price) / risk_per_unit
            if (not be_moved) and rr_reached >= ctx.break_even_rr:
                current_sl = entry_ref_price
                be_moved = True
            if (not partial_done) and rr_reached >= ctx.take_partial_rr:
                partial_done = True
            hit_tp = (high >= tp_price)
            hit_sl = (low <= current_sl)
            if hit_tp and not hit_sl:
                if partial_done:
                    price_partial = entry_ref_price + ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (price_partial - entry_ref_price) * 0.5
                    pnl_rest = (tp_price - entry_ref_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = tp_price - entry_ref_price
                return {"exit_price": tp_price, "exit_idx": i, "outcome":"tp", "pnl": pnl_total}
            if hit_sl and not hit_tp:
                if partial_done:
                    price_partial = entry_ref_price + ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (price_partial - entry_ref_price) * 0.5
                    pnl_rest = (current_sl - entry_ref_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = current_sl - entry_ref_price
                return {"exit_price": current_sl, "exit_idx": i, "outcome":"sl", "pnl": pnl_total}
            if hit_tp and hit_sl:
                o = df['open'].iloc[i]
                dist_tp = abs(tp_price - o)
                dist_sl = abs(current_sl - o)
                outcome = "tp" if dist_tp <= dist_sl else "sl"
                exit_price = tp_price if outcome=="tp" else current_sl
                if partial_done:
                    price_partial = entry_ref_price + ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (price_partial - entry_ref_price) * 0.5
                    pnl_rest = (exit_price - entry_ref_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = exit_price - entry_ref_price
                return {"exit_price": exit_price, "exit_idx": i, "outcome": outcome, "pnl": pnl_total}
        else:
            min_price = low
            rr_reached = (entry_ref_price - min_price) / risk_per_unit
            if (not be_moved) and rr_reached >= ctx.break_even_rr:
                current_sl = entry_ref_price
                be_moved = True
            if (not partial_done) and rr_reached >= ctx.take_partial_rr:
                partial_done = True
            hit_tp = (low <= tp_price)
            hit_sl = (high >= current_sl)
            if hit_tp and not hit_sl:
                if partial_done:
                    price_partial = entry_ref_price - ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (entry_ref_price - price_partial) * 0.5
                    pnl_rest = (entry_ref_price - tp_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = entry_ref_price - tp_price
                return {"exit_price": tp_price, "exit_idx": i, "outcome":"tp", "pnl": pnl_total}
            if hit_sl and not hit_tp:
                if partial_done:
                    price_partial = entry_ref_price - ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (entry_ref_price - price_partial) * 0.5
                    pnl_rest = (entry_ref_price - current_sl) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = entry_ref_price - current_sl
                return {"exit_price": current_sl, "exit_idx": i, "outcome":"sl", "pnl": pnl_total}
            if hit_tp and hit_sl:
                o = df['open'].iloc[i]
                dist_tp = abs(o - tp_price)
                dist_sl = abs(o - current_sl)
                outcome = "tp" if dist_tp <= dist_sl else "sl"
                exit_price = tp_price if outcome=="tp" else current_sl
                if partial_done:
                    price_partial = entry_ref_price - ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (entry_ref_price - price_partial) * 0.5
                    pnl_rest = (entry_ref_price - exit_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = entry_ref_price - exit_price
                return {"exit_price": exit_price, "exit_idx": i, "outcome": outcome, "pnl": pnl_total}

    # se nunca bateu SL/TP, sai no close final
    final_price = df['close'].iloc[-1]
    if entry_side == 'buy':
        if partial_done:
            price_partial = entry_ref_price + ctx.take_partial_rr * risk_per_unit
            pnl_partial = (price_partial - entry_ref_price) * 0.5
            pnl_rest = (final_price - entry_ref_price) * 0.5
            pnl_total = pnl_partial + pnl_rest
        else:
            pnl_total = final_price - entry_ref_price
    else:
        if partial_done:
            price_partial = entry_ref_price - ctx.take_partial_rr * risk_per_unit
            pnl_partial = (entry_ref_price - price_partial) * 0.5
            pnl_rest = (entry_ref_price - final_price) * 0.5
            pnl_total = pnl_partial + pnl_rest
        else:
            pnl_total = entry_ref_price - final_price
    return {"exit_price": final_price, "exit_idx": n-1, "outcome": "none", "pnl": pnl_total}

def backtest_pair(symbol, timeframe, lookback_days=90, strategy=None, bias=None, cross_lookback=None, require_divergence=None):
    strategy = strategy or ctx.strategy_mode
    effective_bias = bias or ctx.trade_bias
    cross_lb = cross_lookback if cross_lookback is not None else ctx.ema_macd_cross_lookback
    divergence_required = ctx.ema_macd_require_divergence if require_divergence is None else require_divergence
    active_timeframes = [tf for tf in ctx.timeframes if tf == timeframe] or [timeframe]
    logging.info("Backtest %s %s last %d days", symbol, timeframe, lookback_days)
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = int((24*60/ minutes) * lookback_days) + 200
    df_main, main_cov = fetch_backtest_series(symbol, timeframe, candles_needed)
    coverage_info = {timeframe: main_cov}
    if df_main is None or len(df_main) < 200:
        logging.error("Dados insuficientes para backtest.")
        return None, coverage_info

    sim_risk = ctx.simulation_risk_per_trade
    current_equity = ctx.simulation_base_capital

    slice_store = {}
    for tf in ctx.timeframes:
        tf_minutes = timeframe_to_minutes(tf)
        candles_tf = int((24*60/ tf_minutes) * lookback_days) + 200
        if tf == timeframe:
            slice_store[tf] = df_main
            coverage_info[tf] = main_cov
            continue
        df_tf, cov_tf = fetch_backtest_series(symbol, tf, candles_tf)
        slice_store[tf] = df_tf
        coverage_info[tf] = cov_tf

    results = []
    bt_loss_streaks = defaultdict(int)
    for i in range(200, len(df_main)-1):
        ts = df_main.index[i]
        slices = {}
        for tf, dfrag in slice_store.items():
            if dfrag is None:
                slices[tf] = None
                continue
            d = dfrag[dfrag.index <= ts].tail(ctx.lookback)
            slices[tf] = d if len(d) >= 60 else None
        bias_snapshot = {tf: bias_from_slices(tf, ts, slice_store) for tf in ctx.timeframes}
        selected_trade = None

        if strategy == "momentum":
            if effective_bias == "short":
                continue
            momentum_ready = []
            for tf in active_timeframes:
                slice_df = slices.get(tf)
                if slice_df is None or len(slice_df) < 50:
                    continue
                bias_dir = bias_snapshot.get(tf, "both")
                has_div = has_bullish_rsi_divergence(slice_df)
                confirmed, details = momentum_confirm(slice_df)
                if not confirmed or not details.get("atr"):
                    continue
                if has_div:
                    details["score"] = details.get("score", 0) + 0.5
                if bias_dir == "down" and details.get("score", 0) < ctx.momentum_bias_override_score:
                    continue
                momentum_ready.append((tf, details))
            required = max(1, min(ctx.momentum_min_tf_agree, len(momentum_ready)))
            if len(momentum_ready) < required:
                continue
            momentum_ready.sort(key=lambda item: item[1].get("score", 0), reverse=True)
            ref_tf, detail = momentum_ready[0]
            entry_price = detail.get("last_close")
            atr_val = detail.get("atr")
            if entry_price is None or atr_val in (None, 0):
                continue
            stop_price = entry_price - atr_val * ctx.momentum_stop_atr
            tp_price = entry_price + atr_val * ctx.momentum_tp_atr
            rr = compute_rr(entry_price, stop_price, tp_price, side='buy')
            if rr is None or rr < ctx.min_rr_required:
                continue
            selected_trade = {
                "tf": ref_tf,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "rr": rr,
                "side": "buy",
                "risk_key": "momentum_long",
                "direction": "long",
                "details": detail,
            }
        elif strategy == "ema_macd":
            candidates = []
            if effective_bias in ("long", "both"):
                for tf in active_timeframes:
                    slice_df = slices.get(tf)
                    if slice_df is None or len(slice_df) < ctx.ema_slow_period + 10:
                        continue
                    confirmed_long, details_long = ema_macd_confirm(
                        slice_df,
                        cross_lookback=cross_lb,
                        require_divergence=divergence_required,
                    )
                    if not confirmed_long or not details_long.get("atr"):
                        continue
                    if not bias_allows_long(tf, bias_snapshot) and details_long.get("score", 0) < ctx.ema_macd_bias_override_score:
                        continue
                    entry_price = details_long.get("last_close")
                    atr_val = details_long.get("atr")
                    if entry_price is None or atr_val in (None, 0):
                        continue
                    stop_price = entry_price - atr_val * ctx.ema_macd_stop_atr
                    tp_price = entry_price + atr_val * ctx.ema_macd_tp_atr
                    rr = compute_rr(entry_price, stop_price, tp_price, side='buy')
                    if rr is None or rr < ctx.min_rr_required:
                        continue
                    candidates.append({
                        "tf": tf,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "tp_price": tp_price,
                        "rr": rr,
                        "score": details_long.get("score", 0),
                        "side": "buy",
                        "risk_key": "ema_macd_long",
                        "direction": "long",
                        "details": details_long,
                    })
            if effective_bias in ("short", "both"):
                for tf in active_timeframes:
                    slice_df = slices.get(tf)
                    if slice_df is None or len(slice_df) < ctx.ema_slow_period + 10:
                        continue
                    confirmed_short, details_short = ema_macd_confirm_short(
                        slice_df,
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
                        cross_lookback=cross_lb,
                        require_divergence=divergence_required,
                    )
                    if not confirmed_short and logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(
                            "EMA+MACD short rejeitado (bt) %s %s @ %s — detalhes: %s",
                            symbol,
                            tf,
                            ts,
                            details_short,
                        )
                    if not confirmed_short or not details_short.get("atr"):
                        continue
                    if not bias_allows_short(tf, bias_snapshot) and details_short.get("score", 0) < ctx.ema_macd_bias_override_score:
                        continue
                    entry_price = details_short.get("last_close")
                    atr_val = details_short.get("atr")
                    if entry_price is None or atr_val in (None, 0):
                        continue
                    stop_price = entry_price + atr_val * ctx.ema_macd_stop_atr
                    tp_price = entry_price - atr_val * ctx.ema_macd_tp_atr
                    rr = compute_rr(entry_price, stop_price, tp_price, side='sell')
                    if rr is None or rr < ctx.min_rr_required:
                        continue
                    candidates.append({
                        "tf": tf,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "tp_price": tp_price,
                        "rr": rr,
                        "score": details_short.get("score", 0),
                        "side": "sell",
                        "risk_key": "ema_macd_short",
                        "direction": "short",
                        "details": details_short,
                    })
            if not candidates:
                continue
            candidates.sort(key=lambda c: (c.get("score", 0), c.get("rr", 0)), reverse=True)
            selected_trade = candidates[0]
        else:
            logging.warning("Estratégia %s não suportada no backtest.", strategy)
            break

        if selected_trade is None:
            continue

        entry_price = selected_trade["entry_price"]
        stop_price = selected_trade["stop_price"]
        tp_price = selected_trade["tp_price"]
        rr_value = selected_trade["rr"]
        entry_side = selected_trade["side"]
        risk_key = selected_trade["risk_key"]
        trade_direction = selected_trade.get("direction", "long")
        entry_idx = i
        risk_mult = get_risk_multiplier(symbol, risk_key, store=bt_loss_streaks)
        effective_risk = max(sim_risk * risk_mult, sim_risk * 0.25)
        position_size = compute_size(current_equity, entry_price, stop_price, effective_risk)
        if position_size <= 0:
            continue
        outcome = simulate_trade_on_series(df_main, entry_idx, entry_side, stop_price, tp_price, entry_price)
        pnl_price = outcome['pnl']
        pnl_value = pnl_price * position_size
        actual_risk_value = position_size * abs(stop_price - entry_price)
        planned_risk_value = current_equity * effective_risk
        trade = {
            "timestamp": ts,
            "symbol": symbol,
            "strategy": risk_key,
            "base_strategy": strategy,
            "direction": trade_direction,
            "side": entry_side,
            "timeframe": selected_trade.get("tf"),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "tp_price": tp_price,
            "exit_price": outcome['exit_price'],
            "exit_idx": outcome['exit_idx'],
            "outcome": outcome['outcome'],
            "pnl_price": pnl_price,
            "position_size": position_size,
            "risk_amount": actual_risk_value,
            "risk_budget": planned_risk_value,
            "pnl": pnl_value,
            "rr": rr_value,
            "risk_multiplier": risk_mult,
            "risk_per_trade": effective_risk,
        }
        if pnl_value > 1e-8:
            trade["net_result"] = "win"
        elif pnl_value < -1e-8:
            trade["net_result"] = "loss"
        else:
            trade["net_result"] = "breakeven"
        results.append(trade)
        current_equity += pnl_value
        if outcome['outcome'] == 'tp' or outcome['pnl'] > 0:
            outcome_label = "win"
        elif outcome['outcome'] == 'sl' or outcome['pnl'] < 0:
            outcome_label = "loss"
        else:
            outcome_label = "breakeven"
        register_trade_outcome(symbol, risk_key, outcome_label, store=bt_loss_streaks)
    if len(results) == 0:
        logging.info("Nenhuma operação gerada no backtest.")
        return pd.DataFrame(), coverage_info
    df_trades = pd.DataFrame(results)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    breakevens = df_trades[df_trades['pnl'].abs() <= 1e-8]
    total_pnl = df_trades['pnl'].sum()
    eligible = len(wins) + len(losses)
    winrate = (len(wins) / eligible) if eligible > 0 else 0
    avg_win = wins['pnl'].mean() if len(wins)>0 else 0
    avg_loss = losses['pnl'].mean() if len(losses)>0 else 0
    logging.info(
        "Backtest summary: trades=%d winrate=%.2f total_pnl=%.4f avg_win=%.4f avg_loss=%.4f breakevens=%d",
        len(df_trades), winrate, total_pnl, avg_win, avg_loss, len(breakevens)
    )
    return df_trades, coverage_info

# ---------- Main loop ----------
def main_loop(symbols: Optional[List[str]] = None):
    active_symbols = list(symbols) if symbols else list(ctx.pairs)
    reset_runtime_state()
    logging.info(
        "Bot iniciado — pares: %s — timeframes: %s — entry_tfs: %s",
        active_symbols,
        ctx.timeframes,
        ctx.entry_timeframes,
    )
    while True:
        start = time.time()
        iteration_snapshot = []
        for symbol in active_symbols:
            try:
                pos, meta = analyze_and_maybe_trade(symbol)
                iteration_snapshot.append(meta)
            except Exception as e:
                logging.exception("Erro ao analisar %s: %s", symbol, e)
                iteration_snapshot.append({"symbol": symbol, "status": "error", "error": str(e)})
            time.sleep(0.5)
        # monitor open positions for addon/SL/TP
        try:
            monitor_and_close_positions(state.open_positions)
        except Exception as e:
            logging.exception("Erro no monitor loop: %s", e)
        write_runtime_state(serialize_open_positions(), iteration_snapshot)
        elapsed = time.time() - start
        sleep_for = max(0, ctx.loop_interval_seconds - elapsed)
        logging.info("Iteração completa. A dormir %.1f s", sleep_for)
        time.sleep(sleep_for)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Momentum breakout bot — backtest or live run')
    parser.add_argument('--live', dest='live', action='store_true', help='Run in live mode (main loop)')
    parser.add_argument('--symbol', dest='symbol', default='ETH/USDC:USDC', help='Symbol for backtest or live run')
    parser.add_argument('--timeframe', dest='timeframe', default='15m', help='Timeframe for the backtest (e.g., 15m)')
    parser.add_argument('--lookback-days', dest='lookback_days', type=int, default=60, help='Lookback days for backtest')
    paper_group = parser.add_mutually_exclusive_group()
    paper_group.add_argument('--paper-mode', dest='paper', action='store_const', const=True, help='Força modo paper (test) mesmo se o config estiver em live')
    paper_group.add_argument('--no-paper', dest='paper', action='store_const', const=False, help='Disable paper mode (will execute real orders)')
    parser.add_argument('--strategy', choices=['momentum','ema_macd'], default=ctx.strategy_mode, help='Seleciona estratégia principal')
    parser.add_argument('--cross-lookback', dest='cross_lookback', type=int, default=ctx.ema_macd_cross_lookback, help='Janela (nº de velas) para aceitar cruzamentos EMA/MACD recentes')
    parser.add_argument('--trade-bias', choices=['long','short','both'], default=ctx.trade_bias, help='Direção de trade: only long, only short ou ambos')
    parser.add_argument('--risk-percent', dest='risk_percent', type=float, help='Percentual do capital alocado a arriscar por trade (ex.: 1.0 para 1%).')
    parser.add_argument('--capital-base', dest='capital_base', type=float, help='Capital base máximo para cálculo de risco (padrão segue config).')
    parser.add_argument('--risk-mode', dest='risk_mode', choices=['standard','hunter'], help='Modo de risco: standard limita pelo capital base; hunter usa 100% do saldo.')
    parser.add_argument('--leverage', dest='leverage', type=float, help='Alavancagem alvo para limitar o tamanho máximo da posição.')
    parser.add_argument('--log-level', dest='log_level', default='INFO', help='Nível de logging (DEBUG, INFO, WARNING, ...)')
    divergence_group = parser.add_mutually_exclusive_group()
    divergence_group.add_argument('--require-divergence', dest='require_divergence', action='store_true', default=ctx.ema_macd_require_divergence, help='Exige divergência RSI para setups EMA+MACD (padrão)')
    divergence_group.add_argument('--allow-no-divergence', dest='require_divergence', action='store_false', help='Permite setups EMA+MACD mesmo sem divergência RSI')
    parser.add_argument('--check-symbol', dest='check_symbol', help='Valida se o par existe na corretora e termina imediatamente')
    parser.add_argument('--all-pairs', dest='all_pairs', action='store_true', help='Loop live cobre todos os pares configurados (ignora --symbol)')
    parser.set_defaults(paper=None)
    args = parser.parse_args()

    if args.check_symbol:
        symbol_to_check = args.check_symbol.strip()
        is_valid = validate_symbol(symbol_to_check)
        if is_valid:
            logging.info("Par %s disponível para negociação.", symbol_to_check)
            print(f"{symbol_to_check}: disponível")
            sys.exit(0)
        else:
            logging.error("Par %s não encontrado na corretora %s", symbol_to_check, exchange_id)
            print(f"{symbol_to_check}: indisponível")
            sys.exit(1)

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.info("Log level definido para %s", logging.getLevelName(log_level))

    logging.info("Script iniciado. ambiente configurado=%s (paper=%s)", ctx.environment_mode, ctx.paper)

    if args.paper is not None:
        ctx.update_environment_mode("paper" if args.paper else "live")
        update_user_config(environment=ctx.environment_mode)
        logging.info("Ambiente ajustado via CLI para %s", ctx.environment_mode)

    if ctx.paper:
        logging.info("MODO PAPER ATIVO — ordens reais NÃO serão colocadas.")
    else:
        logging.warning("MODO REAL: As ordens reais serão enviadas — esteja certo do seu API_KEY/API_SECRET e capital.")

    cli_overrides = {
        "strategy_mode": args.strategy,
        "trade_bias": args.trade_bias,
        "ema_cross_lookback": args.cross_lookback,
        "ema_require_divergence": args.require_divergence,
    }
    if args.risk_percent is not None:
        cli_overrides["risk_percent"] = max(0.0001, args.risk_percent / 100.0)
    if args.capital_base is not None:
        cli_overrides["capital_base"] = max(0.0, args.capital_base)
    if args.risk_mode is not None:
        cli_overrides["risk_mode"] = args.risk_mode
    if args.leverage is not None:
        cli_overrides["leverage"] = max(1.0, args.leverage)

    ctx.override_from_cli(cli_overrides)

    logging.info("Estratégia ativa: %s", ctx.strategy_mode)
    logging.info("Bias de trade ativo: %s", ctx.trade_bias)
    logging.info("EMA/MACD cross lookback: %d velas", ctx.ema_macd_cross_lookback)
    logging.info("EMA/MACD exige divergência RSI: %s", "sim" if ctx.ema_macd_require_divergence else "não")
    logging.info("Modo de risco: %s", ctx.risk_mode)
    logging.info("Risco por trade: %.2f%%", ctx.risk_percent * 100)
    logging.info("Capital base para sizing: %.2f", ctx.capital_base)
    logging.info("Alavancagem alvo: %.2fx", ctx.leverage)

    update_user_config(
        environment=ctx.environment_mode,
        trade_bias=ctx.trade_bias,
        ema_cross_lookback=ctx.ema_macd_cross_lookback,
        ema_require_divergence=ctx.ema_macd_require_divergence,
        symbol=args.symbol,
        timeframe=args.timeframe,
        risk_mode=ctx.risk_mode,
        risk_percent=ctx.risk_percent,
        capital_base=ctx.capital_base,
        leverage=ctx.leverage,
    )

    if args.live:
        selected_symbols: List[str]
        if args.all_pairs:
            selected_symbols = list(ctx.pairs)
        elif args.symbol:
            selected_symbols = [args.symbol]
        else:
            selected_symbols = list(ctx.pairs)
        ctx.set_entry_timeframes([args.timeframe])
        logging.info(
            "Running live main loop for symbols=%s com timeframe base %s",
            selected_symbols,
            ctx.entry_timeframes,
        )
        # user should set API keys
        main_loop(symbols=selected_symbols)
    else:
        logging.info("Running backtest for %s %s (lookback %d days)", args.symbol, args.timeframe, args.lookback_days)
        try:
            df_trades, coverage_info = backtest_pair(
                args.symbol,
                args.timeframe,
                lookback_days=args.lookback_days,
                strategy=args.strategy,
                bias=ctx.trade_bias,
                cross_lookback=ctx.ema_macd_cross_lookback,
                require_divergence=ctx.ema_macd_require_divergence,
            )
            if df_trades is not None and not df_trades.empty:
                print(df_trades.head())
                df_trades.to_csv(f"backtest_{args.symbol.replace('/','_')}_{args.timeframe}_trades.csv", index=False)
                logging.info("Backtest guardado em CSV")
            write_backtest_summary(
                args.symbol,
                args.timeframe,
                args.lookback_days,
                df_trades if df_trades is not None else pd.DataFrame(),
                coverage=coverage_info,
                simulation={
                    "base_capital": ctx.simulation_base_capital,
                    "risk_per_trade": ctx.simulation_risk_per_trade,
                    "ema_cross_lookback": ctx.ema_macd_cross_lookback,
                    "ema_require_divergence": ctx.ema_macd_require_divergence,
                    "trade_bias": ctx.trade_bias,
                    "environment": ctx.environment_mode,
                }
            )
        except Exception as e:
            logging.error("Erro ao fazer backtest: %s", e)
