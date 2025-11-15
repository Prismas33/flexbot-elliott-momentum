"""Backtesting utilities for FlexBot."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import bias, context as ctx, data, divergence, indicators, state
from .risk import get_risk_multiplier, register_trade_outcome
from .strategies import momentum_confirm_long, momentum_confirm_short
from .strategies.ema_macd import ema_macd_confirm
from .strategies.ema_macd_short import ema_macd_confirm_short

__all__ = ["simulate_trade_on_series", "backtest_pair", "backtest_many"]

fetch_backtest_series = data.fetch_backtest_series
timeframe_to_minutes = data.timeframe_to_minutes
bias_allows_long = bias.bias_allows_long
bias_allows_short = bias.bias_allows_short
bias_from_slices = bias.bias_from_slices
has_bullish_rsi_divergence = divergence.has_bullish_rsi_divergence
has_bearish_rsi_divergence = divergence.has_bearish_rsi_divergence
compute_atr = indicators.compute_atr
compute_rsi = indicators.compute_rsi
compute_macd = indicators.compute_macd


def simulate_trade_on_series(df: pd.DataFrame, entry_idx: int, entry_side: str, sl_price: float, tp_price: float, entry_price: Optional[float] = None):
    """Replays candles after entry to determine exit outcome for a single trade."""
    n = len(df)
    entry_ref_price = entry_price if entry_price is not None else df['close'].iloc[entry_idx]

    risk_per_unit = entry_ref_price - sl_price if entry_side == 'buy' else sl_price - entry_ref_price
    if risk_per_unit <= 0:
        risk_per_unit = 1e-8

    partial_done = False
    be_moved = False
    current_sl = sl_price

    for i in range(entry_idx + 1, n):
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
                return {"exit_price": tp_price, "exit_idx": i, "outcome": "tp", "pnl": pnl_total}
            if hit_sl and not hit_tp:
                if partial_done:
                    price_partial = entry_ref_price + ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (price_partial - entry_ref_price) * 0.5
                    pnl_rest = (current_sl - entry_ref_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = current_sl - entry_ref_price
                return {"exit_price": current_sl, "exit_idx": i, "outcome": "sl", "pnl": pnl_total}
            if hit_tp and hit_sl:
                o = df['open'].iloc[i]
                dist_tp = abs(tp_price - o)
                dist_sl = abs(current_sl - o)
                outcome = "tp" if dist_tp <= dist_sl else "sl"
                exit_price = tp_price if outcome == "tp" else current_sl
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
                return {"exit_price": tp_price, "exit_idx": i, "outcome": "tp", "pnl": pnl_total}
            if hit_sl and not hit_tp:
                if partial_done:
                    price_partial = entry_ref_price - ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (entry_ref_price - price_partial) * 0.5
                    pnl_rest = (entry_ref_price - current_sl) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = entry_ref_price - current_sl
                return {"exit_price": current_sl, "exit_idx": i, "outcome": "sl", "pnl": pnl_total}
            if hit_tp and hit_sl:
                o = df['open'].iloc[i]
                dist_tp = abs(o - tp_price)
                dist_sl = abs(o - current_sl)
                outcome = "tp" if dist_tp <= dist_sl else "sl"
                exit_price = tp_price if outcome == "tp" else current_sl
                if partial_done:
                    price_partial = entry_ref_price - ctx.take_partial_rr * risk_per_unit
                    pnl_partial = (entry_ref_price - price_partial) * 0.5
                    pnl_rest = (entry_ref_price - exit_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = entry_ref_price - exit_price
                return {"exit_price": exit_price, "exit_idx": i, "outcome": outcome, "pnl": pnl_total}

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
    return {"exit_price": final_price, "exit_idx": n - 1, "outcome": "none", "pnl": pnl_total}


def backtest_pair(symbol: str, timeframe: str, lookback_days: int = 90, strategy: Optional[str] = None, bias: Optional[str] = None, cross_lookback: Optional[int] = None, require_divergence: Optional[bool] = None):
    """Runs the selected strategy on historical data and returns trade statistics."""
    strategy = strategy or ctx.strategy_mode
    effective_bias = bias or ctx.trade_bias
    cross_lb = cross_lookback if cross_lookback is not None else ctx.ema_macd_cross_lookback
    divergence_required = ctx.ema_macd_require_divergence if require_divergence is None else require_divergence
    active_timeframes = [tf for tf in ctx.timeframes if tf == timeframe] or [timeframe]
    logging.info("Backtest %s %s last %d days", symbol, timeframe, lookback_days)
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = int((24 * 60 / minutes) * lookback_days) + 200
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
        candles_tf = int((24 * 60 / tf_minutes) * lookback_days) + 200
        if tf == timeframe:
            slice_store[tf] = df_main
            coverage_info[tf] = main_cov
            continue
        df_tf, cov_tf = fetch_backtest_series(symbol, tf, candles_tf)
        slice_store[tf] = df_tf
        coverage_info[tf] = cov_tf

    results = []
    bt_loss_streaks = defaultdict(int)
    i = 200
    last_index = len(df_main) - 1
    while i < last_index:
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
            candidates = []
            long_ready = []
            short_ready = []
            for tf in active_timeframes:
                slice_df = slices.get(tf)
                if slice_df is None or len(slice_df) < 50:
                    continue
                bias_dir = bias_snapshot.get(tf, "both")
                if effective_bias in ("long", "both"):
                    has_div_long = has_bullish_rsi_divergence(slice_df)
                    confirmed_long, details_long = momentum_confirm_long(slice_df)
                    if confirmed_long and has_div_long:
                        details_long["score"] = details_long.get("score", 0) + 1.0
                    if confirmed_long and bias_dir == "down" and details_long.get("score", 0) < ctx.momentum_bias_override_score:
                        confirmed_long = False
                    if confirmed_long and details_long.get("atr"):
                        long_ready.append((tf, details_long, bias_dir))
                if effective_bias in ("short", "both"):
                    has_div_short = has_bearish_rsi_divergence(slice_df)
                    confirmed_short, details_short = momentum_confirm_short(slice_df)
                    if confirmed_short and has_div_short:
                        details_short["score"] = details_short.get("score", 0) + 1.0
                    if confirmed_short and bias_dir == "up" and details_short.get("score", 0) < ctx.momentum_bias_override_score:
                        confirmed_short = False
                    if confirmed_short and details_short.get("atr"):
                        short_ready.append((tf, details_short, bias_dir))

            if long_ready:
                required_long = max(1, min(ctx.momentum_min_tf_agree, len(long_ready)))
                if len(long_ready) >= required_long:
                    long_ready.sort(key=lambda item: item[1].get("score", 0), reverse=True)
                    ref_tf, detail_long, _ = long_ready[0]
                    entry_price = detail_long.get("last_close")
                    atr_val = detail_long.get("atr")
                    if entry_price is not None and atr_val not in (None, 0):
                        stop_price = entry_price - atr_val * ctx.momentum_stop_atr
                        tp_price = entry_price + atr_val * ctx.momentum_tp_atr
                        rr_val = state.compute_rr(entry_price, stop_price, tp_price, side='buy')
                        if rr_val is not None and rr_val >= ctx.min_rr_required:
                            candidates.append({
                                "tf": ref_tf,
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "rr": rr_val,
                                "side": "buy",
                                "risk_key": "momentum_long",
                                "direction": "long",
                                "details": detail_long,
                                "score": detail_long.get("score", 0),
                            })

            if short_ready:
                required_short = max(1, min(ctx.momentum_min_tf_agree, len(short_ready)))
                if len(short_ready) >= required_short:
                    short_ready.sort(key=lambda item: item[1].get("score", 0), reverse=True)
                    ref_tf, detail_short, _ = short_ready[0]
                    entry_price = detail_short.get("last_close")
                    atr_val = detail_short.get("atr")
                    if entry_price is not None and atr_val not in (None, 0):
                        stop_price = entry_price + atr_val * ctx.momentum_stop_atr
                        tp_price = entry_price - atr_val * ctx.momentum_tp_atr
                        rr_val = state.compute_rr(entry_price, stop_price, tp_price, side='sell')
                        if rr_val is not None and rr_val >= ctx.min_rr_required:
                            candidates.append({
                                "tf": ref_tf,
                                "entry_price": entry_price,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "rr": rr_val,
                                "side": "sell",
                                "risk_key": "momentum_short",
                                "direction": "short",
                                "details": detail_short,
                                "score": detail_short.get("score", 0),
                            })

            if not candidates:
                i += 1
                continue
            candidates.sort(key=lambda c: (c.get("score", 0), c.get("rr", 0)), reverse=True)
            selected_trade = candidates[0]
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
                        require_rsi_zone=ctx.ema_require_rsi_zone,
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
                    rr = state.compute_rr(entry_price, stop_price, tp_price, side='buy')
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
                        require_rsi_zone=ctx.ema_require_rsi_zone,
                        rsi_zone_short_min=ctx.ema_rsi_zone_short_min,
                    )
                    if not confirmed_short and logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(
                            "OMDs short rejeitado (bt) %s %s @ %s — detalhes: %s",
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
                    rr = state.compute_rr(entry_price, stop_price, tp_price, side='sell')
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
                i += 1
                continue
            candidates.sort(key=lambda c: (c.get("score", 0), c.get("rr", 0)), reverse=True)
            selected_trade = candidates[0]
        else:
            logging.warning("Estratégia %s não suportada no backtest.", strategy)
            break

        if selected_trade is None:
            i += 1
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
        position_size = state.compute_size(current_equity, entry_price, stop_price, effective_risk)
        leverage = max(float(ctx.leverage), 1.0)
        if entry_price > 0:
            available_equity = max(current_equity, 0.0)
            max_qty_by_margin = (available_equity * leverage) / entry_price
            if max_qty_by_margin > 0 and position_size > max_qty_by_margin:
                logging.debug(
                    "Backtest: quantidade limitada pela margem: %.6f -> %.6f (leverage %.2fx)",
                    position_size,
                    max_qty_by_margin,
                    leverage,
                )
                position_size = max_qty_by_margin
        if position_size <= 0:
            i += 1
            continue
        outcome = simulate_trade_on_series(df_main, entry_idx, entry_side, stop_price, tp_price, entry_price)
        pnl_price = outcome['pnl']
        pnl_value = pnl_price * position_size
        risk_per_unit = abs(stop_price - entry_price)
        actual_risk_value = position_size * risk_per_unit
        risk_allocation = max(current_equity, 0.0)
        planned_risk_value = risk_allocation * effective_risk
        margin_required = (entry_price * position_size) / leverage if entry_price > 0 else 0.0
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
            "risk_allocation": risk_allocation,
            "leverage": leverage,
            "margin_required": margin_required,
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
        exit_idx = outcome.get("exit_idx")
        i = max(i + 1, exit_idx + 1) if isinstance(exit_idx, int) else i + 1
        continue
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
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    logging.info(
        "Backtest summary: trades=%d winrate=%.2f total_pnl=%.4f avg_win=%.4f avg_loss=%.4f breakevens=%d",
        len(df_trades), winrate, total_pnl, avg_win, avg_loss, len(breakevens)
    )
    return df_trades, coverage_info


def backtest_many(
    symbols: Sequence[str],
    timeframe: str,
    lookback_days: int = 90,
    *,
    strategy: Optional[str] = None,
    bias: Optional[str] = None,
    cross_lookback: Optional[int] = None,
    require_divergence: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Runs backtests sequentially for multiple symbols and aggregates the results."""
    aggregated: List[pd.DataFrame] = []
    coverage_bundle: Dict[str, object] = {}
    for symbol in symbols:
        df_trades, coverage = backtest_pair(
            symbol,
            timeframe,
            lookback_days=lookback_days,
            strategy=strategy,
            bias=bias,
            cross_lookback=cross_lookback,
            require_divergence=require_divergence,
        )
        coverage_bundle[symbol] = coverage
        if df_trades is None or df_trades.empty:
            continue
        df_copy = df_trades.copy()
        df_copy["symbol"] = symbol
        aggregated.append(df_copy)
    if not aggregated:
        return pd.DataFrame(), coverage_bundle
    combined = pd.concat(aggregated, ignore_index=True)
    if "timestamp" in combined.columns:
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)
    return combined, coverage_bundle
