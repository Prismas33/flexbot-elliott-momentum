from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import context
from .data import fetch_ohlcv
from .orders import place_conditional_orders, place_market_order
from .risk import get_risk_multiplier, register_trade_outcome, loss_streaks

open_positions: List[Dict[str, object]] = []
last_entry_tracker: Dict[Tuple[str, str, str], pd.Timestamp] = {}


def has_open_position(symbol: str) -> bool:
    for pos in open_positions:
        if pos.get("symbol") == symbol and (pos.get("qty_current") or 0) > 0:
            return True
    return False


def serialize_open_positions() -> List[Dict[str, object]]:
    serialized = []
    for pos in open_positions:
        serialized.append({
            "symbol": pos.get("symbol"),
            "side": pos.get("side"),
            "qty_current": pos.get("qty_current"),
            "qty_total": pos.get("qty_total"),
            "entry_price": pos.get("entry_price"),
            "stop": pos.get("stop"),
            "tp": pos.get("tp"),
            "opened_at": pos.get("opened_at").isoformat() if pos.get("opened_at") else None,
            "addon_pending": pos.get("addon_pending"),
            "strategy": pos.get("strategy", "unknown"),
            "partial_taken": pos.get("partial_taken", False),
            "breakeven_moved": pos.get("breakeven_moved", False),
            "risk_multiplier": pos.get("risk_multiplier", 1.0),
            "timeframe": pos.get("timeframe"),
            "min_price_seen": pos.get("min_price_seen"),
            "max_price_seen": pos.get("max_price_seen"),
            "risk_percent": pos.get("risk_percent"),
            "risk_base_percent": pos.get("risk_base_percent"),
            "risk_target_percent": pos.get("risk_target_percent"),
            "risk_allocation": pos.get("risk_allocation"),
            "risk_amount": pos.get("risk_amount"),
            "leverage": pos.get("leverage"),
            "margin_required": pos.get("margin_required"),
        })
    return serialized


def compute_size(account_balance: float, entry_price: float, stop_price: float, risk_per_trade: float = context.risk_percent) -> float:
    risk_amount = account_balance * risk_per_trade
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        return 0.0
    qty = risk_amount / risk_per_unit
    return max(0.0, qty)


def compute_rr(entry_price: float, stop_price: float, tp_price: float, side: str = "buy") -> float | None:
    if side == "buy":
        risk = entry_price - stop_price
        reward = tp_price - entry_price
    else:
        risk = stop_price - entry_price
        reward = entry_price - tp_price
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def fetch_account_balance() -> float:
    try:
        bal = context.exchange.fetch_balance(params={"type": "future"})
        if isinstance(bal, dict):
            totals = bal.get("total", {})
            for cur in ["USDC", "USD", "USDT"]:
                if cur in totals and totals[cur] is not None:
                    return float(totals[cur])
    except Exception as exc:
        logging.warning("fetch_balance falhou: %s", exc)
    return context.account_balance_fallback


def get_environment_mode() -> str:
    return "paper" if context.paper else "live"


def validate_symbol(symbol: str) -> bool:
    try:
        markets = context.exchange.markets or {}
        if not markets:
            markets = context.exchange.load_markets()
        return symbol in markets
    except Exception as exc:
        logging.error("Falha ao validar par %s: %s", symbol, exc)
        return False


def enter_position(
    symbol: str,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    account_balance: float,
    *,
    side: str,
    strategy_label: str,
    risk_multiplier: Optional[float] = None,
    bar_time=None,
    timeframe=None,
):
    if has_open_position(symbol):
        logging.info("%s trade ignorado — já existe posição aberta em %s", strategy_label, symbol)
        return None
    if bar_time is not None and timeframe is not None:
        key = (symbol, strategy_label, timeframe)
        last_bar = last_entry_tracker.get(key)
        if last_bar is not None and bar_time <= last_bar:
            logging.info("%s trade ignorado — já foi executado nesta vela (%s @ %s)", strategy_label, symbol, bar_time)
            return None
    manual_multiplier = risk_multiplier if risk_multiplier is not None else 1.0
    streak_multiplier = get_risk_multiplier(symbol, strategy_label)
    effective_multiplier = manual_multiplier * streak_multiplier

    base_percent = max(0.0, context.risk_percent)

    if context.risk_mode != "hunter" and context.risk_floor_fraction > 0:
        floor = base_percent * context.risk_floor_fraction
    else:
        floor = 0.0

    risk_percent = base_percent * effective_multiplier
    if floor > 0:
        risk_percent = max(risk_percent, floor)
    risk_percent = min(risk_percent, 1.0)

    if context.risk_mode == "hunter":
        allocation_balance = max(0.0, account_balance)
    else:
        cap_base = context.capital_base if context.capital_base is not None else 0.0
        if cap_base <= 0:
            allocation_balance = max(0.0, account_balance)
        else:
            allocation_balance = min(max(0.0, cap_base), max(0.0, account_balance))

    qty_requested = compute_size(allocation_balance, entry_price, stop_price, risk_percent)
    qty = max(0.0, qty_requested)
    leverage = max(float(context.leverage), 1.0)
    if entry_price > 0:
        max_qty_by_margin = (account_balance * leverage) / entry_price
        if max_qty_by_margin > 0 and qty > max_qty_by_margin:
            logging.debug(
                "Quantidade limitada pela margem: %.6f -> %.6f (leverage %.2fx)",
                qty,
                max_qty_by_margin,
                leverage,
            )
            qty = max_qty_by_margin

    risk_per_unit = abs(entry_price - stop_price)
    risk_amount = risk_per_unit * qty
    margin_required = (entry_price * qty) / leverage if entry_price > 0 else 0.0
    risk_percent_actual = (risk_amount / allocation_balance) if allocation_balance > 0 else 0.0
    if risk_percent_actual > 1.0:
        risk_percent_actual = 1.0

    if qty == 0:
        logging.info(
            "%s trade abortado — qty=0 (capital alocado=%.2f, risco alvo=%.2f%%)",
            strategy_label,
            allocation_balance,
            risk_percent * 100,
        )
        return None
    order = place_market_order(symbol, "buy" if side == "buy" else "sell", qty)
    if order is None:
        return None
    pos = {
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "qty_total": qty,
        "qty_partial": qty,
        "qty_addon": 0.0,
        "qty_current": qty,
        "stop": stop_price,
        "tp": tp_price,
        "opened_at": datetime.utcnow(),
        "addon_pending": False,
        "strategy": strategy_label,
        "partial_taken": False,
        "breakeven_moved": False,
        "risk_multiplier": effective_multiplier,
        "timeframe": timeframe,
        "min_price_seen": entry_price,
        "max_price_seen": entry_price,
        "risk_percent": risk_percent_actual,
        "risk_base_percent": base_percent,
        "risk_target_percent": risk_percent,
        "risk_allocation": allocation_balance,
        "risk_amount": risk_amount,
        "leverage": leverage,
        "margin_required": margin_required,
    }
    open_positions.append(pos)
    place_conditional_orders(symbol, side, qty, stop_price, tp_price)
    if bar_time is not None and timeframe is not None:
        last_entry_tracker[(symbol, strategy_label, timeframe)] = bar_time
    return pos


def monitor_and_close_positions(position_store: List[Dict[str, object]]):
    closed = []
    for pos in list(position_store):
        df = fetch_ohlcv(pos["symbol"], "1m", limit=10)
        if df is None or df.empty:
            continue
        high = float(df["high"].max())
        low = float(df["low"].min())
        last_close = float(df["close"].iloc[-1]) if len(df["close"]) else pos["entry_price"]

        entry = pos["entry_price"]
        stop = pos["stop"]
        tp = pos["tp"]

        pos["max_price_seen"] = max(pos.get("max_price_seen", entry), high)
        pos["min_price_seen"] = min(pos.get("min_price_seen", entry), low)

        if stop is not None and tp is not None and entry is not None:
            risk_per_unit = entry - stop if pos["side"] == "buy" else stop - entry
            if risk_per_unit > 0:
                if pos["side"] == "buy":
                    progress_price = pos["max_price_seen"]
                    rr_reached = (progress_price - entry) / risk_per_unit
                else:
                    progress_price = pos["min_price_seen"]
                    rr_reached = (entry - progress_price) / risk_per_unit
            else:
                rr_reached = 0
        else:
            rr_reached = 0

        if (not pos.get("breakeven_moved", False)) and rr_reached >= context.break_even_rr and entry is not None:
            pos["stop"] = entry
            pos["breakeven_moved"] = True
            logging.info("Move SL para break-even em %s (%.2fR)", pos["symbol"], rr_reached)

        if (not pos.get("partial_taken", False)) and rr_reached >= context.take_partial_rr:
            qty_close = pos["qty_current"] * 0.5
            if qty_close > 0:
                side = "sell" if pos["side"] == "buy" else "buy"
                if context.paper:
                    logging.info("PAPER: parcial em %s, fechar %.6f", pos["symbol"], qty_close)
                    pos["qty_current"] -= qty_close
                    pos["partial_taken"] = True
                else:
                    try:
                        close_order = context.exchange.create_market_order(pos["symbol"], side, qty_close)
                        logging.info("Parcial fechada %s: %s", pos["symbol"], close_order)
                        pos["qty_current"] -= qty_close
                        pos["partial_taken"] = True
                    except Exception as exc:
                        logging.error("Erro fechar parcial em %s: %s", pos["symbol"], exc)

        if pos["side"] == "buy":
            hit_tp = (tp is not None) and (pos["max_price_seen"] >= tp)
            hit_sl = (stop is not None) and (pos["min_price_seen"] <= stop)
        else:
            hit_tp = (tp is not None) and (pos["min_price_seen"] <= tp)
            hit_sl = (stop is not None) and (pos["max_price_seen"] >= stop)
        if hit_tp or hit_sl:
            side = "sell" if pos["side"] == "buy" else "buy"
            qty = pos["qty_current"]
            if qty <= 0:
                outcome_label = "breakeven"
                register_trade_outcome(pos["symbol"], pos.get("strategy", "unknown"), outcome_label)
                closed.append(pos)
                position_store.remove(pos)
                continue
            if hit_tp and hit_sl:
                ref_price = last_close
                dist_tp = abs((tp or ref_price) - ref_price)
                dist_sl = abs((stop or ref_price) - ref_price)
                outcome_label = "win" if dist_tp <= dist_sl else "loss"
            elif hit_tp:
                outcome_label = "win"
            elif hit_sl:
                if pos.get("breakeven_moved") and entry is not None and stop is not None and abs(stop - entry) <= max(1e-6, entry * 0.0001):
                    outcome_label = "breakeven"
                else:
                    outcome_label = "loss"
            else:
                outcome_label = "breakeven"
            streak = register_trade_outcome(pos["symbol"], pos.get("strategy", "unknown"), outcome_label)
            if outcome_label == "loss" and streak >= context.loss_streak_limit:
                base_pct = context.risk_percent * 100
                logging.info(
                    "Loss streak %d em %s (%s) — reduzir risco para %.2f%%",
                    streak,
                    pos["symbol"],
                    pos.get("strategy", "unknown"),
                    base_pct * context.loss_streak_risk_factor,
                )
            if context.paper:
                logging.info("PAPER: fechar pos %s por %s (tp/sl atingido).", pos["symbol"], "TP" if hit_tp else "SL")
                closed.append(pos)
                position_store.remove(pos)
            else:
                try:
                    close_order = context.exchange.create_market_order(pos["symbol"], side, qty)
                    logging.info("Fechada pos %s: %s", pos["symbol"], close_order)
                    closed.append(pos)
                    position_store.remove(pos)
                except Exception as exc:
                    logging.error("Erro fechar pos %s: %s", pos["symbol"], exc)
            if pos in closed:
                key = (pos.get("symbol"), pos.get("strategy"), pos.get("timeframe"))
                last_entry_tracker.pop(key, None)
    return closed


def reset_runtime_state():
    open_positions.clear()
    last_entry_tracker.clear()
    loss_streaks.clear()
