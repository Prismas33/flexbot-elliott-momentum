from __future__ import annotations

import logging
import time

from . import context


def place_market_order(symbol: str, side: str, qty: float):
    logging.info("ORDER MARKET %s %s %s", side, qty, symbol)
    if context.paper:
        return {
            "id": f"paper-{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "status": "paper",
            "price": None,
        }
    try:
        order = context.exchange.create_market_order(symbol, side.lower(), qty)
        return order
    except Exception as exc:
        logging.error("Erro colocar market order: %s", exc)
        return None


def place_conditional_orders(symbol: str, side: str, qty: float, stop_price: float, tp_price: float) -> bool:
    if context.paper:
        logging.info("PAPER: criar ordens condicionais (simuladas) for %s", symbol)
        return True
    try:
        params_sl = {"stopPrice": stop_price, "reduceOnly": True, "closeOnTrigger": True}
        params_tp = {"stopPrice": tp_price, "reduceOnly": True, "closeOnTrigger": True}
        exit_side = "sell" if side == "buy" else "buy"
        context.exchange.create_order(symbol, "stop", exit_side, qty, None, params_sl)
        context.exchange.create_order(symbol, "stop", exit_side, qty, None, params_tp)
        return True
    except Exception as exc:
        logging.warning("Falha ordens condicionais: %s", exc)
        return False
