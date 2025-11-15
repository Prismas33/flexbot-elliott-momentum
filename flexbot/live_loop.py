"""Live trading loop utilities."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from state_store import write_runtime_state

from . import context as ctx, state
from .analyzer import analyze_and_maybe_trade

__all__ = ["main_loop"]


def main_loop(symbols: Optional[List[str]] = None) -> None:
    """Continuously scan configured symbols and dispatch trades."""
    default_pairs = list(ctx.active_pairs) if getattr(ctx, "active_pairs", None) else []
    active_symbols = list(symbols) if symbols else default_pairs
    if not active_symbols:
        active_symbols = list(ctx.pairs)
    if not active_symbols:
        active_symbols = list(ctx.pairs)
    state.reset_runtime_state()
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
            except Exception as exc:  # pragma: no cover - safeguard for runtime
                logging.exception("Erro ao analisar %s: %s", symbol, exc)
                iteration_snapshot.append({"symbol": symbol, "status": "error", "error": str(exc)})
            time.sleep(0.5)
        try:
            state.monitor_and_close_positions(state.open_positions)
        except Exception as exc:  # pragma: no cover - safeguard for runtime
            logging.exception("Erro no monitor loop: %s", exc)
        write_runtime_state(state.serialize_open_positions(), iteration_snapshot)
        elapsed = time.time() - start
        sleep_for = max(0, ctx.loop_interval_seconds - elapsed)
        logging.info("Iteração completa. A dormir %.1f s", sleep_for)
        time.sleep(sleep_for)
