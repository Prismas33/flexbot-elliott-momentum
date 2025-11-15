from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

from . import context

loss_streaks = defaultdict(int)


def _loss_key(symbol: str, strategy: str) -> Tuple[str, str]:
    return (symbol, strategy)


def get_risk_multiplier(symbol: str, strategy: str, store: Optional[Dict[Tuple[str, str], int]] = None) -> float:
    book = store if store is not None else loss_streaks
    key = _loss_key(symbol, strategy)
    streak = book[key]
    if streak >= context.loss_streak_limit:
        return context.loss_streak_risk_factor
    return 1.0


def register_trade_outcome(symbol: str, strategy: str, outcome: str, store: Optional[Dict[Tuple[str, str], int]] = None) -> int:
    book = store if store is not None else loss_streaks
    key = _loss_key(symbol, strategy)
    if outcome == "loss":
        book[key] += 1
    else:
        book[key] = 0
    return book[key]
