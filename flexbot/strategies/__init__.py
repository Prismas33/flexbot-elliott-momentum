"""Strategy modules for FlexBot."""

from .momentum import analyze_momentum_and_maybe_trade, momentum_confirm
from .ema_macd import ema_macd_confirm

__all__ = [
    "analyze_momentum_and_maybe_trade",
    "momentum_confirm",
    "ema_macd_confirm",
]
