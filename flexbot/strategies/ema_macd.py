"""Helper for OMDs strategy configuration."""

from __future__ import annotations

from typing import Optional

from .. import context as ctx, indicators, divergence
from .ema_macd_long import ema_macd_confirm_long

__all__ = ["ema_macd_confirm"]


def ema_macd_confirm(
    df,
    cross_lookback: Optional[int] = None,
    require_divergence: Optional[bool] = None,
    require_rsi_zone: Optional[bool] = None,
    rsi_zone_long_max: Optional[float] = None,
):
    """Wrapper that applies context defaults before delegating to the long OMDs confirmation."""
    cross_lb = cross_lookback if cross_lookback is not None else ctx.ema_macd_cross_lookback
    divergence_required = ctx.ema_macd_require_divergence if require_divergence is None else require_divergence
    rsi_zone_required = ctx.ema_require_rsi_zone if require_rsi_zone is None else require_rsi_zone
    rsi_long_cap = ctx.ema_rsi_zone_long_max if rsi_zone_long_max is None else rsi_zone_long_max
    return ema_macd_confirm_long(
        df,
        ema_fast_period=ctx.ema_fast_period,
        ema_slow_period=ctx.ema_slow_period,
        ema_macd_fast=ctx.ema_macd_fast,
        ema_macd_slow=ctx.ema_macd_slow,
        ema_macd_signal=ctx.ema_macd_signal,
        atr_period=ctx.atr_period,
        rsi_period=ctx.rsi_period,
        compute_atr=indicators.compute_atr,
        compute_rsi=indicators.compute_rsi,
        compute_macd=indicators.compute_macd,
        has_bullish_rsi_divergence=divergence.has_bullish_rsi_divergence,
        cross_lookback=cross_lb,
        require_divergence=divergence_required,
        require_rsi_zone=rsi_zone_required,
        rsi_zone_long_max=rsi_long_cap,
    )
