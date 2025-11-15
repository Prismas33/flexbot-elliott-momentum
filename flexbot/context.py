import os
import logging
from typing import Dict, List, Sequence

import ccxt
from dotenv import load_dotenv

from state_store import read_user_config

load_dotenv()

API_KEY = os.getenv("API_KEY", "SUA_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SEU_API_SECRET")
exchange_id = "bybit"
paper = True
risk_percent = 0.01
risk_floor_fraction = 0.25
capital_base = 100.0
risk_mode = "standard"
leverage = 5.0

timeframes = ["5m", "15m", "1h", "4h", "1d"]
pairs = [
    "BTC/USDC:USDC",
    "ETH/USDC:USDC",
    "SOL/USDC:USDC",
    "XRP/USDC:USDC",
    "ADA/USDC:USDC",
    "LINK/USDC:USDC",
    "AVAX/USDC:USDC",
    "MATIC/USDC:USDC",
    "APT/USDC:USDC",
    "DOGE/USDC:USDC",
]
active_pairs = pairs[:3]
multi_asset_enabled = True
entry_timeframes = list(timeframes)

simulation_base_capital = 100.0
simulation_risk_per_trade = 0.10

tf_minutes_map = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

tf_bias_parent = {
    "5m": "15m",
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
    "1d": None,
}

strategy_mode = "momentum"
trade_bias = "long"
momentum_stop_atr = 1.5
momentum_tp_atr = 4.5
momentum_min_tf_agree = 1
min_rr_required = 2.0
ema_fast_period = 5
ema_slow_period = 21
ema_macd_fast = 26
ema_macd_slow = 55
ema_macd_signal = 9
ema_macd_min_tf_agree = 1
ema_macd_stop_atr = 1.8
ema_macd_tp_atr = 5.4
ema_macd_cross_lookback = 8
ema_macd_require_divergence = True
divergence_min_drop_pct = 0.015
ema_require_rsi_zone = False
ema_rsi_zone_long_max = 29.0
ema_rsi_zone_short_min = 70.0

trend_fast_span = 20
trend_slow_span = 50
trend_flat_tolerance = 0.0005

lookback = 400
volume_ma_period = 20
rsi_period = 14
local_extrema_order = 6
loop_interval_seconds = 30
account_balance_fallback = 10000
use_trailing = False
trailing_atr_multiplier = 1.5
atr_period = 14
take_partial_rr = 1.5
break_even_rr = 1.0
backtest_min_hold_candles = 3
max_backtest_candles_per_tf = 20000

momentum_bias_override_score = 3
ema_macd_bias_override_score = 3
loss_streak_limit = 2
loss_streak_risk_factor = 0.5

user_config = read_user_config()
environment_mode = user_config.get("environment", "paper")
if environment_mode not in ("paper", "live"):
    environment_mode = "paper"
paper = environment_mode != "live"

config_trade_bias = user_config.get("trade_bias")
if isinstance(config_trade_bias, str) and config_trade_bias in ("long", "short", "both"):
    trade_bias = config_trade_bias

config_cross_lb = user_config.get("ema_cross_lookback")
if isinstance(config_cross_lb, int) and config_cross_lb > 0:
    ema_macd_cross_lookback = config_cross_lb

config_require_div = user_config.get("ema_require_divergence")
if isinstance(config_require_div, bool):
    ema_macd_require_divergence = config_require_div

config_divergence_drop = user_config.get("divergence_min_drop_pct")
try:
    if config_divergence_drop is not None:
        divergence_min_drop_pct = max(0.0, float(config_divergence_drop))
except (TypeError, ValueError):
    pass

config_require_rsi_zone = user_config.get("ema_require_rsi_zone")
if isinstance(config_require_rsi_zone, bool):
    ema_require_rsi_zone = config_require_rsi_zone

config_rsi_long = user_config.get("ema_rsi_zone_long_max")
try:
    if config_rsi_long is not None:
        ema_rsi_zone_long_max = float(config_rsi_long)
except (TypeError, ValueError):
    pass

config_rsi_short = user_config.get("ema_rsi_zone_short_min")
try:
    if config_rsi_short is not None:
        ema_rsi_zone_short_min = float(config_rsi_short)
except (TypeError, ValueError):
    pass

config_risk_percent = user_config.get("risk_percent")
try:
    if config_risk_percent is not None:
        risk_percent = max(0.0001, min(float(config_risk_percent), 1.0))
except (TypeError, ValueError):
    pass

config_capital_base = user_config.get("capital_base")
try:
    if config_capital_base is not None:
        capital_base = max(0.0, float(config_capital_base))
except (TypeError, ValueError):
    pass

config_risk_mode = user_config.get("risk_mode")
if isinstance(config_risk_mode, str) and config_risk_mode in ("standard", "hunter"):
    risk_mode = config_risk_mode

config_leverage = user_config.get("leverage")
try:
    if config_leverage is not None:
        leverage = max(1.0, float(config_leverage))
except (TypeError, ValueError):
    pass

config_active_pairs = user_config.get("active_pairs")
if isinstance(config_active_pairs, Sequence) and not isinstance(config_active_pairs, (str, bytes)):
    filtered = [p for p in config_active_pairs if p in pairs]
    if filtered:
        active_pairs = list(dict.fromkeys(filtered))

config_multi_asset = user_config.get("multi_asset_enabled")
if isinstance(config_multi_asset, bool):
    multi_asset_enabled = config_multi_asset

exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})
try:
    exchange.load_markets()
except Exception as e:
    logging.warning("load_markets falhou: %s", e)

if paper:
    logging.info("MODO PAPER ATIVO — ordens reais NÃO serão colocadas.")

if API_KEY in (None, "", "SUA_API_KEY") or API_SECRET in (None, "", "SEU_API_SECRET"):
    logging.warning("API_KEY/API_SECRET não definidos. Configure variáveis de ambiente ou .env antes de usar --no-paper.")


def set_entry_timeframes(allowed: List[str] | None) -> None:
    global entry_timeframes
    if not allowed:
        entry_timeframes = list(timeframes)
        return
    unique: List[str] = []
    for tf in allowed:
        if tf in timeframes and tf not in unique:
            unique.append(tf)
    entry_timeframes = unique if unique else list(timeframes)


def update_environment_mode(mode: str) -> None:
    global environment_mode, paper
    if mode not in ("paper", "live"):
        return
    environment_mode = mode
    paper = environment_mode != "live"


def override_from_cli(settings: Dict[str, object]) -> None:
    global trade_bias, ema_macd_cross_lookback, ema_macd_require_divergence, strategy_mode, risk_percent, capital_base, risk_mode, leverage, active_pairs, multi_asset_enabled, divergence_min_drop_pct, ema_require_rsi_zone, ema_rsi_zone_long_max, ema_rsi_zone_short_min
    if "trade_bias" in settings and settings["trade_bias"] in ("long", "short", "both"):
        trade_bias = settings["trade_bias"]  # type: ignore[assignment]
    if "ema_cross_lookback" in settings and isinstance(settings["ema_cross_lookback"], int):
        ema_macd_cross_lookback = settings["ema_cross_lookback"]  # type: ignore[assignment]
    if "ema_require_divergence" in settings and isinstance(settings["ema_require_divergence"], bool):
        ema_macd_require_divergence = settings["ema_require_divergence"]  # type: ignore[assignment]
    if "ema_require_rsi_zone" in settings and isinstance(settings["ema_require_rsi_zone"], bool):
        ema_require_rsi_zone = settings["ema_require_rsi_zone"]  # type: ignore[assignment]
    if "divergence_min_drop_pct" in settings:
        try:
            val = float(settings["divergence_min_drop_pct"])
        except (TypeError, ValueError):
            pass
        else:
            if val >= 0:
                divergence_min_drop_pct = val  # type: ignore[assignment]
    if "ema_rsi_zone_long_max" in settings:
        try:
            val = float(settings["ema_rsi_zone_long_max"])
        except (TypeError, ValueError):
            pass
        else:
            ema_rsi_zone_long_max = val  # type: ignore[assignment]
    if "ema_rsi_zone_short_min" in settings:
        try:
            val = float(settings["ema_rsi_zone_short_min"])
        except (TypeError, ValueError):
            pass
        else:
            ema_rsi_zone_short_min = val  # type: ignore[assignment]
    if "strategy_mode" in settings and settings["strategy_mode"] in ("momentum", "ema_macd"):
        strategy_mode = settings["strategy_mode"]  # type: ignore[assignment]
    if "risk_percent" in settings:
        try:
            val = float(settings["risk_percent"])
        except (TypeError, ValueError):
            pass
        else:
            if val > 0:
                risk_percent = max(0.0001, min(val, 1.0))  # type: ignore[assignment]
    if "capital_base" in settings:
        try:
            val = float(settings["capital_base"])
        except (TypeError, ValueError):
            pass
        else:
            if val >= 0:
                capital_base = max(0.0, val)  # type: ignore[assignment]
    if "risk_mode" in settings and settings["risk_mode"] in ("standard", "hunter"):
        risk_mode = settings["risk_mode"]  # type: ignore[assignment]
    if "leverage" in settings:
        try:
            val = float(settings["leverage"])
        except (TypeError, ValueError):
            pass
        else:
            if val >= 1:
                leverage = max(1.0, val)  # type: ignore[assignment]
    if "active_pairs" in settings:
        seq = settings["active_pairs"]
        if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
            selected = [p for p in seq if p in pairs]
            if selected:
                active_pairs = list(dict.fromkeys(selected))  # type: ignore[assignment]
    if "multi_asset_enabled" in settings and isinstance(settings["multi_asset_enabled"], bool):
        multi_asset_enabled = settings["multi_asset_enabled"]  # type: ignore[assignment]


def set_active_pairs(selected: Sequence[str] | None) -> None:
    global active_pairs
    if not selected:
        if not multi_asset_enabled and pairs:
            active_pairs = [pairs[0]]
        else:
            active_pairs = pairs[:3]
        return
    filtered = [p for p in selected if p in pairs]
    active_pairs = list(dict.fromkeys(filtered)) if filtered else pairs[:3]


def set_multi_asset_enabled(flag: bool) -> None:
    global multi_asset_enabled
    multi_asset_enabled = bool(flag)
