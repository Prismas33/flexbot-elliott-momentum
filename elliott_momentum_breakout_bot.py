"""
elliott_momentum_breakout_bot.py

Sistema integrado: Elliott 2->3 + Momentum + Breakout
- Multi-par (ETH/BTC/SOL) + Multi-timeframe (5m,15m,1h)
- Entrada em dois estágios: partial entry no breakout + add-on no retest (opção C)
- SL/TP automáticos, trailing stop, position sizing por risco
- Backtester simples para simular sinais e métricas

INSTRUÇÕES RÁPIDAS
1) pip install ccxt pandas numpy ta
2) Preenche API_KEY / API_SECRET
3) Mantém paper=True para testar
4) Para correr backtest: executar o ficheiro; por defeito faz backtest ETH 15m
5) Para correr em live: set paper=False e chama main_loop()

Nota: testa em paper e faz backtests antes de usar capital real.
"""

import os
import ccxt
import argparse
import time
import math
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from dotenv import load_dotenv

from state_store import write_runtime_state, write_backtest_summary

# Dependency check — helpful message if a module is missing
try:
    import pandas as _pd  # quick check for common libs
    import numpy as _np
    from ta.momentum import RSIIndicator as _rsi
except ModuleNotFoundError as e:
    logging.error("Missing Python package: %s. Install dependencies with:\n  py -3 -m pip install -r requirements.txt\nOr: py -3 -m pip install ccxt pandas numpy ta", e.name)
    raise

# Load environment variables (.env)
load_dotenv()

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY", "SUA_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SEU_API_SECRET")
exchange_id = "bybit"
paper = True                      # True = NÃO envia ordens reais
default_risk_per_trade = 0.01     # 1% do capital por trade
leverage = 5                      # alavancagem desejada (se aplicável)
# timeframes and markets
timeframes = ["5m", "15m", "1h", "4h", "1d"]
pairs = [
    "ETH/USDC:USDC",
    "BTC/USDC:USDC",
    "SOL/USDC:USDC"
]
# simulation defaults
simulation_base_capital = 100.0
simulation_risk_per_trade = 0.10
# timeframe helpers (minutes)
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
# strategy toggles
strategy_mode = "elliott"  # options: elliott, momentum, ema_macd
momentum_stop_atr = 1.5
momentum_tp_atr = 4.5  # garante 3R natural (tp/stop = 3)
momentum_min_tf_agree = 2
min_rr_required = 3.0
ema_fast_period = 5
ema_slow_period = 21
ema_macd_fast = 26
ema_macd_slow = 55
ema_macd_signal = 9
ema_macd_min_tf_agree = 1
ema_macd_stop_atr = 1.8
ema_macd_tp_atr = 5.4  # garante 3R natural (tp/stop = 3)
# trend filter
trend_fast_span = 20
trend_slow_span = 50
trend_flat_tolerance = 0.0005
# analysis params
lookback = 400                    # candles a buscar
volume_ma_period = 20
rsi_period = 14
local_extrema_order = 5           # sensibilidade para detectar picos/vales
fib_min = 0.38
fib_max = 0.78
entry_fib_zone = (0.5, 0.618)
tp_extension = 1.618
min_tf_agree = 2                  # quantas TFs têm de concordar
loop_interval_seconds = 30
account_balance_fallback = 10000
# Entry staging
partial_entry_fraction = 0.35     # 35% on breakout
add_on_fraction = 0.65            # 65% on retest
max_total_fraction = 1.0
# trailing stop config
use_trailing = False
trailing_atr_multiplier = 1.5
atr_period = 14
# parciais e break-even
take_partial_rr = 1.5           # realizar parcial a ~1.5R
break_even_rr = 1.0             # mover SL para BE a 1R
# backtest params
backtest_min_hold_candles = 3
max_backtest_candles_per_tf = 5000

# ----------------------------------------

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# instantiate exchange
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

# ---------- Helpers ----------

def fetch_ohlcv(symbol, timeframe, limit=200):
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logging.error("Erro fetch_ohlcv %s %s: %s", symbol, timeframe, e)
        return None
    df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def fetch_ohlcv_paginated(symbol, timeframe, bars_needed):
    try:
        tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    except Exception:
        tf_ms = 60_000
    since = max(0, exchange.milliseconds() - (bars_needed + 5) * tf_ms)
    all_rows = []
    attempts = 0
    while len(all_rows) < bars_needed and attempts < 50:
        limit = min(1000, bars_needed - len(all_rows))
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            logging.error("Erro fetch_ohlcv_paginated %s %s: %s", symbol, timeframe, e)
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + tf_ms
        attempts += 1
        if len(batch) < limit:
            break
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
    df.drop_duplicates(subset='ts', inplace=True)
    df.sort_values('ts', inplace=True)
    df = df.tail(bars_needed)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def fetch_backtest_series(symbol, timeframe, candles_needed):
    candles_needed = min(candles_needed, max_backtest_candles_per_tf)
    df = fetch_ohlcv_paginated(symbol, timeframe, candles_needed)
    return df, (len(df) if df is not None else 0)

def timeframe_to_minutes(tf):
    if tf in tf_minutes_map:
        return tf_minutes_map[tf]
    try:
        seconds = exchange.parse_timeframe(tf)
        return max(1, int(seconds / 60))
    except Exception:
        raise ValueError(f"Timeframe não suportado: {tf}")

def compute_rsi(df, period=14):
    return RSIIndicator(df['close'], period).rsi()

def compute_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def is_local_max(series, idx, order):
    start = max(0, idx - order)
    end = min(len(series)-1, idx + order)
    center = series.iloc[idx]
    window = series.iloc[start:end+1]
    return center >= window.max()

def is_local_min(series, idx, order):
    start = max(0, idx - order)
    end = min(len(series)-1, idx + order)
    center = series.iloc[idx]
    window = series.iloc[start:end+1]
    return center <= window.min()

def find_turning_points(df_close, order=5):
    pts = []
    n = len(df_close)
    for i in range(order, n-order):
        if is_local_max(df_close, i, order):
            pts.append((i, 'max'))
        elif is_local_min(df_close, i, order):
            pts.append((i, 'min'))
    return pts

# ---------- Wave detection (Elliott-ish) ----------
def detect_wave2_to_3(df, trend_regime="flat"):
    if df is None or len(df) < 80:
        return None
    df = df.copy()
    df['rsi'] = compute_rsi(df, rsi_period)
    df['vol_ma'] = df['volume'].rolling(volume_ma_period, min_periods=1).mean()
    close = df['close']
    pts = find_turning_points(close, order=local_extrema_order)
    if len(pts) < 4:
        return None
    last4 = pts[-4:]
    idx0 = last4[0][0]
    idx1 = last4[1][0]
    idx2 = last4[2][0]
    # bullish pattern
    p0 = close.iloc[idx0]
    p1 = close.iloc[idx1]
    p2 = close.iloc[idx2]
    if not (p1 > p0 and p2 < p1):
        return None
    wave1_len = p1 - p0
    corr_len = p1 - p2
    if wave1_len <= 0:
        return None
    corr_ratio = corr_len / wave1_len
    # ajustar zona de correcao consoante regime de tendencia
    if trend_regime == "trend":
        fib_min_eff, fib_max_eff = max(fib_min, 0.45), min(fib_max, 0.68)
    else:
        fib_min_eff, fib_max_eff = min(fib_min, 0.33), max(fib_max, 0.80)
    if not (fib_min_eff <= corr_ratio <= fib_max_eff):
        return None
    rsi_p0 = df['rsi'].iloc[idx0] if idx0 < len(df) else df['rsi'].iloc[-6]
    rsi_p2 = df['rsi'].iloc[idx2]
    divergence = (p2 < p0) and (rsi_p2 > rsi_p0)
    last_close = close.iloc[-1]
    breakout = last_close > p1
    vol_spike = df['volume'].iloc[-1] > df['vol_ma'].iloc[-1] * 1.2
    fib50 = p1 - wave1_len * 0.5
    fib618 = p1 - wave1_len * 0.618
    in_fib_entry = (p2 <= fib50 and p2 >= fib618)
    # score ponderado para Elliott
    score = 0
    if divergence:
        score += 2
    if breakout:
        score += 2
    if vol_spike:
        score += 1
    if in_fib_entry:
        score += 1

    signal = {
        "wave1_start_price": p0,
        "wave1_top_price": p1,
        "wave2_bottom_price": p2,
        "wave1_len": wave1_len,
        "corr_ratio": corr_ratio,
        "divergence": divergence,
        "breakout": breakout,
        "vol_spike": vol_spike,
        "in_fib_entry": in_fib_entry,
        "last_close": last_close,
        "timestamp": df.index[-1],
        "rsi": rsi_p2,
        "probability_score": score
    }
    return signal

# ---------- Momentum & Breakout filters ----------
def momentum_confirm(df):
    if df is None or len(df) < 50:
        return False, {}
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['atr'] = compute_atr(df, atr_period)
    df['vol_ma'] = df['volume'].rolling(volume_ma_period, min_periods=1).mean()
    df['rsi'] = compute_rsi(df, rsi_period)

    ema_cross = df['ema20'].iloc[-1] > df['ema50'].iloc[-1]
    atr_expansion = df['atr'].iloc[-1] > df['atr'].iloc[-5:].mean() * 1.1
    vol_spike = df['volume'].iloc[-1] > df['vol_ma'].iloc[-1] * 1.2

    # evitar entradas com RSI extremamente sobrecomprado
    rsi_val = float(df['rsi'].iloc[-1]) if not np.isnan(df['rsi'].iloc[-1]) else None
    rsi_ok = True
    if rsi_val is not None and rsi_val > 80:
        rsi_ok = False

    # score ponderado
    score = 0
    if ema_cross:
        score += 2
    if atr_expansion:
        score += 1
    if vol_spike:
        score += 1

    confirmed = (score >= 2) and rsi_ok
    return confirmed, {
        "ema_cross": ema_cross,
        "atr_expansion": atr_expansion,
        "vol_spike": vol_spike,
        "score": score,
        "atr": df['atr'].iloc[-1],
        "last_close": df['close'].iloc[-1],
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
    }

def ema_macd_confirm(df):
    if df is None or len(df) < max(ema_slow_period, ema_macd_slow) + 5:
        return False, {}
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=ema_fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow_period, adjust=False).mean()
    macd_line, signal_line, hist = compute_macd(df['close'], ema_macd_fast, ema_macd_slow, ema_macd_signal)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['atr'] = compute_atr(df, atr_period)
    df['rsi'] = compute_rsi(df, rsi_period)

    ema_trend = df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]
    macd_above_signal = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
    histogram_positive = df['macd_hist'].iloc[-1] > 0
    atr_available = df['atr'].iloc[-1] > 0

    # evitar entradas em pico de histograma muito esticado
    recent_hist = df['macd_hist'].iloc[-5:]
    hist_now = recent_hist.iloc[-1]
    hist_peak = recent_hist.max()
    hist_ok = abs(hist_now) <= abs(hist_peak) * 1.05  # evitar quando ja disparou demais

    # slope das EMAs
    ema_fast_recent = df['ema_fast'].iloc[-5:]
    ema_slow_recent = df['ema_slow'].iloc[-5:]
    slope_fast = ema_fast_recent.iloc[-1] - ema_fast_recent.iloc[0]
    slope_slow = ema_slow_recent.iloc[-1] - ema_slow_recent.iloc[0]
    slope_ok = slope_fast > 0 and slope_slow >= 0

    # RSI em zona saudavel
    rsi_val = float(df['rsi'].iloc[-1]) if not np.isnan(df['rsi'].iloc[-1]) else None
    rsi_ok = True
    if rsi_val is not None and (rsi_val < 40 or rsi_val > 70):
        rsi_ok = False

    score = 0
    if ema_trend:
        score += 1
    if macd_above_signal:
        score += 1
    if histogram_positive and hist_ok:
        score += 1

    confirmed = (score >= 2) and atr_available and slope_ok and rsi_ok
    return confirmed, {
        "ema_fast": float(df['ema_fast'].iloc[-1]),
        "ema_slow": float(df['ema_slow'].iloc[-1]),
        "macd": float(df['macd'].iloc[-1]),
        "macd_signal": float(df['macd_signal'].iloc[-1]),
        "macd_hist": float(df['macd_hist'].iloc[-1]),
        "score": score,
        "atr": float(df['atr'].iloc[-1]),
        "last_close": float(df['close'].iloc[-1]),
        "rsi": rsi_val,
        "rsi_ok": rsi_ok,
        "hist_ok": hist_ok,
        "slope_fast": float(slope_fast),
        "slope_slow": float(slope_slow),
    }

# ---------- Aggregation multi-TF ----------
def aggregate_signals(signals_by_tf):
    agrees = 0
    details = {}
    available = 0
    for tf, s in signals_by_tf.items():
        details[tf] = s
        if s is None:
            continue
        available += 1
        strong = (s['divergence'] and s['breakout'] and s['vol_spike'])
        # dar prioridade a TFs medios (15m,1h,4h) com score alto
        tf_weight = 1.0
        if tf in ("15m", "1h", "4h"):
            tf_weight = 1.5
        if strong or s['probability_score'] >= 4:
            agrees += tf_weight
    if available == 0:
        return False, details
    required = min(min_tf_agree, available)
    return agrees >= max(1, required), details

# ---------- Position sizing ----------
def fetch_account_balance():
    try:
        bal = exchange.fetch_balance(params={"type":"future"})
        if isinstance(bal, dict):
            totals = bal.get('total', {})
            for cur in ['USDC','USD','USDT']:
                if cur in totals and totals[cur] is not None:
                    return float(totals[cur])
    except Exception as e:
        logging.warning("fetch_balance falhou: %s", e)
    return account_balance_fallback

def compute_size(account_balance, entry_price, stop_price, risk_per_trade=default_risk_per_trade):
    risk_amount = account_balance * risk_per_trade
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        return 0
    qty = risk_amount / risk_per_unit
    return max(0, qty)

def compute_rr(entry_price, stop_price, tp_price):
    risk = entry_price - stop_price
    reward = tp_price - entry_price
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk

def has_bullish_rsi_divergence(df, order=None):
    if df is None or len(df) < 40:
        return False
    order = order or local_extrema_order
    close = df['close']
    rsi = compute_rsi(df, rsi_period)
    if rsi.isna().all():
        return False
    pts = find_turning_points(close, order=order)
    lows = [idx for idx, typ in pts if typ == 'min']
    if len(lows) < 2:
        return False
    idx1, idx2 = lows[-2], lows[-1]
    price1 = close.iloc[idx1]
    price2 = close.iloc[idx2]
    rsi1 = rsi.iloc[idx1]
    rsi2 = rsi.iloc[idx2]
    if np.isnan(price1) or np.isnan(price2) or np.isnan(rsi1) or np.isnan(rsi2):
        return False
    return price2 < price1 and rsi2 > rsi1

def determine_trend_direction(df, fast=trend_fast_span, slow=trend_slow_span):
    if df is None or len(df) < slow + 5:
        return "unknown"
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
    if diff > trend_flat_tolerance:
        return "up"
    if diff < -trend_flat_tolerance:
        return "down"
    return "flat"

def compute_symbol_bias(symbol):
    bias = {}
    parent_cache = {}
    for tf in timeframes:
        parent = tf_bias_parent.get(tf)
        if parent is None:
            bias[tf] = "both"
            continue
        if parent not in parent_cache:
            parent_cache[parent] = fetch_ohlcv(symbol, parent, limit=lookback)
        bias[tf] = determine_trend_direction(parent_cache[parent])
    return bias

def bias_allows_long(tf, bias_map):
    direction = bias_map.get(tf, "both")
    if direction == "down":
        return False
    return True

def bias_from_slices(tf, ts, slice_store):
    parent = tf_bias_parent.get(tf)
    if parent is None:
        return "both"
    parent_df = slice_store.get(parent)
    if parent_df is None or parent_df.empty:
        return "unknown"
    df = parent_df[parent_df.index <= ts].tail(lookback)
    if df is None or df.empty:
        return "unknown"
    return determine_trend_direction(df)

# ---------- Order helpers (paper fallback) ----------
def place_market_order(symbol, side, qty):
    logging.info("ORDER MARKET %s %s %s", side, qty, symbol)
    if paper:
        return {"id": f"paper-{int(time.time())}", "symbol": symbol, "side": side, "qty": qty, "status": "paper", "price": None}
    try:
        order = exchange.create_market_order(symbol, side.lower(), qty)
        return order
    except Exception as e:
        logging.error("Erro colocar market order: %s", e)
        return None

def place_conditional_orders(symbol, side, qty, stop_price, tp_price):
    # Try to create conditional SL/TP (may vary by CCXT version). If fails, return False to rely on monitor fallback.
    if paper:
        logging.info("PAPER: criar ordens condicionais (simuladas) for %s", symbol)
        return True
    try:
        params_sl = {"stopPrice": stop_price, "reduceOnly": True, "closeOnTrigger": True}
        params_tp = {"stopPrice": tp_price, "reduceOnly": True, "closeOnTrigger": True}
        exchange.create_order(symbol, 'stop', 'sell' if side=='buy' else 'buy', qty, None, params_sl)
        exchange.create_order(symbol, 'stop', 'sell' if side=='buy' else 'buy', qty, None, params_tp)
        return True
    except Exception as e:
        logging.warning("Falha ordens condicionais: %s", e)
        return False

# ---------- Execution logic: partial entry + add-on on retest ----------
# tracking paper/trades for fallback
open_positions = []  # list of dicts

def serialize_open_positions():
    serialized = []
    for pos in open_positions:
        serialized.append({
            "symbol": pos.get("symbol"),
            "qty_current": pos.get("qty_current"),
            "qty_total": pos.get("qty_total"),
            "entry_price": pos.get("entry_price"),
            "stop": pos.get("stop"),
            "tp": pos.get("tp"),
            "opened_at": pos.get("opened_at").isoformat() if pos.get("opened_at") else None,
            "addon_pending": pos.get("addon_pending"),
            "strategy": pos.get("strategy", "elliott"),
            "partial_taken": pos.get("partial_taken", False),
            "breakeven_moved": pos.get("breakeven_moved", False),
        })
    return serialized

def enter_partial_and_plan_addon(symbol, ref_signal, account_balance):
    entry_price = ref_signal['last_close']
    # SL estrutura + ATR: minimo entre fundo da wave2 e k*ATR abaixo da entrada
    df_ref = fetch_ohlcv(symbol, '15m', limit=lookback)
    atr_val = compute_atr(df_ref, atr_period).iloc[-1] if df_ref is not None and len(df_ref) > atr_period + 1 else None
    structural_sl = ref_signal['wave2_bottom_price'] * 0.995
    atr_sl = entry_price - (atr_val * 2.0) if atr_val is not None else structural_sl
    stop_price = min(structural_sl, atr_sl)
    qty_full = compute_size(account_balance, entry_price, stop_price, default_risk_per_trade)
    qty_partial = qty_full * partial_entry_fraction
    qty_addon = qty_full * add_on_fraction
    qty_partial = max(0, qty_partial)
    qty_addon = max(0, qty_addon)
    tp_price = ref_signal['wave1_top_price'] + ref_signal['wave1_len'] * tp_extension
    logging.info("PLANNING TRADE %s entry=%.6f partial_qty=%.6f addon_qty=%.6f stop=%.6f tp=%.6f", symbol, entry_price, qty_partial, qty_addon, stop_price, tp_price)
    order = place_market_order(symbol, 'buy', qty_partial)
    if order is None:
        return None
    # track position for fallback monitoring and addon logic
    pos = {
        "symbol": symbol,
        "side": "buy",
        "entry_price": entry_price,
        "qty_total": qty_partial + qty_addon,
        "qty_partial": qty_partial,
        "qty_addon": qty_addon,
        "qty_current": qty_partial,
        "stop": stop_price,
        "tp": tp_price,
        "opened_at": datetime.utcnow(),
        "addon_pending": True,
        "ref_signal": ref_signal,
        "strategy": "elliott",
        "partial_taken": False,
        "breakeven_moved": False,
    }
    open_positions.append(pos)
    # try set conditional SL/TP for the partial position (best-effort)
    place_conditional_orders(symbol, 'buy', qty_partial, stop_price, tp_price)
    return pos

def enter_momentum_trade(symbol, entry_price, stop_price, tp_price, account_balance, strategy_label="momentum"):
    qty = compute_size(account_balance, entry_price, stop_price, default_risk_per_trade)
    qty = max(0, qty)
    if qty == 0:
        logging.info("Momentum trade abortado — qty=0")
        return None
    order = place_market_order(symbol, 'buy', qty)
    if order is None:
        return None
    pos = {
        "symbol": symbol,
        "side": "buy",
        "entry_price": entry_price,
        "qty_total": qty,
        "qty_partial": qty,
        "qty_addon": 0,
        "qty_current": qty,
        "stop": stop_price,
        "tp": tp_price,
        "opened_at": datetime.utcnow(),
        "addon_pending": False,
        "strategy": strategy_label,
        "partial_taken": False,
        "breakeven_moved": False,
    }
    open_positions.append(pos)
    place_conditional_orders(symbol, 'buy', qty, stop_price, tp_price)
    return pos

def check_retest_and_addon(pos):
    # retest logic: after partial entry, wait for price to pull back to resistance-turned-support (approx wave1_top_price)
    try:
        df_5m = fetch_ohlcv(pos['symbol'], '5m', limit=200)
        if df_5m is None or len(df_5m) < 10:
            return False
        last_close = df_5m['close'].iloc[-1]
        ref = pos['ref_signal']
        # consider retest if price drops back near wave1_top_price (within ATR range)
        atr = compute_atr(df_5m, atr_period).iloc[-1]
        target_zone_low = ref['wave1_top_price'] - atr * 1.2
        target_zone_high = ref['wave1_top_price'] + atr * 0.6
        # if price re-enters zone and bounces (current close > previous low) -> add
        if last_close >= target_zone_low and last_close <= target_zone_high:
            # check bounce: last candle close higher than previous candle low
            if df_5m['close'].iloc[-1] > df_5m['low'].iloc[-2]:
                # place addon market order
                qty = pos['qty_addon']
                order = place_market_order(pos['symbol'], 'buy', qty)
                if order:
                    pos['qty_current'] += qty
                    pos['addon_pending'] = False
                    # attempt to set SL/TP for full position
                    place_conditional_orders(pos['symbol'], 'buy', pos['qty_current'], pos['stop'], pos['tp'])
                    logging.info("Addon placed for %s qty=%.6f", pos['symbol'], qty)
                    return True
    except Exception as e:
        logging.exception("Erro check_retest_and_addon: %s", e)
    return False

def monitor_and_close_positions(open_positions):
    closed = []
    for pos in list(open_positions):
        df = fetch_ohlcv(pos['symbol'], '1m', limit=10)
        if df is None or df.empty:
            continue
        high = df['high'].max()
        low = df['low'].min()

        entry = pos['entry_price']
        stop = pos['stop']
        tp = pos['tp']

        # calcular R atual
        if stop is not None and tp is not None and entry is not None:
            risk_per_unit = entry - stop if pos['side'] == 'buy' else stop - entry
            if risk_per_unit > 0:
                # preco max/min desde que a posicao foi aberta (aqui, janela reduzida)
                if pos['side'] == 'buy':
                    max_price = high
                    rr_reached = (max_price - entry) / risk_per_unit
                else:
                    min_price = low
                    rr_reached = (entry - min_price) / risk_per_unit
            else:
                rr_reached = 0
        else:
            rr_reached = 0

        # mover SL para break-even a 1R
        if (not pos.get('breakeven_moved', False)) and rr_reached >= break_even_rr and entry is not None:
            pos['stop'] = entry
            pos['breakeven_moved'] = True
            logging.info("Move SL para break-even em %s (%.2fR)", pos['symbol'], rr_reached)

        # parcial a 1.5R: fechar metade da posicao (se ainda nao feito)
        if (not pos.get('partial_taken', False)) and rr_reached >= take_partial_rr:
            qty_close = pos['qty_current'] * 0.5
            if qty_close > 0:
                side = 'sell' if pos['side'] == 'buy' else 'buy'
                if paper:
                    logging.info("PAPER: parcial em %s, fechar %.6f", pos['symbol'], qty_close)
                    pos['qty_current'] -= qty_close
                    pos['partial_taken'] = True
                else:
                    try:
                        close_order = exchange.create_market_order(pos['symbol'], side, qty_close)
                        logging.info("Parcial fechada %s: %s", pos['symbol'], close_order)
                        pos['qty_current'] -= qty_close
                        pos['partial_taken'] = True
                    except Exception as e:
                        logging.error("Erro fechar parcial em %s: %s", pos['symbol'], e)

        # TP/SL final para o restante
        hit_tp = (tp is not None) and (high >= tp)
        hit_sl = (stop is not None) and (low <= stop)
        if hit_tp or hit_sl:
            side = 'sell' if pos['side'] == 'buy' else 'buy'
            qty = pos['qty_current']
            if qty <= 0:
                closed.append(pos)
                open_positions.remove(pos)
                continue
            if paper:
                logging.info("PAPER: fechar pos %s por %s (tp/sl atingido).", pos['symbol'], 'TP' if hit_tp else 'SL')
                closed.append(pos)
                open_positions.remove(pos)
            else:
                try:
                    close_order = exchange.create_market_order(pos['symbol'], side, qty)
                    logging.info("Fechada pos %s: %s", pos['symbol'], close_order)
                    closed.append(pos)
                    open_positions.remove(pos)
                except Exception as e:
                    logging.error("Erro fechar pos %s: %s", pos['symbol'], e)
    return closed

# ---------- Live analyze & execute ----------
def analyze_elliott_and_maybe_trade(symbol):
    summary = {
        "symbol": symbol,
        "strategy": "elliott",
        "status": "ok",
        "should_enter": False,
        "reference_tf": None,
        "score": None,
    }
    bias_map = compute_symbol_bias(symbol)
    summary["bias"] = bias_map
    signals = {}
    # regime de mercado com base no TF pai do timeframe principal (15m por defeito)
    main_parent_tf = tf_bias_parent.get("15m", "1h")
    parent_df = fetch_ohlcv(symbol, main_parent_tf, limit=lookback) if main_parent_tf else None
    trend_regime = determine_trend_direction(parent_df) if parent_df is not None else "flat"

    for tf in timeframes:
        if not bias_allows_long(tf, bias_map):
            signals[tf] = None
            continue
        df = fetch_ohlcv(symbol, tf, limit=lookback)
        if df is None or len(df) < 80:
            signals[tf] = None
            continue
        # divergencia RSI obrigatoria apenas no TF principal de entrada (15m/1h)
        requires_div = tf in ("15m", "1h")
        if requires_div and not has_bullish_rsi_divergence(df):
            signals[tf] = None
            continue
        sig = detect_wave2_to_3(df, trend_regime=trend_regime if tf in ("15m", "1h") else "flat")
        signals[tf] = sig
    should_enter, details = aggregate_signals(signals)
    logging.info("Market %s signals summary: enter=%s details=%s", symbol, should_enter, {k: (v is not None) for k,v in details.items()})
    summary["should_enter"] = should_enter
    summary["signals"] = {tf: (det is not None) for tf, det in details.items()}
    if not should_enter:
        return None, summary
    # choose reference TF (highest score)
    scored = [(tf, details[tf]['probability_score']) for tf in details if details[tf] is not None]
    if not scored:
        return None, summary
    scored.sort(key=lambda x: x[1], reverse=True)
    reference_tf = scored[0][0]
    summary["reference_tf"] = reference_tf
    summary["score"] = scored[0][1]
    s = details[reference_tf]
    entry_price = s['last_close'] if s else None
    stop_price = s['wave2_bottom_price'] * 0.995 if s else None
    tp_price = s['wave1_top_price'] + s['wave1_len'] * tp_extension if s else None
    rr = compute_rr(entry_price, stop_price, tp_price) if entry_price and stop_price and tp_price else None
    summary["entry_price"] = entry_price
    summary["rr"] = rr
    if rr is None or rr < min_rr_required:
        summary["status"] = "min_rr_not_met"
        summary["should_enter"] = False
        return None, summary
    account_balance = fetch_account_balance()
    pos = enter_partial_and_plan_addon(symbol, s, account_balance)
    return pos, summary

def analyze_momentum_and_maybe_trade(symbol):
    summary = {
        "symbol": symbol,
        "status": "ok",
        "strategy": "momentum",
        "should_enter": False,
        "reference_tf": None,
        "score": None,
    }
    bias_map = compute_symbol_bias(symbol)
    summary["bias"] = bias_map
    confirmations = {}
    # definir TFs core para momentum
    core_tfs = ("15m", "1h")
    for tf in timeframes:
        if not bias_allows_long(tf, bias_map):
            confirmations[tf] = {"confirmed": False, "bias_blocked": True}
            continue
        df = fetch_ohlcv(symbol, tf, limit=lookback)
        if df is None or len(df) < 50:
            confirmations[tf] = {"confirmed": False}
            continue
        # divergencia como obrigatoria apenas no modo breakout/reversao (TFs core)
        if tf in core_tfs and not has_bullish_rsi_divergence(df):
            confirmations[tf] = {"confirmed": False, "divergence": False}
            continue

        confirmed, details = momentum_confirm(df)
        confirmations[tf] = {
            "confirmed": confirmed,
            "details": details,
            "price": details.get("last_close"),
            "atr": details.get("atr")
        }
    summary["signals"] = {tf: data["confirmed"] for tf, data in confirmations.items()}
    ready = [
        (tf, data)
        for tf, data in confirmations.items()
        if data.get("confirmed") and data.get("details")
    ]
    required = min(momentum_min_tf_agree, len(ready))
    if len(ready) < max(1, required):
        return None, summary
    # dar prioridade a TFs core com melhor score
    ready.sort(key=lambda item: (
        1 if item[0] in core_tfs else 0,
        item[1]["details"].get("score", 0)
    ), reverse=True)
    ref_tf, ref_data = ready[0]
    summary["reference_tf"] = ref_tf
    summary["score"] = ref_data["details"].get("score")
    entry_price = ref_data["price"]
    atr_val = ref_data.get("atr") or 0
    if atr_val == 0:
        summary["status"] = "no_atr"
        return None, summary
    stop_price = entry_price - atr_val * momentum_stop_atr
    tp_price = entry_price + atr_val * momentum_tp_atr
    rr = compute_rr(entry_price, stop_price, tp_price)
    summary["rr"] = rr
    if rr is None or rr < min_rr_required:
        summary["status"] = "min_rr_not_met"
        return None, summary
    account_balance = fetch_account_balance()
    pos = enter_momentum_trade(symbol, entry_price, stop_price, tp_price, account_balance)
    summary["should_enter"] = pos is not None
    summary["entry_price"] = entry_price
    return pos, summary

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
    for tf in timeframes:
        if not bias_allows_long(tf, bias_map):
            confirmations[tf] = {"confirmed": False, "bias_blocked": True}
            continue
        df = fetch_ohlcv(symbol, tf, limit=lookback)
        if df is None or len(df) < ema_slow_period + 10:
            confirmations[tf] = {"confirmed": False}
            continue
        if not has_bullish_rsi_divergence(df):
            confirmations[tf] = {"confirmed": False, "divergence": False}
            continue
        confirmed, details = ema_macd_confirm(df)
        confirmations[tf] = {
            "confirmed": confirmed,
            "details": details,
            "price": details.get("last_close"),
            "atr": details.get("atr")
        }
    summary["signals"] = {tf: data.get("confirmed", False) for tf, data in confirmations.items()}
    ready = [
        (tf, data)
        for tf, data in confirmations.items()
        if data.get("confirmed") and data.get("details")
    ]
    required = min(ema_macd_min_tf_agree, len(ready))
    if len(ready) < max(1, required):
        return None, summary
    ready.sort(key=lambda item: item[1]["details"].get("score", 0), reverse=True)
    ref_tf, ref_data = ready[0]
    summary["reference_tf"] = ref_tf
    summary["score"] = ref_data["details"].get("score")
    entry_price = ref_data["price"]
    atr_val = ref_data.get("atr") or ref_data["details"].get("atr")
    if not entry_price or not atr_val:
        summary["status"] = "no_price_or_atr"
        return None, summary
    stop_price = entry_price - atr_val * ema_macd_stop_atr
    tp_price = entry_price + atr_val * ema_macd_tp_atr
    rr = compute_rr(entry_price, stop_price, tp_price)
    summary["rr"] = rr
    if rr is None or rr < min_rr_required:
        summary["status"] = "min_rr_not_met"
        return None, summary
    account_balance = fetch_account_balance()
    pos = enter_momentum_trade(symbol, entry_price, stop_price, tp_price, account_balance, strategy_label="ema_macd")
    summary["should_enter"] = pos is not None
    summary["entry_price"] = entry_price
    return pos, summary

def analyze_and_maybe_trade(symbol):
    if strategy_mode == "momentum":
        return analyze_momentum_and_maybe_trade(symbol)
    if strategy_mode == "ema_macd":
        return analyze_ema_macd_and_maybe_trade(symbol)
    return analyze_elliott_and_maybe_trade(symbol)

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
            # mover SL para BE a 1R
            if (not be_moved) and rr_reached >= break_even_rr:
                current_sl = entry_ref_price
                be_moved = True
            # parcial a 1.5R (na simulacao assumimos metade fechada nesse ponto)
            if (not partial_done) and rr_reached >= take_partial_rr:
                partial_done = True
            # verificar hits com SL/TP atual
            hit_tp = (high >= tp_price)
            hit_sl = (low <= current_sl)
            if hit_tp and not hit_sl:
                # se parcial foi tomada, metade do lucro ja foi antes em take_partial_rr*R
                if partial_done:
                    price_partial = entry_ref_price + take_partial_rr * risk_per_unit
                    pnl_partial = (price_partial - entry_ref_price) * 0.5
                    pnl_rest = (tp_price - entry_ref_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = tp_price - entry_ref_price
                return {"exit_price": tp_price, "exit_idx": i, "outcome":"tp", "pnl": pnl_total}
            if hit_sl and not hit_tp:
                if partial_done:
                    price_partial = entry_ref_price + take_partial_rr * risk_per_unit
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
                    price_partial = entry_ref_price + take_partial_rr * risk_per_unit
                    pnl_partial = (price_partial - entry_ref_price) * 0.5
                    pnl_rest = (exit_price - entry_ref_price) * 0.5
                    pnl_total = pnl_partial + pnl_rest
                else:
                    pnl_total = exit_price - entry_ref_price
                return {"exit_price": exit_price, "exit_idx": i, "outcome": outcome, "pnl": pnl_total}

    # se nunca bateu SL/TP, sai no close final
    final_price = df['close'].iloc[-1]
    if partial_done:
        price_partial = entry_ref_price + take_partial_rr * risk_per_unit
        pnl_partial = (price_partial - entry_ref_price) * 0.5
        pnl_rest = (final_price - entry_ref_price) * 0.5
        pnl_total = pnl_partial + pnl_rest
    else:
        pnl_total = final_price - entry_ref_price
    return {"exit_price": final_price, "exit_idx": n-1, "outcome": "none", "pnl": pnl_total}

def backtest_pair(symbol, timeframe, lookback_days=90, strategy=None):
    strategy = strategy or strategy_mode
    logging.info("Backtest %s %s last %d days", symbol, timeframe, lookback_days)
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = int((24*60/ minutes) * lookback_days) + 200
    df_main, main_cov = fetch_backtest_series(symbol, timeframe, candles_needed)
    coverage_info = {timeframe: main_cov}
    if df_main is None or len(df_main) < 200:
        logging.error("Dados insuficientes para backtest.")
        return None, coverage_info

    sim_risk = simulation_risk_per_trade
    current_equity = simulation_base_capital

    slice_store = {}
    for tf in timeframes:
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
    for i in range(200, len(df_main)-1):
        ts = df_main.index[i]
        slices = {}
        for tf, dfrag in slice_store.items():
            if dfrag is None:
                slices[tf] = None
                continue
            d = dfrag[dfrag.index <= ts].tail(lookback)
            slices[tf] = d if len(d) >= 60 else None
        bias_snapshot = {tf: bias_from_slices(tf, ts, slice_store) for tf in timeframes}
        rr_value = None
        if strategy == "momentum":
            momentum_ready = []
            for tf in timeframes:
                if not bias_allows_long(tf, bias_snapshot):
                    continue
                slice_df = slices.get(tf)
                if slice_df is None or len(slice_df) < 50:
                    continue
                if not has_bullish_rsi_divergence(slice_df):
                    continue
                confirmed, details = momentum_confirm(slice_df)
                if confirmed and details.get("atr"):
                    momentum_ready.append((tf, details))
            required = max(1, min(momentum_min_tf_agree, len(momentum_ready)))
            if len(momentum_ready) < required:
                continue
            momentum_ready.sort(key=lambda item: item[1].get("score", 0), reverse=True)
            ref_tf, detail = momentum_ready[0]
            entry_price = detail.get("last_close")
            atr_val = detail.get("atr")
            if entry_price is None or atr_val in (None, 0):
                continue
            stop_price = entry_price - atr_val * momentum_stop_atr
            tp_price = entry_price + atr_val * momentum_tp_atr
            rr = compute_rr(entry_price, stop_price, tp_price)
            if rr is None or rr < min_rr_required:
                continue
            rr_value = rr
        elif strategy == "ema_macd":
            ema_ready = []
            for tf in timeframes:
                if not bias_allows_long(tf, bias_snapshot):
                    continue
                slice_df = slices.get(tf)
                if slice_df is None or len(slice_df) < ema_slow_period + 10:
                    continue
                if not has_bullish_rsi_divergence(slice_df):
                    continue
                confirmed, details = ema_macd_confirm(slice_df)
                if confirmed and details.get("atr"):
                    ema_ready.append((tf, details))
            required = max(1, min(ema_macd_min_tf_agree, len(ema_ready)))
            if len(ema_ready) < required:
                continue
            ema_ready.sort(key=lambda item: item[1].get("score", 0), reverse=True)
            ref_tf, detail = ema_ready[0]
            entry_price = detail.get("last_close")
            atr_val = detail.get("atr")
            if entry_price is None or atr_val in (None, 0):
                continue
            stop_price = entry_price - atr_val * ema_macd_stop_atr
            tp_price = entry_price + atr_val * ema_macd_tp_atr
            rr = compute_rr(entry_price, stop_price, tp_price)
            if rr is None or rr < min_rr_required:
                continue
            rr_value = rr
        else:
            sigs = {}
            for tf in timeframes:
                if not bias_allows_long(tf, bias_snapshot):
                    sigs[tf] = None
                    continue
                slice_df = slices.get(tf)
                if slice_df is None or len(slice_df) < 80:
                    sigs[tf] = None
                    continue
                if not has_bullish_rsi_divergence(slice_df):
                    sigs[tf] = None
                    continue
                sigs[tf] = detect_wave2_to_3(slice_df)
            should_enter, details = aggregate_signals(sigs)
            if not should_enter:
                continue
            scored = [(tf, details[tf]['probability_score']) for tf in details if details[tf] is not None]
            if not scored:
                continue
            scored.sort(key=lambda x: x[1], reverse=True)
            ref_tf = scored[0][0]
            s = details[ref_tf]
            entry_price = s['last_close']
            stop_price = s['wave2_bottom_price'] * 0.995
            tp_price = s['wave1_top_price'] + s['wave1_len'] * tp_extension
            rr = compute_rr(entry_price, stop_price, tp_price)
            if rr is None or rr < min_rr_required:
                continue
            rr_value = rr
        entry_idx = i
        position_size = compute_size(current_equity, entry_price, stop_price, sim_risk)
        if position_size <= 0:
            continue
        outcome = simulate_trade_on_series(df_main, entry_idx, 'buy', stop_price, tp_price, entry_price)
        pnl_price = outcome['pnl']
        pnl_value = pnl_price * position_size
        actual_risk_value = position_size * max(0, entry_price - stop_price)
        planned_risk_value = current_equity * sim_risk
        trade = {
            "timestamp": ts,
            "symbol": symbol,
            "strategy": strategy,
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
            "rr": rr_value
        }
        results.append(trade)
        current_equity += pnl_value
    if len(results) == 0:
        logging.info("Nenhuma operação gerada no backtest.")
        return pd.DataFrame(), coverage_info
    df_trades = pd.DataFrame(results)
    wins = df_trades[df_trades['outcome']=='tp']
    losses = df_trades[df_trades['outcome']=='sl']
    total_pnl = df_trades['pnl'].sum()
    winrate = len(wins) / len(df_trades)
    avg_win = wins['pnl'].mean() if len(wins)>0 else 0
    avg_loss = losses['pnl'].mean() if len(losses)>0 else 0
    logging.info("Backtest summary: trades=%d winrate=%.2f total_pnl=%.4f avg_win=%.4f avg_loss=%.4f",
                 len(df_trades), winrate, total_pnl, avg_win, avg_loss)
    return df_trades, coverage_info

# ---------- Main loop ----------
def main_loop():
    logging.info("Bot iniciado — pares: %s — timeframes: %s", pairs, timeframes)
    while True:
        start = time.time()
        iteration_snapshot = []
        for symbol in pairs:
            try:
                pos, meta = analyze_and_maybe_trade(symbol)
                iteration_snapshot.append(meta)
            except Exception as e:
                logging.exception("Erro ao analisar %s: %s", symbol, e)
                iteration_snapshot.append({"symbol": symbol, "status": "error", "error": str(e)})
            time.sleep(0.5)
        # monitor open positions for addon/SL/TP
        try:
            for pos in list(open_positions):
                if pos.get('addon_pending', False):
                    check_retest_and_addon(pos)
            monitor_and_close_positions(open_positions)
        except Exception as e:
            logging.exception("Erro no monitor loop: %s", e)
        write_runtime_state(serialize_open_positions(), iteration_snapshot)
        elapsed = time.time() - start
        sleep_for = max(0, loop_interval_seconds - elapsed)
        logging.info("Iteração completa. A dormir %.1f s", sleep_for)
        time.sleep(sleep_for)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Elliott momentum breakout bot — backtest or live run')
    parser.add_argument('--live', dest='live', action='store_true', help='Run in live mode (main loop)')
    parser.add_argument('--symbol', dest='symbol', default='ETH/USDC:USDC', help='Symbol for backtest or live run')
    parser.add_argument('--timeframe', dest='timeframe', default='15m', help='Timeframe for the backtest (e.g., 15m)')
    parser.add_argument('--lookback-days', dest='lookback_days', type=int, default=60, help='Lookback days for backtest')
    parser.add_argument('--no-paper', dest='paper', action='store_false', help='Disable paper mode (will execute real orders)')
    parser.add_argument('--strategy', choices=['elliott','momentum','ema_macd'], default='elliott', help='Seleciona estratégia principal')
    args = parser.parse_args()

    logging.info("Script iniciado. paper=%s", paper)
    # apply CLI --no-paper to switch execution mode
    if hasattr(args, 'paper'):
        # apply CLI --no-paper to switch execution mode
        paper = args.paper
        if paper:
            logging.info("MODO PAPER ATIVO — ordens reais NÃO serão colocadas.")
        else:
            logging.warning("MODO REAL: As ordens reais serão enviadas — esteja certo do seu API_KEY/API_SECRET e capital.")

    strategy_mode = args.strategy
    logging.info("Estratégia ativa: %s", strategy_mode)

    if args.live:
        logging.info("Running live main loop for symbol=%s", args.symbol)
        # user should set API keys
        main_loop()
    else:
        logging.info("Running backtest for %s %s (lookback %d days)", args.symbol, args.timeframe, args.lookback_days)
        try:
            df_trades, coverage_info = backtest_pair(args.symbol, args.timeframe, lookback_days=args.lookback_days, strategy=args.strategy)
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
                    "base_capital": simulation_base_capital,
                    "risk_per_trade": simulation_risk_per_trade,
                }
            )
        except Exception as e:
            logging.error("Erro ao fazer backtest: %s", e)
