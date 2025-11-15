from __future__ import annotations

import json
import textwrap
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import streamlit as st
import pandas as pd

from elliott_momentum_breakout_bot import (
    backtest_many,
    backtest_pair,
    fetch_account_balance,
    get_environment_mode,
    validate_symbol,
)
from flexbot import context as ctx
from state_store import (
    read_user_config,
    update_user_config,
    read_control_state,
    write_control_state,
    save_timeframe_preset,
    get_timeframe_preset,
)

st.set_page_config(page_title="FlexBot Dashboard", layout="wide")


def _rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()  # type: ignore[attr-defined]

CUSTOM_CSS = """
<style>
:root {
    --flexbot-accent: #7c6cf3;
    --flexbot-accent-soft: rgba(124, 108, 243, 0.12);
    --flexbot-bg-card: rgba(22, 27, 38, 0.92);
}
.stApp > header {
    background: linear-gradient(135deg, rgba(16, 20, 30, 0.92), rgba(16, 22, 36, 0.72));
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.stApp main .block-container {
    padding-top: 1.5rem;
    background: radial-gradient(circle at 20% 20%, rgba(111, 94, 255, 0.08), transparent 35%),
                radial-gradient(circle at 80% 0%, rgba(94, 210, 255, 0.05), transparent 40%),
                linear-gradient(160deg, #0f1423 0%, #141b2d 40%, #0d101d 100%);
}
section[data-testid="stSidebar"] {
    background: #101521;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: var(--flexbot-bg-card);
    padding: 1.05rem 1.1rem;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    box-shadow: inset 0 0 0 1px rgba(124, 108, 243, 0.05);
}
.metric-card .label {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9aa0b3;
}
.metric-card .value {
    margin-top: 0.35rem;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f5f6f8;
}
.metric-card .context {
    margin-top: 0.45rem;
    font-size: 0.82rem;
    color: #6d7380;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(124, 108, 243, 0.18);
    color: #c7c3fb;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-pill.live {
    background: rgba(16, 199, 151, 0.18);
    color: #57efc9;
}
.status-pill.paper {
    background: rgba(124, 108, 243, 0.18);
    color: #c7c3fb;
}
.status-pill.slim {
    background: rgba(120, 129, 255, 0.12);
    color: #9ea7ff;
}
.hero-card {
    margin-bottom: 1.4rem;
    padding: 1.25rem 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(24, 34, 61, 0.9), rgba(18, 22, 33, 0.78));
    box-shadow: 0 20px 36px rgba(10, 12, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.hero-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #f3f4fa;
}
.hero-meta {
    margin-top: 0.35rem;
    color: #8791a9;
    font-size: 0.9rem;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #aab2c8;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid var(--flexbot-accent) !important;
}
.stButton>button,
.stDownloadButton>button {
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.1rem;
    background: linear-gradient(135deg, #7c6cf3, #5d8af0);
    color: white;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 22px rgba(94, 115, 245, 0.25);
}
.stSlider>div>div>div>div {
    background: linear-gradient(135deg, rgba(124,108,243,0.6), rgba(93,138,240,0.6));
}
.card-section {
    padding: 1.1rem 1.3rem;
    border-radius: 16px;
    background: rgba(17, 25, 40, 0.76);
    border: 1px solid rgba(255, 255, 255, 0.04);
    box-shadow: 0 14px 30px rgba(10, 12, 20, 0.25);
    margin-bottom: 1.4rem;
}
.card-section h3 {
    margin-bottom: 0.8rem;
}
.stDataFrame thead tr th {
    background: rgba(24, 29, 45, 0.92);
    color: #cbd3f5;
}
.stDataFrame tbody tr {
    background: rgba(12, 16, 28, 0.68);
}
.stDataFrame tbody tr:hover {
    background: rgba(39, 45, 68, 0.8) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("📈 FlexBot Dashboard")
st.caption("Visualize resultados de backtests e acompanhe métricas do loop live.")

if "preset_feedback" in st.session_state:
    st.success(st.session_state.pop("preset_feedback"))

STATE_DIR = Path('.flexbot_state')
RUNTIME_FILE = STATE_DIR / 'runtime.json'
BACKTEST_FILE = STATE_DIR / 'latest_backtest.json'
ROOT_DIR = Path(__file__).resolve().parent


def _pid_exists(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False
    except Exception:
        return False


def _terminate_pid(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            return result.returncode == 0
        os.kill(pid, signal.SIGTERM)
        return True
    except FileNotFoundError:
        return False
    except PermissionError:
        return False
    except OSError:
        return False
    except Exception:
        return False

with st.sidebar:
    st.header("Parâmetros")
    config_data = read_user_config()
    env_options = ["paper", "live"]
    env_labels = {
        "paper": "Teste (Paper)",
        "live": "Produção (Live)",
    }
    current_env = config_data.get("environment", get_environment_mode())
    if current_env not in env_options:
        current_env = "paper"
    env_choice = st.radio(
        "Ambiente",
        options=env_options,
        index=env_options.index(current_env),
        format_func=lambda k: env_labels.get(k, k),
        help="Alterna entre o modo Paper (testes) e Live (ordens reais). O script principal segue esta preferência a menos que seja sobrescrito na CLI.",
    )
    if env_choice != current_env:
        update_user_config(environment=env_choice)
        st.success(f"Ambiente atualizado para {env_labels.get(env_choice, env_choice)}.")

    if "account_balance" not in st.session_state:
        st.session_state["account_balance"] = None
    if st.button("Consultar saldo", use_container_width=True):
        balance_value = fetch_account_balance()
        st.session_state["account_balance"] = balance_value
    if st.session_state["account_balance"] is not None:
        st.metric("Saldo estimado", f"${st.session_state['account_balance']:.2f}")
        st.caption("Valor estimado reportado pela corretora; se estiver em paper/fallback, o total padrão é $10.000.")

    multi_asset_default = config_data.get("multi_asset_enabled")
    if not isinstance(multi_asset_default, bool):
        multi_asset_default = getattr(ctx, "multi_asset_enabled", True)
    multi_asset_enabled = st.checkbox(
        "Monitorizar múltiplos ativos",
        value=multi_asset_default,
        help="Quando desativado, o loop acompanha apenas o par selecionado acima.",
    )

    default_symbol = config_data.get("symbol")
    available_pairs = list(ctx.pairs)
    symbol_index = available_pairs.index(default_symbol) if isinstance(default_symbol, str) and default_symbol in available_pairs else 0
    pair_label = "Par principal" if multi_asset_enabled else "Par"
    pair_help = "Define o par usado como referência (ex.: backtests e valores padrão)." if multi_asset_enabled else "Seleciona o par que o loop vai acompanhar."
    symbol = st.selectbox(pair_label, options=available_pairs, index=symbol_index, help=pair_help)

    configured_active_pairs = config_data.get("active_pairs")
    if not isinstance(configured_active_pairs, list) or not configured_active_pairs:
        configured_active_pairs = list(ctx.active_pairs)
    configured_active_pairs = [p for p in configured_active_pairs if p in available_pairs]
    if not configured_active_pairs:
        configured_active_pairs = available_pairs[:3]
    if symbol not in configured_active_pairs:
        configured_active_pairs = [symbol] + [p for p in configured_active_pairs if p != symbol]

    if multi_asset_enabled:
        active_pairs = st.multiselect(
            "Ativos monitorizados",
            options=available_pairs,
            default=configured_active_pairs,
            help="Lista de pares que o loop live irá monitorizar. Mantemos no máximo um trade aberto por par.",
        )
        if not active_pairs:
            st.warning("Selecione pelo menos um par para o loop.")
            active_pairs = configured_active_pairs
        if symbol not in active_pairs:
            st.info("Par principal adicionado automaticamente à lista de monitorização.")
            active_pairs = [symbol] + [p for p in active_pairs if p != symbol]
    else:
        active_pairs = [symbol]
    default_timeframe = config_data.get("timeframe")
    available_timeframes = list(ctx.timeframes)
    default_tf_index = available_timeframes.index(default_timeframe) if isinstance(default_timeframe, str) and default_timeframe in available_timeframes else 1 if len(available_timeframes) > 1 else 0
    timeframe = st.selectbox("Timeframe", options=available_timeframes, index=default_tf_index)
    lookback_days = st.slider("Lookback (dias)", min_value=30, max_value=180, value=60, step=5)
    strategy_labels = {
        "momentum": "Momentum",
        "ema_macd": "OMDs",
    }
    configured_strategy = config_data.get("strategy_mode", ctx.strategy_mode)
    if configured_strategy not in strategy_labels:
        configured_strategy = ctx.strategy_mode if ctx.strategy_mode in strategy_labels else "momentum"
    strategy_options = list(strategy_labels.keys())
    strategy_idx = strategy_options.index(configured_strategy)
    strategy = st.selectbox(
        "Estratégia",
        options=strategy_options,
        index=strategy_idx,
        format_func=lambda key: strategy_labels.get(key, key),
    )
    strategy_display = strategy_labels.get(strategy, strategy)
    bias_label = {
        "long": "Comprado",
        "short": "Vendido",
        "both": "Ambos",
    }
    bias_options = ["long", "short", "both"]
    default_bias = config_data.get("trade_bias")
    if default_bias not in bias_options:
        default_bias = "long"
    trade_bias = st.selectbox(
        "Direção",
        options=bias_options,
        format_func=lambda k: bias_label.get(k, k),
        index=bias_options.index(default_bias),
    )

    stored_fixed_bias_flag = config_data.get("use_fixed_bias_timeframe")
    if not isinstance(stored_fixed_bias_flag, bool):
        stored_fixed_bias_flag = ctx.use_fixed_bias_timeframe
    use_fixed_bias_value = st.checkbox(
        "Forçar bias pelo 4h",
        value=stored_fixed_bias_flag,
        help="Quando ativo, todas as entradas consultam a tendência do timeframe 4h como filtro direcional.",
    )

    risk_mode_labels = {
        "standard": "Alocado",
        "hunter": "Hunter (100%)",
    }
    current_risk_mode = config_data.get("risk_mode", ctx.risk_mode)
    if current_risk_mode not in risk_mode_labels:
        current_risk_mode = ctx.risk_mode
    risk_mode_choice = st.selectbox(
        "Modo de risco",
        options=list(risk_mode_labels.keys()),
        index=list(risk_mode_labels.keys()).index(current_risk_mode),
        format_func=lambda k: risk_mode_labels.get(k, k),
    )

    configured_risk_percent = config_data.get("risk_percent")
    try:
        configured_risk_percent = float(configured_risk_percent)
    except (TypeError, ValueError):
        configured_risk_percent = ctx.risk_percent
    if configured_risk_percent <= 0:
        configured_risk_percent = ctx.risk_percent
    risk_percent_value = st.number_input(
        "Risco por trade (%)",
        min_value=0.01,
        max_value=100.0,
        value=float(round(configured_risk_percent * 100, 2)),
        step=0.05,
        help="Percentual do capital alocado que será arriscado até o stop. No modo Hunter, aplica-se sobre o saldo completo.",
        format="%.2f",
    ) / 100.0

    configured_capital_base = config_data.get("capital_base", ctx.capital_base)
    try:
        configured_capital_base = float(configured_capital_base)
    except (TypeError, ValueError):
        configured_capital_base = ctx.capital_base
    capital_base_value = st.number_input(
        "Capital base para sizing",
        min_value=0.0,
        value=float(round(configured_capital_base, 2)),
        step=50.0,
        help="Montante máximo considerado para calcular o risco. Se o saldo for menor, usa o saldo disponível.",
        disabled=risk_mode_choice == "hunter",
        format="%.2f",
    )

    configured_leverage = config_data.get("leverage", ctx.leverage)
    try:
        configured_leverage = float(configured_leverage)
    except (TypeError, ValueError):
        configured_leverage = ctx.leverage
    leverage_value = st.number_input(
        "Alavancagem alvo",
        min_value=1.0,
        max_value=125.0,
        value=float(round(configured_leverage, 2)),
        step=0.5,
        help="Usado para limitar o tamanho máximo da posição de acordo com a margem disponível.",
        format="%.2f",
    )
    default_cross = config_data.get("ema_cross_lookback", 8)
    if not isinstance(default_cross, int) or default_cross < 2 or default_cross > 30:
        default_cross = 8
    require_default = config_data.get("ema_require_divergence")
    if not isinstance(require_default, bool):
        require_default = True
    rsi_zone_default = config_data.get("ema_require_rsi_zone")
    if not isinstance(rsi_zone_default, bool):
        rsi_zone_default = ctx.ema_require_rsi_zone
    configured_drop_pct = config_data.get("divergence_min_drop_pct")
    try:
        configured_drop_pct = float(configured_drop_pct)
    except (TypeError, ValueError):
        configured_drop_pct = ctx.divergence_min_drop_pct
    configured_rsi_long_max = config_data.get("ema_rsi_zone_long_max", ctx.ema_rsi_zone_long_max)
    try:
        configured_rsi_long_max = float(configured_rsi_long_max)
    except (TypeError, ValueError):
        configured_rsi_long_max = ctx.ema_rsi_zone_long_max
    configured_rsi_short_min = config_data.get("ema_rsi_zone_short_min", ctx.ema_rsi_zone_short_min)
    try:
        configured_rsi_short_min = float(configured_rsi_short_min)
    except (TypeError, ValueError):
        configured_rsi_short_min = ctx.ema_rsi_zone_short_min
    configured_trailing_flag = config_data.get("ema_macd_use_trailing")
    if not isinstance(configured_trailing_flag, bool):
        configured_trailing_flag = ctx.ema_macd_use_trailing
    configured_trailing_rr = config_data.get("ema_macd_trailing_rr", ctx.ema_macd_trailing_rr)
    try:
        configured_trailing_rr = float(configured_trailing_rr)
    except (TypeError, ValueError):
        configured_trailing_rr = ctx.ema_macd_trailing_rr
    if configured_trailing_rr <= 0:
        configured_trailing_rr = ctx.ema_macd_trailing_rr
    configured_trailing_activate = config_data.get("ema_macd_trailing_activate_rr", ctx.ema_macd_trailing_activate_rr)
    try:
        configured_trailing_activate = float(configured_trailing_activate)
    except (TypeError, ValueError):
        configured_trailing_activate = ctx.ema_macd_trailing_activate_rr
    if configured_trailing_activate < 0:
        configured_trailing_activate = ctx.ema_macd_trailing_activate_rr
    momentum_require_default = config_data.get("momentum_require_divergence")
    if not isinstance(momentum_require_default, bool):
        momentum_require_default = ctx.momentum_require_divergence
    momentum_bonus_default = config_data.get("momentum_use_divergence_bonus")
    if not isinstance(momentum_bonus_default, bool):
        momentum_bonus_default = ctx.momentum_use_divergence_bonus
    configured_momentum_rsi_long = config_data.get("momentum_rsi_long_max", ctx.momentum_rsi_long_max)
    try:
        configured_momentum_rsi_long = float(configured_momentum_rsi_long)
    except (TypeError, ValueError):
        configured_momentum_rsi_long = ctx.momentum_rsi_long_max
    configured_momentum_rsi_short = config_data.get("momentum_rsi_short_min", ctx.momentum_rsi_short_min)
    try:
        configured_momentum_rsi_short = float(configured_momentum_rsi_short)
    except (TypeError, ValueError):
        configured_momentum_rsi_short = ctx.momentum_rsi_short_min
    configured_momentum_trailing_flag = config_data.get("momentum_use_trailing")
    if not isinstance(configured_momentum_trailing_flag, bool):
        configured_momentum_trailing_flag = ctx.momentum_use_trailing
    configured_momentum_trailing_rr = config_data.get("momentum_trailing_rr", ctx.momentum_trailing_rr)
    try:
        configured_momentum_trailing_rr = float(configured_momentum_trailing_rr)
    except (TypeError, ValueError):
        configured_momentum_trailing_rr = ctx.momentum_trailing_rr
    if configured_momentum_trailing_rr <= 0:
        configured_momentum_trailing_rr = ctx.momentum_trailing_rr
    configured_momentum_trailing_activate = config_data.get("momentum_trailing_activate_rr", ctx.momentum_trailing_activate_rr)
    try:
        configured_momentum_trailing_activate = float(configured_momentum_trailing_activate)
    except (TypeError, ValueError):
        configured_momentum_trailing_activate = ctx.momentum_trailing_activate_rr
    if configured_momentum_trailing_activate < 0:
        configured_momentum_trailing_activate = ctx.momentum_trailing_activate_rr
    if configured_drop_pct < 0:
        configured_drop_pct = 0.0
    drop_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    closest_drop = min(drop_options, key=lambda opt: abs(opt - configured_drop_pct * 100))
    momentum_require_divergence_value = momentum_require_default
    momentum_use_divergence_bonus_value = momentum_bonus_default
    momentum_rsi_long_max_value = configured_momentum_rsi_long
    momentum_rsi_short_min_value = configured_momentum_rsi_short
    use_trailing_value = configured_trailing_flag
    trailing_rr_value = configured_trailing_rr
    trailing_activate_value = configured_trailing_activate
    momentum_use_trailing_value = configured_momentum_trailing_flag
    momentum_trailing_rr_value = configured_momentum_trailing_rr
    momentum_trailing_activate_value = configured_momentum_trailing_activate
    if strategy == "ema_macd":
        cross_lookback = st.slider("Velas para cruzamento OMDs", min_value=2, max_value=30, value=default_cross, step=1)
        require_divergence = st.checkbox(
            "Exigir divergência RSI (OMDs)",
            value=require_default,
            help="Quando desmarcado, a confirmação OMDs aceita sinais sem divergência RSI.",
        )
        require_rsi_zone = st.checkbox(
            "Exigir RSI em zona (OMDs)",
            value=rsi_zone_default,
            help="Quando marcada, a confirmação OMDs só aceita sinais com RSI em zona definida (long ≤ limite, short ≥ limite).",
        )
        if require_rsi_zone:
            rsi_long_max = st.slider(
                "RSI máximo (long ≤ valor)",
                min_value=10.0,
                max_value=60.0,
                value=float(round(configured_rsi_long_max, 1)),
                step=1.0,
                help="Quando ativo, o RSI precisa estar abaixo ou igual a este valor para validar entries long.",
            )
            rsi_short_min = st.slider(
                "RSI mínimo (short ≥ valor)",
                min_value=40.0,
                max_value=90.0,
                value=float(round(configured_rsi_short_min, 1)),
                step=1.0,
                help="Quando ativo, o RSI precisa estar acima ou igual a este valor para validar entries short.",
            )
        else:
            rsi_long_max = configured_rsi_long_max
            rsi_short_min = configured_rsi_short_min
        if require_divergence:
            divergence_min_drop_choice = st.select_slider(
                "Queda mínima para divergência (%)",
                options=drop_options,
                value=closest_drop,
                help="Define o quanto o preço precisa recuar entre pivôs consecutivos para a divergência RSI ser considerada (0 desliga o filtro).",
            )
        else:
            divergence_min_drop_choice = closest_drop
        use_trailing_value = st.checkbox(
            "Ativar trailing stop dinâmico",
            value=configured_trailing_flag,
            help="Quando ativo, move o stop conforme o trade avança após atingir o RR mínimo configurado.",
        )
        trailing_activate_value = st.slider(
            "Ativar trailing após RR",
            min_value=0.0,
            max_value=6.0,
            value=float(round(configured_trailing_activate, 1)),
            step=0.1,
            help="RR mínimo alcançado para começar a ajustar o stop. Por exemplo, 2.0 inicia o trailing após 2R.",
            disabled=not use_trailing_value,
        )
        trailing_rr_value = st.slider(
            "Distância do trailing (RR)",
            min_value=0.2,
            max_value=3.0,
            value=float(round(configured_trailing_rr, 1)),
            step=0.1,
            help="Quanto do RR manter entre o preço e o stop dinâmico. 1.0 mantém o stop a 1R do topo atual.",
            disabled=not use_trailing_value,
        )
    elif strategy == "momentum":
        cross_lookback = default_cross
        require_divergence = require_default
        require_rsi_zone = rsi_zone_default
        rsi_long_max = configured_rsi_long_max
        rsi_short_min = configured_rsi_short_min
        divergence_min_drop_choice = closest_drop
        st.subheader("Ajustes Momentum")
        momentum_require_divergence_value = st.checkbox(
            "Exigir divergência RSI (Momentum)",
            value=momentum_require_default,
            help="Quando marcado, sinais Momentum só validam quando há divergência RSI na direção esperada.",
        )
        momentum_use_divergence_bonus_value = st.checkbox(
            "Somar bônus por divergência (Momentum)",
            value=momentum_bonus_default,
            help="Quando ativo, adiciona pontuação extra aos sinais Momentum com divergência RSI.",
        )
        momentum_rsi_long_max_value = st.slider(
            "RSI máximo (Momentum long ≤ valor)",
            min_value=40.0,
            max_value=80.0,
            value=float(round(configured_momentum_rsi_long, 1)),
            step=1.0,
            help="Limite de RSI permitido para entradas long Momentum.",
        )
        momentum_rsi_short_min_value = st.slider(
            "RSI mínimo (Momentum short ≥ valor)",
            min_value=20.0,
            max_value=70.0,
            value=float(round(configured_momentum_rsi_short, 1)),
            step=1.0,
            help="Limite inferior de RSI para aceitar entradas short Momentum.",
        )
        momentum_use_trailing_value = st.checkbox(
            "Ativar trailing stop (Momentum)",
            value=configured_momentum_trailing_flag,
            help="Quando ativo, Momentum move o stop após atingir o RR mínimo configurado.",
        )
        momentum_trailing_activate_value = st.slider(
            "Ativar trailing Momentum após RR",
            min_value=0.0,
            max_value=6.0,
            value=float(round(configured_momentum_trailing_activate, 1)),
            step=0.1,
            help="RR mínimo alcançado para iniciar o trailing nos trades Momentum.",
            disabled=not momentum_use_trailing_value,
        )
        momentum_trailing_rr_value = st.slider(
            "Distância trailing Momentum (RR)",
            min_value=0.2,
            max_value=3.0,
            value=float(round(configured_momentum_trailing_rr, 1)),
            step=0.1,
            help="Quanto de RR manter entre o preço e o stop ao ajustar Momentum.",
            disabled=not momentum_use_trailing_value,
        )
    else:
        cross_lookback = default_cross
        require_divergence = require_default
        require_rsi_zone = rsi_zone_default
        rsi_long_max = configured_rsi_long_max
        rsi_short_min = configured_rsi_short_min
        divergence_min_drop_choice = closest_drop
        st.caption("Parâmetros de cruzamento/divergência aplicam-se apenas à estratégia OMDs.")
    divergence_min_drop_pct = max(0.0, float(divergence_min_drop_choice) / 100.0)
    ctx.divergence_min_drop_pct = divergence_min_drop_pct

    preset_payload = {
        "strategy_mode": strategy,
        "trade_bias": trade_bias,
        "ema_cross_lookback": cross_lookback,
        "ema_require_divergence": require_divergence,
        "ema_require_rsi_zone": require_rsi_zone,
        "divergence_min_drop_pct": divergence_min_drop_pct,
        "ema_rsi_zone_long_max": rsi_long_max,
        "ema_rsi_zone_short_min": rsi_short_min,
        "use_fixed_bias_timeframe": use_fixed_bias_value,
        "fixed_bias_timeframe": ctx.fixed_bias_timeframe,
        "ema_macd_use_trailing": use_trailing_value,
        "ema_macd_trailing_rr": trailing_rr_value,
        "ema_macd_trailing_activate_rr": trailing_activate_value,
        "risk_mode": risk_mode_choice,
        "risk_percent": risk_percent_value,
        "capital_base": capital_base_value,
        "leverage": leverage_value,
        "multi_asset_enabled": multi_asset_enabled,
        "active_pairs": active_pairs,
        "momentum_require_divergence": momentum_require_divergence_value,
        "momentum_use_divergence_bonus": momentum_use_divergence_bonus_value,
        "momentum_rsi_long_max": momentum_rsi_long_max_value,
        "momentum_rsi_short_min": momentum_rsi_short_min_value,
        "momentum_use_trailing": momentum_use_trailing_value,
        "momentum_trailing_rr": momentum_trailing_rr_value,
        "momentum_trailing_activate_rr": momentum_trailing_activate_value,
    }
    preset_cols = st.columns(2)
    if preset_cols[0].button("💾 G", use_container_width=True):
        save_timeframe_preset(timeframe, preset_payload)
        st.session_state["preset_feedback"] = f"Preset para {timeframe} atualizado."
        _rerun_app()
    current_preset = get_timeframe_preset(timeframe, strategy=strategy)
    restore_disabled = current_preset is None
    if preset_cols[1].button("🔄 R", use_container_width=True, disabled=restore_disabled):
        if current_preset:
            update_user_config(**current_preset, timeframe=timeframe)
            st.session_state["preset_feedback"] = f"Preset para {timeframe} restaurado."
            _rerun_app()
    if (
        config_data.get("symbol") != symbol
        or config_data.get("timeframe") != timeframe
        or config_data.get("trade_bias") != trade_bias
        or config_data.get("ema_cross_lookback") != cross_lookback
        or config_data.get("ema_require_divergence") != require_divergence
        or config_data.get("ema_require_rsi_zone") != require_rsi_zone
        or config_data.get("use_fixed_bias_timeframe") != use_fixed_bias_value
        or config_data.get("fixed_bias_timeframe") != ctx.fixed_bias_timeframe
        or bool(configured_trailing_flag) != bool(use_trailing_value)
        or abs(configured_trailing_rr - trailing_rr_value) > 1e-9
        or abs(configured_trailing_activate - trailing_activate_value) > 1e-9
        or abs(configured_drop_pct - divergence_min_drop_pct) > 1e-9
        or float(configured_rsi_long_max) != float(rsi_long_max)
        or float(configured_rsi_short_min) != float(rsi_short_min)
        or config_data.get("momentum_require_divergence") != momentum_require_divergence_value
        or config_data.get("momentum_use_divergence_bonus") != momentum_use_divergence_bonus_value
        or float(configured_momentum_rsi_long) != float(momentum_rsi_long_max_value)
        or float(configured_momentum_rsi_short) != float(momentum_rsi_short_min_value)
        or bool(configured_momentum_trailing_flag) != bool(momentum_use_trailing_value)
        or abs(configured_momentum_trailing_rr - momentum_trailing_rr_value) > 1e-9
        or abs(configured_momentum_trailing_activate - momentum_trailing_activate_value) > 1e-9
        or config_data.get("risk_mode") != risk_mode_choice
        or abs(configured_risk_percent - risk_percent_value) > 1e-9
        or abs(configured_capital_base - capital_base_value) > 1e-9
        or abs(configured_leverage - leverage_value) > 1e-9
        or config_data.get("strategy_mode") != strategy
        or set(config_data.get("active_pairs", [])) != set(active_pairs)
        or config_data.get("multi_asset_enabled") != multi_asset_enabled
    ):
        config_data = update_user_config(
            symbol=symbol,
            timeframe=timeframe,
            trade_bias=trade_bias,
            ema_cross_lookback=cross_lookback,
            ema_require_divergence=require_divergence,
            ema_require_rsi_zone=require_rsi_zone,
            divergence_min_drop_pct=divergence_min_drop_pct,
            ema_rsi_zone_long_max=rsi_long_max,
            ema_rsi_zone_short_min=rsi_short_min,
            use_fixed_bias_timeframe=use_fixed_bias_value,
            fixed_bias_timeframe=ctx.fixed_bias_timeframe,
            ema_macd_use_trailing=use_trailing_value,
            ema_macd_trailing_rr=trailing_rr_value,
            ema_macd_trailing_activate_rr=trailing_activate_value,
            momentum_require_divergence=momentum_require_divergence_value,
            momentum_use_divergence_bonus=momentum_use_divergence_bonus_value,
            momentum_rsi_long_max=momentum_rsi_long_max_value,
            momentum_rsi_short_min=momentum_rsi_short_min_value,
            momentum_use_trailing=momentum_use_trailing_value,
            momentum_trailing_rr=momentum_trailing_rr_value,
            momentum_trailing_activate_rr=momentum_trailing_activate_value,
            strategy_mode=strategy,
            risk_mode=risk_mode_choice,
            risk_percent=risk_percent_value,
            capital_base=capital_base_value,
            leverage=leverage_value,
            active_pairs=active_pairs,
            multi_asset_enabled=multi_asset_enabled,
        )

    ctx.strategy_mode = strategy
    ctx.set_multi_asset_enabled(multi_asset_enabled)
    ctx.set_active_pairs(active_pairs)
    ctx.divergence_min_drop_pct = divergence_min_drop_pct
    ctx.ema_require_rsi_zone = require_rsi_zone
    ctx.ema_rsi_zone_long_max = rsi_long_max
    ctx.ema_rsi_zone_short_min = rsi_short_min
    ctx.use_fixed_bias_timeframe = use_fixed_bias_value
    ctx.ema_macd_use_trailing = use_trailing_value
    ctx.ema_macd_trailing_rr = trailing_rr_value
    ctx.ema_macd_trailing_activate_rr = trailing_activate_value
    ctx.momentum_require_divergence = momentum_require_divergence_value
    ctx.momentum_use_divergence_bonus = momentum_use_divergence_bonus_value
    ctx.momentum_rsi_long_max = momentum_rsi_long_max_value
    ctx.momentum_rsi_short_min = momentum_rsi_short_min_value
    ctx.momentum_use_trailing = momentum_use_trailing_value
    ctx.momentum_trailing_rr = momentum_trailing_rr_value
    ctx.momentum_trailing_activate_rr = momentum_trailing_activate_value
    symbols_for_backtest = (
        tuple(dict.fromkeys(active_pairs))
        if multi_asset_enabled and len(active_pairs) > 1
        else (symbol,)
    )
    if "pair_validation" not in st.session_state:
        st.session_state["pair_validation"] = None
    pair_to_validate = st.text_input("Validar par disponível na corretora", value=symbol)
    if st.button("Validar par", use_container_width=True):
        is_valid = validate_symbol(pair_to_validate.strip()) if pair_to_validate else False
        st.session_state["pair_validation"] = {
            "symbol": pair_to_validate.strip(),
            "is_valid": is_valid,
        }
    validation_state = st.session_state.get("pair_validation")
    if validation_state is not None and validation_state.get("symbol"):
        if validation_state["is_valid"]:
            st.success(f"Par {validation_state['symbol']} disponível para negociação.")
        else:
            st.error(f"Par {validation_state['symbol']} não encontrado ou indisponível na corretora.")

    run_button = st.button("▶️ Executar backtest", use_container_width=True)
    st.markdown("---")
    st.caption(".")

divergence_badge = "On" if require_divergence else "Off"
risk_badge_label = (
    f"Hunter · Risco {risk_percent_value*100:.1f}%"
    if risk_mode_choice == "hunter"
    else f"Risco {risk_percent_value*100:.1f}% · Base {capital_base_value:.0f}"
)
leverage_badge_label = f"Lev {leverage_value:.1f}x"
if multi_asset_enabled:
    assets_badge_label = ", ".join(active_pairs[:3]) + ("…" if len(active_pairs) > 3 else "")
    assets_badge_title = "Ativos"
else:
    assets_badge_label = symbol
    assets_badge_title = "Ativo"
meta_parts = [
    timeframe,
    f"Lookback {lookback_days} dias",
]
if strategy == "ema_macd":
    meta_parts.append(f"Cross {cross_lookback} velas")
badge_spans = [
    f"<span class='status-pill {env_choice}'>{env_labels.get(env_choice, env_choice.title())}</span>",
    f"<span class='status-pill slim'>{strategy_display}</span>",
    f"<span class='status-pill slim'>Bias {bias_label.get(trade_bias, trade_bias)}</span>",
    f"<span class='status-pill slim'>{risk_badge_label}</span>",
    f"<span class='status-pill slim'>{assets_badge_title} {assets_badge_label}</span>",
    f"<span class='status-pill slim'>{leverage_badge_label}</span>",
]
if strategy == "ema_macd":
    badge_spans.insert(2, f"<span class='status-pill slim'>Divergência {divergence_badge}</span>")
badges_html = "\n            ".join(badge_spans)
summary_html = textwrap.dedent(
    f"""
    <div class='hero-card'>
        <div>
            <div class='hero-title'>{symbol}</div>
            <div class='hero-meta'>{' · '.join(meta_parts)}</div>
        </div>
        <div class='hero-badges'>
            {badges_html}
        </div>
    </div>
    """
)
st.markdown(summary_html, unsafe_allow_html=True)

control_state = read_control_state()
if "bot_handle" not in st.session_state:
    st.session_state["bot_handle"] = None
if "bot_meta" not in st.session_state:
    st.session_state["bot_meta"] = control_state if control_state else None

handle = st.session_state.get("bot_handle")
running_via_handle = handle is not None and handle.poll() is None
running_via_pid = False
if not running_via_handle and control_state.get("pid"):
    running_via_pid = _pid_exists(control_state["pid"])
    if running_via_pid and st.session_state.get("bot_meta") is None:
        st.session_state["bot_meta"] = control_state

if control_state and not (running_via_handle or running_via_pid):
    control_state = {}
    write_control_state({})

is_loop_running = running_via_handle or running_via_pid
active_meta = st.session_state.get("bot_meta") or control_state or {}

st.markdown("<div class='card-section'>", unsafe_allow_html=True)
st.subheader("Controle do loop live/paper")

status_label = "Loop em execução" if is_loop_running else "Loop parado"
status_badge = "live" if active_meta.get("environment") == "live" else "paper"
status_badge_html = f"<span class='status-pill {status_badge}'>{status_label}</span>"
st.markdown(status_badge_html, unsafe_allow_html=True)

if is_loop_running:
    started_at = active_meta.get("started_at")
    started_str = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(started_at)) if started_at else "—"
    symbol_running = active_meta.get("symbol", symbol)
    timeframe_running = active_meta.get("timeframe", timeframe)
    st.info(
        f"Rodando para {symbol_running} @ {timeframe_running} · PID {active_meta.get('pid', '—')} · iniciado {started_str}"
    )
    if active_meta.get("command_line"):
        st.code(active_meta["command_line"], language="bash")
else:
    st.caption("Nenhum processo ativo detectado. Inicie o loop para sincronizar métricas em tempo real.")

start_col, stop_col = st.columns(2)

with start_col:
    start_clicked = st.button(
        "🚀 Iniciar loop",
        use_container_width=True,
        disabled=is_loop_running,
        key="start_loop_button",
    )
with stop_col:
    stop_clicked = st.button(
        "🛑 Parar loop",
        use_container_width=True,
        disabled=not is_loop_running,
        key="stop_loop_button",
    )

loop_feedback = None
if start_clicked:
    monitored_pairs = active_pairs if active_pairs else list(ctx.active_pairs)
    command = [
        sys.executable,
        "elliott_momentum_breakout_bot.py",
        "--live",
        "--timeframe",
        timeframe,
        "--strategy",
        strategy,
        "--trade-bias",
        trade_bias,
        "--cross-lookback",
        str(cross_lookback),
        "--all-pairs",
    ]
    if symbol:
        command.extend(["--symbol", symbol])
    command.append("--require-divergence" if require_divergence else "--allow-no-divergence")
    command.append("--paper-mode" if env_choice == "paper" else "--no-paper")

    try:
        process = subprocess.Popen(command, cwd=ROOT_DIR)
    except Exception as exc:
        loop_feedback = ("error", f"Falha ao iniciar o loop: {exc}")
    else:
        started_meta = {
            "pid": process.pid,
            "symbol": symbol,
            "symbols": monitored_pairs,
            "timeframe": timeframe,
            "entry_timeframes": [timeframe],
            "environment": env_choice,
            "trade_bias": trade_bias,
            "cross_lookback": cross_lookback,
            "require_divergence": require_divergence,
            "command": command,
            "command_line": " ".join(command),
            "started_at": time.time(),
        }
        st.session_state["bot_handle"] = process
        st.session_state["bot_meta"] = started_meta
        write_control_state(started_meta)
        control_state = started_meta
        is_loop_running = True
        loop_feedback = ("success", "Loop principal iniciado; aguarde a primeira atualização do runtime.")

if stop_clicked:
    terminated = False
    handle = st.session_state.get("bot_handle")
    if handle and handle.poll() is None:
        try:
            handle.terminate()
            handle.wait(timeout=10)
            terminated = True
        except subprocess.TimeoutExpired:
            handle.kill()
            terminated = True
        except Exception:
            terminated = False
    if not terminated:
        pid = active_meta.get("pid")
        if pid:
            terminated = _terminate_pid(pid)
    if terminated:
        st.session_state["bot_handle"] = None
        st.session_state["bot_meta"] = None
        write_control_state({})
        control_state = {}
        is_loop_running = False
        loop_feedback = ("success", "Loop interrompido com sucesso.")
    else:
        loop_feedback = ("error", "Não foi possível encerrar o loop automaticamente. Finalize o processo manualmente.")

if loop_feedback:
    level, message = loop_feedback
    if level == "success":
        st.success(message)
    else:
        st.error(message)

st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run_backtest(
    symbols: Tuple[str, ...],
    timeframe: str,
    lookback_days: int,
    strategy: str,
    bias: str,
    cross_lookback: int | None,
    require_divergence: bool | None,
    divergence_min_drop_pct: float,
    require_rsi_zone: bool,
    rsi_zone_long_max: float,
    rsi_zone_short_min: float,
    momentum_require_divergence: bool,
    momentum_use_divergence_bonus: bool,
    momentum_rsi_long_max: float,
    momentum_rsi_short_min: float,
    momentum_use_trailing: bool,
    momentum_trailing_rr: float,
    momentum_trailing_activate_rr: float,
):
    multi = len(symbols) > 1
    if abs(ctx.divergence_min_drop_pct - divergence_min_drop_pct) > 1e-12:
        ctx.divergence_min_drop_pct = divergence_min_drop_pct
    ctx.ema_require_rsi_zone = require_rsi_zone
    ctx.ema_rsi_zone_long_max = rsi_zone_long_max
    ctx.ema_rsi_zone_short_min = rsi_zone_short_min
    ctx.momentum_require_divergence = momentum_require_divergence
    ctx.momentum_use_divergence_bonus = momentum_use_divergence_bonus
    ctx.momentum_rsi_long_max = momentum_rsi_long_max
    ctx.momentum_rsi_short_min = momentum_rsi_short_min
    ctx.momentum_use_trailing = momentum_use_trailing
    ctx.momentum_trailing_rr = momentum_trailing_rr
    ctx.momentum_trailing_activate_rr = momentum_trailing_activate_rr
    if multi:
        df_trades, coverage = backtest_many(
            symbols,
            timeframe,
            lookback_days=lookback_days,
            strategy=strategy,
            bias=bias,
            cross_lookback=cross_lookback,
            require_divergence=require_divergence,
        )
    else:
        df_trades, coverage = backtest_pair(
            symbols[0],
            timeframe,
            lookback_days=lookback_days,
            strategy=strategy,
            bias=bias,
            cross_lookback=cross_lookback,
            require_divergence=require_divergence,
        )
    if df_trades is None or df_trades.empty:
        return None, coverage, multi
    df_trades = df_trades.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_trades["timestamp"]):
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    if "symbol" not in df_trades.columns:
        df_trades["symbol"] = symbols[0]
    df_trades.sort_values("timestamp", inplace=True)
    df_trades["cum_pnl"] = df_trades["pnl"].cumsum()
    df_trades["cum_pnl_symbol"] = df_trades.groupby("symbol")["pnl"].cumsum()
    return df_trades, coverage, multi

backtest_tab, realtime_tab = st.tabs(["Backtest", "Tempo real"])

with backtest_tab:
    if run_button:
        symbol_display = ", ".join(symbols_for_backtest)
        with st.spinner(
            f"Gerando backtest {symbol_display} {timeframe} (últimos {lookback_days} dias) — estratégia {strategy_display} | bias {bias_label.get(trade_bias, trade_bias)}"
            + (f" | cross {cross_lookback} velas" if strategy == "ema_macd" else "")
            + (" | divergência " + ("obrigatória" if require_divergence else "opcional") if strategy == "ema_macd" else "")
            + "..."
        ):
                df, coverage, multi_mode = run_backtest(
                    symbols_for_backtest,
                    timeframe,
                    lookback_days,
                    strategy,
                    trade_bias,
                    cross_lookback if strategy == "ema_macd" else None,
                    require_divergence if strategy == "ema_macd" else None,
                    divergence_min_drop_pct,
                    require_rsi_zone,
                    rsi_long_max,
                    rsi_short_min,
                    momentum_require_divergence_value,
                    momentum_use_divergence_bonus_value,
                    momentum_rsi_long_max_value,
                    momentum_rsi_short_min_value,
                    momentum_use_trailing_value,
                    momentum_trailing_rr_value,
                    momentum_trailing_activate_value,
                )
        if df is None or df.empty:
            st.warning("Nenhuma operação encontrada para os parâmetros escolhidos.")
        else:
            total_trades = len(df)
            wins = df[df["pnl"] > 0]
            losses = df[df["pnl"] < 0]
            breakevens = df[df["pnl"].abs() <= 1e-8]
            eligible = len(wins) + len(losses)
            winrate = (len(wins) / eligible) * 100 if eligible else 0
            avg_win = wins["pnl"].mean() if not wins.empty else 0
            avg_loss = losses["pnl"].mean() if not losses.empty else 0
            total_pnl = df["pnl"].sum()

            st.subheader("Métricas")
            cards = [
                ("Trades", f"{total_trades}", f"{len(wins)} vit · {len(losses)} der · {len(breakevens)} be"),
                ("Winrate", f"{winrate:.1f}%", "com base em resultados líquidos" if eligible else "sem trades contabilizados"),
                ("PNL total", f"${total_pnl:.2f}", f"Capital inicial ${ctx.simulation_base_capital:.0f}"),
                ("Média Gain", f"${avg_win:.2f}", "por trade vencedor"),
                ("Média Loss", f"${avg_loss:.2f}", "por trade perdedor"),
            ]
            cards_html = "".join(
                f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='context'>{context}</div></div>"
                for label, value, context in cards
            )
            st.markdown(f"<div class='metric-grid'>{cards_html}</div>", unsafe_allow_html=True)
            caption_parts = [
                f"Ativos {symbol_display}" if multi_mode else f"Ativo {symbol_display}",
                f"Simulação: capital inicial ${ctx.simulation_base_capital:.0f}",
                f"risco {ctx.simulation_risk_per_trade*100:.0f}% por trade",
                f"estratégia {strategy_display}",
                f"bias {bias_label.get(trade_bias, trade_bias)}",
            ]
            if strategy == "ema_macd":
                caption_parts.append(f"cross lookback {cross_lookback} velas")
                caption_parts.append(f"divergência {'obrigatória' if require_divergence else 'opcional'}")
                caption_parts.append(f"queda mínima {divergence_min_drop_choice:.1f}%")
                if require_rsi_zone:
                    caption_parts.append(f"RSI zone long ≤ {rsi_long_max:.0f} / short ≥ {rsi_short_min:.0f}")
            caption_parts.extend([
                f"vitórias {len(wins)}",
                f"derrotas {len(losses)}",
                f"breakeven {len(breakevens)}",
            ])
            st.caption(" | ".join(caption_parts))

            if multi_mode:
                st.subheader("Resumo por ativo")
                st.markdown("<div class='card-section'>", unsafe_allow_html=True)
                per_symbol_rows = []
                for sym, subset in df.groupby("symbol", sort=False):
                    sym_wins = subset[subset["pnl"] > 0]
                    sym_losses = subset[subset["pnl"] < 0]
                    sym_breakevens = subset[subset["pnl"].abs() <= 1e-8]
                    sym_eligible = len(sym_wins) + len(sym_losses)
                    sym_winrate = (len(sym_wins) / sym_eligible) * 100 if sym_eligible else 0
                    per_symbol_rows.append({
                        "Ativo": sym,
                        "Trades": len(subset),
                        "Winrate %": round(sym_winrate, 2),
                        "PnL total": round(subset["pnl"].sum(), 2),
                        "Vitórias": len(sym_wins),
                        "Derrotas": len(sym_losses),
                        "Breakeven": len(sym_breakevens),
                        "Avg Win": round(sym_wins["pnl"].mean(), 2) if not sym_wins.empty else 0.0,
                        "Avg Loss": round(sym_losses["pnl"].mean(), 2) if not sym_losses.empty else 0.0,
                    })
                st.dataframe(pd.DataFrame(per_symbol_rows), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Evolução do PnL cumulativo")
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            if multi_mode:
                pnl_pivot = df.groupby(["timestamp", "symbol"], sort=True)["cum_pnl_symbol"].last().unstack()  # type: ignore[assignment]
                pnl_pivot = pnl_pivot.sort_index()
                pnl_pivot = pnl_pivot.ffill()
                st.line_chart(pnl_pivot, use_container_width=True)
            else:
                st.line_chart(df.set_index("timestamp")["cum_pnl"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Trades detalhados")
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if coverage:
                st.caption("Velas carregadas")
                st.markdown("<div class='card-section'>", unsafe_allow_html=True)
                cov_rows = []
                if multi_mode:
                    for sym, cov_map in coverage.items():
                        if isinstance(cov_map, dict):
                            for tf, cov in cov_map.items():
                                if isinstance(cov, dict):
                                    cov_rows.append({
                                        "Ativo": sym,
                                        "Timeframe": tf,
                                        "Candles": cov.get("candles"),
                                        "Inicio": cov.get("start"),
                                        "Fim": cov.get("end"),
                                        "Solicitado": cov.get("requested"),
                                    })
                                else:
                                    cov_rows.append({
                                        "Ativo": sym,
                                        "Timeframe": tf,
                                        "Candles": cov,
                                        "Inicio": None,
                                        "Fim": None,
                                        "Solicitado": None,
                                    })
                        else:
                            cov_rows.append({
                                "Ativo": sym,
                                "Timeframe": timeframe,
                                "Candles": cov_map,
                                "Inicio": None,
                                "Fim": None,
                                "Solicitado": None,
                            })
                else:
                    for tf, cov in coverage.items():
                        if isinstance(cov, dict):
                            cov_rows.append({
                                "Timeframe": tf,
                                "Candles": cov.get("candles"),
                                "Inicio": cov.get("start"),
                                "Fim": cov.get("end"),
                                "Solicitado": cov.get("requested"),
                            })
                        else:
                            cov_rows.append({
                                "Timeframe": tf,
                                "Candles": cov,
                                "Inicio": None,
                                "Fim": None,
                                "Solicitado": None,
                            })
                cov_df = pd.DataFrame(cov_rows)
                st.dataframe(cov_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            csv = df.to_csv(index=False).encode("utf-8")
            download_name = (
                f"backtest_bundle_{timeframe}.csv"
                if multi_mode
                else f"backtest_{symbol.replace('/', '_')}_{timeframe}.csv"
            )
            st.download_button("⬇️ Exportar CSV", csv, file_name=download_name, mime="text/csv")
    else:
        st.info("Defina os parâmetros e clique em 'Executar backtest' para ver os resultados.")

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

with realtime_tab:
    st.subheader("Estado do loop live")
    refresh_clicked = st.button("Atualizar dados agora")
    if refresh_clicked:
        st.success("Atualizado! (os dados acima são recarregados em cada clique)")
    runtime = load_json(RUNTIME_FILE)
    if runtime is None:
        st.warning("Ainda não há dados de runtime. Execute `main_loop()` (ex.: `.\\start.ps1 -Action live`) e volte a esta aba.")
    else:
        st.write(f"Última atualização: {runtime.get('last_update', '—')}")
        if runtime.get("open_positions"):
            st.markdown("**Posições abertas**")
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(runtime["open_positions"]), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Sem posições abertas no momento.")

        iteration = runtime.get("iteration") or []
        if iteration:
            st.markdown("**Resumo da última iteração**")
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(iteration), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Último backtest salvo pelo bot")
    bt = load_json(BACKTEST_FILE)
    if bt is None:
        st.info("Ainda não há backtests registrados pelo script principal.")
    else:
        summary = bt.get("summary", {})
        st.json(summary)
        sim = summary.get("simulation")
        if sim:
            cross = sim.get('ema_cross_lookback')
            cross_text = f" | cross lookback {cross} velas" if cross is not None else ""
            st.caption(f"Simulação salva: capital ${sim.get('base_capital', 0)} | risco {sim.get('risk_per_trade', 0)*100:.0f}%{cross_text}")
        coverage = summary.get("coverage") or {}
        if coverage:
            st.caption("Velas carregadas na última execução")
            cov_rows = []
            for tf, cov in coverage.items():
                if isinstance(cov, dict):
                    cov_rows.append({
                        "Timeframe": tf,
                        "Candles": cov.get("candles"),
                        "Inicio": cov.get("start"),
                        "Fim": cov.get("end"),
                        "Solicitado": cov.get("requested"),
                    })
                else:
                    cov_rows.append({
                        "Timeframe": tf,
                        "Candles": cov,
                        "Inicio": None,
                        "Fim": None,
                        "Solicitado": None,
                    })
            cov_df = pd.DataFrame(cov_rows)
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.dataframe(cov_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        trades = bt.get("trades") or []
        if trades:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
