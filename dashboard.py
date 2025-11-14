import json
from pathlib import Path

import streamlit as st
import pandas as pd

from elliott_momentum_breakout_bot import (
    backtest_pair,
    pairs,
    timeframes,
    simulation_base_capital,
    simulation_risk_per_trade,
    fetch_account_balance,
    get_environment_mode,
    validate_symbol,
)
from state_store import read_user_config, update_user_config

st.set_page_config(page_title="FlexBot Dashboard", layout="wide")

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

STATE_DIR = Path('.flexbot_state')
RUNTIME_FILE = STATE_DIR / 'runtime.json'
BACKTEST_FILE = STATE_DIR / 'latest_backtest.json'

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

    symbol = st.selectbox("Par", options=pairs, index=0)
    timeframe = st.selectbox("Timeframe", options=timeframes, index=1)
    lookback_days = st.slider("Lookback (dias)", min_value=30, max_value=180, value=60, step=5)
    strategy = "ema_macd"
    strategy_display = "EMA + MACD"
    st.selectbox("Estratégia", options=[strategy_display], index=0, disabled=True)
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
    default_cross = config_data.get("ema_cross_lookback", 8)
    if not isinstance(default_cross, int) or default_cross < 2 or default_cross > 30:
        default_cross = 8
    cross_lookback = st.slider("Velas para cruzamento EMA/MACD", min_value=2, max_value=30, value=default_cross, step=1)
    require_default = config_data.get("ema_require_divergence")
    if not isinstance(require_default, bool):
        require_default = True
    require_divergence = st.checkbox(
        "Exigir divergência RSI (EMA+MACD)",
        value=require_default,
        help="Quando desmarcado, a confirmação EMA+MACD aceita sinais sem divergência RSI.",
    )
    if (
        config_data.get("trade_bias") != trade_bias
        or config_data.get("ema_cross_lookback") != cross_lookback
        or config_data.get("ema_require_divergence") != require_divergence
    ):
        update_user_config(
            trade_bias=trade_bias,
            ema_cross_lookback=cross_lookback,
            ema_require_divergence=require_divergence,
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
    st.caption("Para métricas em tempo real, mantenha `main_loop()` rodando (ex.: `start.ps1 -Action live`).")

divergence_badge = "On" if require_divergence else "Off"
summary_html = f"""
<div class='hero-card'>
    <div>
        <div class='hero-title'>{symbol}</div>
        <div class='hero-meta'>{timeframe} · Lookback {lookback_days} dias · Cross {cross_lookback} velas</div>
    </div>
    <div class='hero-badges'>
        <span class='status-pill {env_choice}'>{env_labels.get(env_choice, env_choice.title())}</span>
        <span class='status-pill slim'>Divergência {divergence_badge}</span>
        <span class='status-pill slim'>Bias {bias_label.get(trade_bias, trade_bias)}</span>
    </div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run_backtest(symbol: str, timeframe: str, lookback_days: int, strategy: str, bias: str, cross_lookback: int, require_divergence: bool):
    df_trades, coverage = backtest_pair(
        symbol,
        timeframe,
        lookback_days=lookback_days,
        strategy=strategy,
        bias=bias,
        cross_lookback=cross_lookback,
        require_divergence=require_divergence,
    )
    if df_trades is None or df_trades.empty:
        return None, coverage
    df_trades = df_trades.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_trades["timestamp"]):
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    df_trades["cum_pnl"] = df_trades["pnl"].cumsum()
    return df_trades, coverage

backtest_tab, realtime_tab = st.tabs(["Backtest", "Tempo real"])

with backtest_tab:
    if run_button:
        with st.spinner(
            f"Gerando backtest {symbol} {timeframe} (últimos {lookback_days} dias) — estratégia {strategy_display} | bias {bias_label.get(trade_bias, trade_bias)} | cross {cross_lookback} velas | divergência {'obrigatória' if require_divergence else 'opcional'}..."
        ):
            df, coverage = run_backtest(symbol, timeframe, lookback_days, strategy, trade_bias, cross_lookback, require_divergence)
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
                ("PNL total", f"${total_pnl:.2f}", f"Capital inicial ${simulation_base_capital:.0f}"),
                ("Média Gain", f"${avg_win:.2f}", "por trade vencedor"),
                ("Média Loss", f"${avg_loss:.2f}", "por trade perdedor"),
            ]
            cards_html = "".join(
                f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='context'>{context}</div></div>"
                for label, value, context in cards
            )
            st.markdown(f"<div class='metric-grid'>{cards_html}</div>", unsafe_allow_html=True)
            st.caption(
                f"Simulação: capital inicial ${simulation_base_capital:.0f} | risco {simulation_risk_per_trade*100:.0f}% por trade | estratégia {strategy_display} | bias {bias_label.get(trade_bias, trade_bias)} | cross lookback {cross_lookback} velas | divergência {'obrigatória' if require_divergence else 'opcional'} | vitórias {len(wins)} | derrotas {len(losses)} | breakeven {len(breakevens)}"
            )

            st.subheader("Evolução do PnL cumulativo")
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.line_chart(df.set_index("timestamp")["cum_pnl"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Trades detalhados")
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if coverage:
                st.caption("Velas carregadas por timeframe")
                st.markdown("<div class='card-section'>", unsafe_allow_html=True)
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
                st.dataframe(cov_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exportar CSV", csv, file_name=f"backtest_{symbol.replace('/', '_')}_{timeframe}.csv", mime="text/csv")
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
