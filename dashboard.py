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
)

st.set_page_config(page_title="FlexBot Dashboard", layout="wide")
st.title("📈 FlexBot Dashboard")
st.caption("Visualize resultados de backtests e acompanhe métricas do loop live.")

STATE_DIR = Path('.flexbot_state')
RUNTIME_FILE = STATE_DIR / 'runtime.json'
BACKTEST_FILE = STATE_DIR / 'latest_backtest.json'

with st.sidebar:
    st.header("Parâmetros")
    symbol = st.selectbox("Par", options=pairs, index=0)
    timeframe = st.selectbox("Timeframe", options=timeframes, index=1)
    lookback_days = st.slider("Lookback (dias)", min_value=30, max_value=180, value=60, step=5)
    def strategy_label(key: str) -> str:
        mapping = {
            "momentum": "Momentum",
            "ema_macd": "EMA + MACD",
        }
        return mapping.get(key, key)

    strategy = st.selectbox("Estratégia", options=["momentum","ema_macd"], format_func=strategy_label, index=0)
    run_button = st.button("▶️ Executar backtest", use_container_width=True)
    st.markdown("---")
    st.caption("Para métricas em tempo real, mantenha `main_loop()` rodando (ex.: `start.ps1 -Action live`).")

@st.cache_data(show_spinner=False)
def run_backtest(symbol: str, timeframe: str, lookback_days: int, strategy: str):
    df_trades, coverage = backtest_pair(symbol, timeframe, lookback_days=lookback_days, strategy=strategy)
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
        with st.spinner(f"Gerando backtest {symbol} {timeframe} (últimos {lookback_days} dias) — estratégia {strategy}..."):
            df, coverage = run_backtest(symbol, timeframe, lookback_days, strategy)
        if df is None or df.empty:
            st.warning("Nenhuma operação encontrada para os parâmetros escolhidos.")
        else:
            total_trades = len(df)
            wins = df[df["outcome"] == "tp"]
            losses = df[df["outcome"] == "sl"]
            winrate = (len(wins) / total_trades) * 100 if total_trades else 0
            avg_win = wins["pnl"].mean() if not wins.empty else 0
            avg_loss = losses["pnl"].mean() if not losses.empty else 0
            total_pnl = df["pnl"].sum()

            st.subheader("Métricas")
            cols = st.columns(4)
            cols[0].metric("Trades", total_trades)
            cols[1].metric("Winrate", f"{winrate:.1f}%")
            cols[2].metric("Média Gain", f"${avg_win:.2f}")
            cols[3].metric("Média Loss", f"${avg_loss:.2f}")
            st.metric("PNL total", f"${total_pnl:.2f}")
            st.caption(f"Simulação: capital inicial ${simulation_base_capital:.0f} | risco {simulation_risk_per_trade*100:.0f}% por trade")

            st.subheader("Evolução do PnL cumulativo")
            st.line_chart(df.set_index("timestamp")["cum_pnl"], use_container_width=True)

            st.subheader("Trades detalhados")
            st.dataframe(df, use_container_width=True)

            if coverage:
                st.caption("Velas carregadas por timeframe")
                cov_df = pd.DataFrame([
                    {"Timeframe": tf, "Candles": cov}
                    for tf, cov in coverage.items()
                ])
                st.dataframe(cov_df, use_container_width=True)

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
            st.dataframe(pd.DataFrame(runtime["open_positions"]), use_container_width=True)
        else:
            st.info("Sem posições abertas no momento.")

        iteration = runtime.get("iteration") or []
        if iteration:
            st.markdown("**Resumo da última iteração**")
            st.dataframe(pd.DataFrame(iteration), use_container_width=True)

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
            st.caption(f"Simulação salva: capital ${sim.get('base_capital', 0)} | risco {sim.get('risk_per_trade', 0)*100:.0f}%")
        coverage = summary.get("coverage") or {}
        if coverage:
            st.caption("Velas carregadas na última execução")
            cov_df = pd.DataFrame([
                {"Timeframe": tf, "Candles": cov}
                for tf, cov in coverage.items()
            ])
            st.dataframe(cov_df, use_container_width=True)
        trades = bt.get("trades") or []
        if trades:
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
