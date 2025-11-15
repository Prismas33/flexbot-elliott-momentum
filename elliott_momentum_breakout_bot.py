"""
elliott_momentum_breakout_bot.py

Sistema Momentum / EMA+MACD Breakout
- Multi-par (ETH/BTC/SOL) + Multi-timeframe (5m,15m,1h)
- SL/TP automáticos, trailing stop opcional, position sizing por risco
- Backtester simples para simular sinais e métricas

INSTRUÇÕES RÁPIDAS
1) pip install ccxt pandas numpy ta
2) Preenche API_KEY / API_SECRET
3) Mantém paper=True para testar
4) Para correr backtest: executar o ficheiro; por defeito faz backtest ETH 15m
5) Para correr em live: set paper=False e chama main_loop()

Nota: testa em paper e faz backtests antes de usar capital real.
"""

import sys
import argparse
import logging
from typing import List

import pandas as pd

from state_store import (
    write_backtest_summary,
    update_user_config,
)

from flexbot import context, state
from flexbot.backtest import backtest_pair
from flexbot.live_loop import main_loop


ctx = context
exchange_id = ctx.exchange_id
validate_symbol = state.validate_symbol
fetch_account_balance = state.fetch_account_balance
get_environment_mode = state.get_environment_mode

# Backtest and live loop logic now reside in flexbot.backtest and flexbot.live_loop.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Momentum breakout bot — backtest or live run')
    parser.add_argument('--live', dest='live', action='store_true', help='Run in live mode (main loop)')
    parser.add_argument('--symbol', dest='symbol', default='ETH/USDC:USDC', help='Symbol for backtest or live run')
    parser.add_argument('--timeframe', dest='timeframe', default='15m', help='Timeframe for the backtest (e.g., 15m)')
    parser.add_argument('--lookback-days', dest='lookback_days', type=int, default=60, help='Lookback days for backtest')
    paper_group = parser.add_mutually_exclusive_group()
    paper_group.add_argument('--paper-mode', dest='paper', action='store_const', const=True, help='Força modo paper (test) mesmo se o config estiver em live')
    paper_group.add_argument('--no-paper', dest='paper', action='store_const', const=False, help='Disable paper mode (will execute real orders)')
    parser.add_argument('--strategy', choices=['momentum','ema_macd'], default=ctx.strategy_mode, help='Seleciona estratégia principal')
    parser.add_argument('--cross-lookback', dest='cross_lookback', type=int, default=ctx.ema_macd_cross_lookback, help='Janela (nº de velas) para aceitar cruzamentos EMA/MACD recentes')
    parser.add_argument('--trade-bias', choices=['long','short','both'], default=ctx.trade_bias, help='Direção de trade: only long, only short ou ambos')
    parser.add_argument('--risk-percent', dest='risk_percent', type=float, help='Percentual do capital alocado a arriscar por trade (ex.: 1.0 para 1%).')
    parser.add_argument('--capital-base', dest='capital_base', type=float, help='Capital base máximo para cálculo de risco (padrão segue config).')
    parser.add_argument('--risk-mode', dest='risk_mode', choices=['standard','hunter'], help='Modo de risco: standard limita pelo capital base; hunter usa 100% do saldo.')
    parser.add_argument('--leverage', dest='leverage', type=float, help='Alavancagem alvo para limitar o tamanho máximo da posição.')
    parser.add_argument('--log-level', dest='log_level', default='INFO', help='Nível de logging (DEBUG, INFO, WARNING, ...)')
    divergence_group = parser.add_mutually_exclusive_group()
    divergence_group.add_argument('--require-divergence', dest='require_divergence', action='store_true', default=ctx.ema_macd_require_divergence, help='Exige divergência RSI para setups EMA+MACD (padrão)')
    divergence_group.add_argument('--allow-no-divergence', dest='require_divergence', action='store_false', help='Permite setups EMA+MACD mesmo sem divergência RSI')
    parser.add_argument('--check-symbol', dest='check_symbol', help='Valida se o par existe na corretora e termina imediatamente')
    parser.add_argument('--all-pairs', dest='all_pairs', action='store_true', help='Loop live cobre todos os pares configurados (ignora --symbol)')
    parser.set_defaults(paper=None)
    args = parser.parse_args()

    if args.check_symbol:
        symbol_to_check = args.check_symbol.strip()
        is_valid = validate_symbol(symbol_to_check)
        if is_valid:
            logging.info("Par %s disponível para negociação.", symbol_to_check)
            print(f"{symbol_to_check}: disponível")
            sys.exit(0)
        else:
            logging.error("Par %s não encontrado na corretora %s", symbol_to_check, exchange_id)
            print(f"{symbol_to_check}: indisponível")
            sys.exit(1)

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.info("Log level definido para %s", logging.getLevelName(log_level))

    logging.info("Script iniciado. ambiente configurado=%s (paper=%s)", ctx.environment_mode, ctx.paper)

    if args.paper is not None:
        ctx.update_environment_mode("paper" if args.paper else "live")
        update_user_config(environment=ctx.environment_mode)
        logging.info("Ambiente ajustado via CLI para %s", ctx.environment_mode)

    if ctx.paper:
        logging.info("MODO PAPER ATIVO — ordens reais NÃO serão colocadas.")
    else:
        logging.warning("MODO REAL: As ordens reais serão enviadas — esteja certo do seu API_KEY/API_SECRET e capital.")

    cli_overrides = {
        "strategy_mode": args.strategy,
        "trade_bias": args.trade_bias,
        "ema_cross_lookback": args.cross_lookback,
        "ema_require_divergence": args.require_divergence,
    }
    if args.risk_percent is not None:
        cli_overrides["risk_percent"] = max(0.0001, args.risk_percent / 100.0)
    if args.capital_base is not None:
        cli_overrides["capital_base"] = max(0.0, args.capital_base)
    if args.risk_mode is not None:
        cli_overrides["risk_mode"] = args.risk_mode
    if args.leverage is not None:
        cli_overrides["leverage"] = max(1.0, args.leverage)

    ctx.override_from_cli(cli_overrides)

    logging.info("Estratégia ativa: %s", ctx.strategy_mode)
    logging.info("Bias de trade ativo: %s", ctx.trade_bias)
    logging.info("EMA/MACD cross lookback: %d velas", ctx.ema_macd_cross_lookback)
    logging.info("EMA/MACD exige divergência RSI: %s", "sim" if ctx.ema_macd_require_divergence else "não")
    logging.info("Modo de risco: %s", ctx.risk_mode)
    logging.info("Risco por trade: %.2f%%", ctx.risk_percent * 100)
    logging.info("Capital base para sizing: %.2f", ctx.capital_base)
    logging.info("Alavancagem alvo: %.2fx", ctx.leverage)

    update_user_config(
        environment=ctx.environment_mode,
        trade_bias=ctx.trade_bias,
        ema_cross_lookback=ctx.ema_macd_cross_lookback,
        ema_require_divergence=ctx.ema_macd_require_divergence,
        symbol=args.symbol,
        timeframe=args.timeframe,
        risk_mode=ctx.risk_mode,
        risk_percent=ctx.risk_percent,
        capital_base=ctx.capital_base,
        leverage=ctx.leverage,
        active_pairs=list(ctx.active_pairs),
        multi_asset_enabled=getattr(ctx, "multi_asset_enabled", True),
    )

    if args.live:
        selected_symbols: List[str]
        if args.all_pairs:
            selected_symbols = list(ctx.active_pairs) if getattr(ctx, "active_pairs", None) else list(ctx.pairs)
        elif args.symbol:
            selected_symbols = [args.symbol]
        else:
            selected_symbols = list(ctx.active_pairs) if getattr(ctx, "active_pairs", None) else list(ctx.pairs)
        ctx.set_entry_timeframes([args.timeframe])
        logging.info(
            "Running live main loop for symbols=%s com timeframe base %s",
            selected_symbols,
            ctx.entry_timeframes,
        )
        # user should set API keys
        main_loop(symbols=selected_symbols)
    else:
        logging.info("Running backtest for %s %s (lookback %d days)", args.symbol, args.timeframe, args.lookback_days)
        try:
            df_trades, coverage_info = backtest_pair(
                args.symbol,
                args.timeframe,
                lookback_days=args.lookback_days,
                strategy=args.strategy,
                bias=ctx.trade_bias,
                cross_lookback=ctx.ema_macd_cross_lookback,
                require_divergence=ctx.ema_macd_require_divergence,
            )
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
                    "base_capital": ctx.simulation_base_capital,
                    "risk_per_trade": ctx.simulation_risk_per_trade,
                    "ema_cross_lookback": ctx.ema_macd_cross_lookback,
                    "ema_require_divergence": ctx.ema_macd_require_divergence,
                    "trade_bias": ctx.trade_bias,
                    "environment": ctx.environment_mode,
                }
            )
        except Exception as e:
            logging.error("Erro ao fazer backtest: %s", e)
