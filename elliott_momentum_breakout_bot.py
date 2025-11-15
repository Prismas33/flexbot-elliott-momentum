"""
elliott_momentum_breakout_bot.py

Sistema Momentum / OMDs Breakout
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
from flexbot.backtest import backtest_pair, backtest_many
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
    parser.add_argument('--cross-lookback', dest='cross_lookback', type=int, default=ctx.ema_macd_cross_lookback, help='Janela (nº de velas) para aceitar cruzamentos OMDs recentes')
    parser.add_argument('--trade-bias', choices=['long','short','both'], default=ctx.trade_bias, help='Direção de trade: only long, only short ou ambos')
    parser.add_argument('--risk-percent', dest='risk_percent', type=float, help='Percentual do capital alocado a arriscar por trade (ex.: 1.0 para 1%).')
    parser.add_argument('--capital-base', dest='capital_base', type=float, help='Capital base máximo para cálculo de risco (padrão segue config).')
    parser.add_argument('--risk-mode', dest='risk_mode', choices=['standard','hunter'], help='Modo de risco: standard limita pelo capital base; hunter usa 100% do saldo.')
    parser.add_argument('--leverage', dest='leverage', type=float, help='Alavancagem alvo para limitar o tamanho máximo da posição.')
    parser.add_argument('--log-level', dest='log_level', default='INFO', help='Nível de logging (DEBUG, INFO, WARNING, ...)')
    divergence_group = parser.add_mutually_exclusive_group()
    divergence_group.add_argument('--require-divergence', dest='require_divergence', action='store_true', default=ctx.ema_macd_require_divergence, help='Exige divergência RSI para setups OMDs (padrão)')
    divergence_group.add_argument('--allow-no-divergence', dest='require_divergence', action='store_false', help='Permite setups OMDs mesmo sem divergência RSI')
    rsi_zone_group = parser.add_mutually_exclusive_group()
    rsi_zone_group.add_argument('--require-rsi-zone', dest='require_rsi_zone', action='store_true', help='Exige que o RSI esteja em zona de sobrevenda/sobrecompra configurada para validar OMDs.')
    rsi_zone_group.add_argument('--disable-rsi-zone', dest='require_rsi_zone', action='store_false', help='Não exige validação adicional de zona de RSI.')
    rsi_zone_group.set_defaults(require_rsi_zone=ctx.ema_require_rsi_zone)
    parser.add_argument('--rsi-zone-long-max', dest='rsi_zone_long_max', type=float, default=ctx.ema_rsi_zone_long_max, help='Máximo de RSI permitido para entradas long quando a validação de zona está ativa.')
    parser.add_argument('--rsi-zone-short-min', dest='rsi_zone_short_min', type=float, default=ctx.ema_rsi_zone_short_min, help='Mínimo de RSI permitido para entradas short quando a validação de zona está ativa.')
    trailing_group = parser.add_mutually_exclusive_group()
    trailing_group.add_argument('--ema-enable-trailing', dest='ema_use_trailing', action='store_true', help='Ativa trailing stop dinâmico após o trade atingir o RR configurado.')
    trailing_group.add_argument('--ema-disable-trailing', dest='ema_use_trailing', action='store_false', help='Desativa o trailing stop dinâmico para OMDs.')
    trailing_group.set_defaults(ema_use_trailing=ctx.ema_macd_use_trailing)
    parser.add_argument('--ema-trailing-rr', dest='ema_trailing_rr', type=float, default=ctx.ema_macd_trailing_rr, help='RR utilizado para calcular a distância do trailing stop (ex.: 1.0 move o SL para 1R abaixo do topo).')
    parser.add_argument('--ema-trailing-activate', dest='ema_trailing_activate', type=float, default=ctx.ema_macd_trailing_activate_rr, help='RR mínimo alcançado para começar a aplicar o trailing stop (ex.: 2.0 ativa após 2R).')
    parser.add_argument('--divergence-min-drop', dest='divergence_min_drop', type=float, default=ctx.divergence_min_drop_pct * 100, help='Variação percentual mínima entre fundos/topos consecutivos para considerar divergência RSI (ex.: 1.5 = 1.5%%).')
    momentum_divergence_group = parser.add_mutually_exclusive_group()
    momentum_divergence_group.add_argument('--momentum-require-divergence', dest='momentum_require_divergence', action='store_true', default=ctx.momentum_require_divergence, help='Exige divergência RSI para confirmar sinais da estratégia Momentum.')
    momentum_divergence_group.add_argument('--momentum-allow-no-divergence', dest='momentum_require_divergence', action='store_false', help='Permite sinais Momentum mesmo sem divergência RSI.')
    momentum_bonus_group = parser.add_mutually_exclusive_group()
    momentum_bonus_group.add_argument('--momentum-disable-divergence-bonus', dest='momentum_use_divergence_bonus', action='store_false', help='Não soma pontos extras quando há divergência RSI nos sinais Momentum.')
    momentum_bonus_group.add_argument('--momentum-enable-divergence-bonus', dest='momentum_use_divergence_bonus', action='store_true', help='Soma pontos extras quando há divergência RSI nos sinais Momentum.')
    momentum_bonus_group.set_defaults(momentum_use_divergence_bonus=ctx.momentum_use_divergence_bonus)
    parser.add_argument('--momentum-rsi-long-max', dest='momentum_rsi_long_max', type=float, default=ctx.momentum_rsi_long_max, help='RSI máximo permitido para entradas long na estratégia Momentum.')
    parser.add_argument('--momentum-rsi-short-min', dest='momentum_rsi_short_min', type=float, default=ctx.momentum_rsi_short_min, help='RSI mínimo permitido para entradas short na estratégia Momentum.')
    momentum_trailing_group = parser.add_mutually_exclusive_group()
    momentum_trailing_group.add_argument('--momentum-enable-trailing', dest='momentum_use_trailing', action='store_true', help='Ativa trailing stop dinâmico para Momentum.')
    momentum_trailing_group.add_argument('--momentum-disable-trailing', dest='momentum_use_trailing', action='store_false', help='Desativa o trailing stop dinâmico para Momentum.')
    momentum_trailing_group.set_defaults(momentum_use_trailing=ctx.momentum_use_trailing)
    parser.add_argument('--momentum-trailing-rr', dest='momentum_trailing_rr', type=float, default=ctx.momentum_trailing_rr, help='RR utilizado para calcular a distância do trailing Momentum (ex.: 1.0 mantém 1R).')
    parser.add_argument('--momentum-trailing-activate', dest='momentum_trailing_activate', type=float, default=ctx.momentum_trailing_activate_rr, help='RR mínimo alcançado para ativar o trailing Momentum (ex.: 2.0 ativa após 2R).')
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
        "ema_require_rsi_zone": args.require_rsi_zone,
        "ema_macd_use_trailing": args.ema_use_trailing,
        "momentum_require_divergence": args.momentum_require_divergence,
        "momentum_use_divergence_bonus": args.momentum_use_divergence_bonus,
        "momentum_use_trailing": args.momentum_use_trailing,
    }
    if args.risk_percent is not None:
        cli_overrides["risk_percent"] = max(0.0001, args.risk_percent / 100.0)
    if args.capital_base is not None:
        cli_overrides["capital_base"] = max(0.0, args.capital_base)
    if args.risk_mode is not None:
        cli_overrides["risk_mode"] = args.risk_mode
    if args.leverage is not None:
        cli_overrides["leverage"] = max(1.0, args.leverage)
    if args.divergence_min_drop is not None:
        cli_overrides["divergence_min_drop_pct"] = max(0.0, args.divergence_min_drop / 100.0)
    if args.rsi_zone_long_max is not None:
        cli_overrides["ema_rsi_zone_long_max"] = args.rsi_zone_long_max
    if args.rsi_zone_short_min is not None:
        cli_overrides["ema_rsi_zone_short_min"] = args.rsi_zone_short_min
    if args.momentum_rsi_long_max is not None:
        cli_overrides["momentum_rsi_long_max"] = args.momentum_rsi_long_max
    if args.momentum_rsi_short_min is not None:
        cli_overrides["momentum_rsi_short_min"] = args.momentum_rsi_short_min
    if args.ema_trailing_rr is not None:
        cli_overrides["ema_macd_trailing_rr"] = max(0.1, args.ema_trailing_rr)
    if args.ema_trailing_activate is not None:
        cli_overrides["ema_macd_trailing_activate_rr"] = max(0.0, args.ema_trailing_activate)
    if args.momentum_trailing_rr is not None:
        cli_overrides["momentum_trailing_rr"] = max(0.1, args.momentum_trailing_rr)
    if args.momentum_trailing_activate is not None:
        cli_overrides["momentum_trailing_activate_rr"] = max(0.0, args.momentum_trailing_activate)

    ctx.override_from_cli(cli_overrides)

    logging.info("Estratégia ativa: %s", ctx.strategy_mode)
    logging.info("Bias de trade ativo: %s", ctx.trade_bias)
    if ctx.strategy_mode == "ema_macd":
        logging.info("OMDs cross lookback: %d velas", ctx.ema_macd_cross_lookback)
        logging.info("OMDs exige divergência RSI: %s", "sim" if ctx.ema_macd_require_divergence else "não")
        logging.info(
            "OMDs exige RSI em zona: %s (long ≤ %.2f | short ≥ %.2f)",
            "sim" if ctx.ema_require_rsi_zone else "não",
            ctx.ema_rsi_zone_long_max,
            ctx.ema_rsi_zone_short_min,
        )
        logging.info(
            "Trailing dinâmico: %s · RR trailing=%.2f · ativa após %.2fR",
            "ativo" if ctx.ema_macd_use_trailing else "inativo",
            ctx.ema_macd_trailing_rr,
            ctx.ema_macd_trailing_activate_rr,
        )
    else:
        logging.info(
            "Momentum exige divergência RSI: %s · bônus ativo: %s",
            "sim" if ctx.momentum_require_divergence else "não",
            "sim" if ctx.momentum_use_divergence_bonus else "não",
        )
        logging.info(
            "Momentum limites RSI — long ≤ %.2f | short ≥ %.2f",
            ctx.momentum_rsi_long_max,
            ctx.momentum_rsi_short_min,
        )
        logging.info(
            "Trailing Momentum: %s · RR=%.2f · ativa após %.2fR",
            "ativo" if ctx.momentum_use_trailing else "inativo",
            ctx.momentum_trailing_rr,
            ctx.momentum_trailing_activate_rr,
        )
    logging.info("Modo de risco: %s", ctx.risk_mode)
    logging.info("Risco por trade: %.2f%%", ctx.risk_percent * 100)
    logging.info("Capital base para sizing: %.2f", ctx.capital_base)
    logging.info("Alavancagem alvo: %.2fx", ctx.leverage)
    logging.info(
        "Bias fixo 4h: %s",
        "ativo" if ctx.use_fixed_bias_timeframe else "desligado",
    )
    logging.info("Divergência RSI: queda mínima %.2f%% entre pivôs", ctx.divergence_min_drop_pct * 100)

    update_user_config(
        environment=ctx.environment_mode,
        trade_bias=ctx.trade_bias,
        ema_cross_lookback=ctx.ema_macd_cross_lookback,
        ema_require_divergence=ctx.ema_macd_require_divergence,
        divergence_min_drop_pct=ctx.divergence_min_drop_pct,
        ema_require_rsi_zone=ctx.ema_require_rsi_zone,
        ema_rsi_zone_long_max=ctx.ema_rsi_zone_long_max,
        ema_rsi_zone_short_min=ctx.ema_rsi_zone_short_min,
        use_fixed_bias_timeframe=ctx.use_fixed_bias_timeframe,
        fixed_bias_timeframe=ctx.fixed_bias_timeframe,
        momentum_require_divergence=ctx.momentum_require_divergence,
        momentum_use_divergence_bonus=ctx.momentum_use_divergence_bonus,
        momentum_rsi_long_max=ctx.momentum_rsi_long_max,
        momentum_rsi_short_min=ctx.momentum_rsi_short_min,
        ema_macd_use_trailing=ctx.ema_macd_use_trailing,
        ema_macd_trailing_rr=ctx.ema_macd_trailing_rr,
        ema_macd_trailing_activate_rr=ctx.ema_macd_trailing_activate_rr,
        momentum_use_trailing=ctx.momentum_use_trailing,
        momentum_trailing_rr=ctx.momentum_trailing_rr,
        momentum_trailing_activate_rr=ctx.momentum_trailing_activate_rr,
        symbol=args.symbol,
        timeframe=args.timeframe,
        risk_mode=ctx.risk_mode,
        risk_percent=ctx.risk_percent,
        capital_base=ctx.capital_base,
        leverage=ctx.leverage,
        active_pairs=list(ctx.active_pairs),
        multi_asset_enabled=getattr(ctx, "multi_asset_enabled", True),
        strategy_mode=ctx.strategy_mode,
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
        if args.all_pairs:
            symbols_for_bt = list(dict.fromkeys(
                list(ctx.active_pairs) if getattr(ctx, "active_pairs", None) else [args.symbol]
            ))
            if args.symbol not in symbols_for_bt:
                symbols_for_bt.insert(0, args.symbol)
        else:
            symbols_for_bt = [args.symbol]

        multi_symbol_bt = len(symbols_for_bt) > 1
        symbols_label = ", ".join(symbols_for_bt)
        logging.info(
            "Running backtest for %s @ %s (lookback %d days)",
            symbols_label,
            args.timeframe,
            args.lookback_days,
        )
        try:
            if multi_symbol_bt:
                df_trades, coverage_info = backtest_many(
                    symbols_for_bt,
                    args.timeframe,
                    lookback_days=args.lookback_days,
                    strategy=args.strategy,
                    bias=ctx.trade_bias,
                    cross_lookback=ctx.ema_macd_cross_lookback,
                    require_divergence=ctx.ema_macd_require_divergence,
                )
            else:
                df_trades, coverage_info = backtest_pair(
                    symbols_for_bt[0],
                    args.timeframe,
                    lookback_days=args.lookback_days,
                    strategy=args.strategy,
                    bias=ctx.trade_bias,
                    cross_lookback=ctx.ema_macd_cross_lookback,
                    require_divergence=ctx.ema_macd_require_divergence,
                )

            if df_trades is not None and not df_trades.empty:
                preview = df_trades.head()
                print(preview)
                csv_name = (
                    f"backtest_bundle_{args.timeframe}_trades.csv"
                    if multi_symbol_bt
                    else f"backtest_{symbols_for_bt[0].replace('/','_')}_{args.timeframe}_trades.csv"
                )
                df_trades.to_csv(csv_name, index=False)
                logging.info("Backtest guardado em CSV (%s)", csv_name)
            write_backtest_summary(
                symbols_for_bt if multi_symbol_bt else symbols_for_bt[0],
                args.timeframe,
                args.lookback_days,
                df_trades if df_trades is not None else pd.DataFrame(),
                coverage=coverage_info,
                simulation={
                    "base_capital": ctx.simulation_base_capital,
                    "risk_per_trade": ctx.simulation_risk_per_trade,
                    "ema_cross_lookback": ctx.ema_macd_cross_lookback,
                    "ema_require_divergence": ctx.ema_macd_require_divergence,
                    "divergence_min_drop_pct": ctx.divergence_min_drop_pct,
                    "ema_require_rsi_zone": ctx.ema_require_rsi_zone,
                    "ema_rsi_zone_long_max": ctx.ema_rsi_zone_long_max,
                    "ema_rsi_zone_short_min": ctx.ema_rsi_zone_short_min,
                    "momentum_require_divergence": ctx.momentum_require_divergence,
                    "momentum_use_divergence_bonus": ctx.momentum_use_divergence_bonus,
                    "momentum_rsi_long_max": ctx.momentum_rsi_long_max,
                    "momentum_rsi_short_min": ctx.momentum_rsi_short_min,
                    "use_fixed_bias_timeframe": ctx.use_fixed_bias_timeframe,
                    "fixed_bias_timeframe": ctx.fixed_bias_timeframe,
                    "ema_macd_use_trailing": ctx.ema_macd_use_trailing,
                    "ema_macd_trailing_rr": ctx.ema_macd_trailing_rr,
                    "ema_macd_trailing_activate_rr": ctx.ema_macd_trailing_activate_rr,
                    "momentum_use_trailing": ctx.momentum_use_trailing,
                    "momentum_trailing_rr": ctx.momentum_trailing_rr,
                    "momentum_trailing_activate_rr": ctx.momentum_trailing_activate_rr,
                    "trade_bias": ctx.trade_bias,
                    "strategy_mode": ctx.strategy_mode,
                    "environment": ctx.environment_mode,
                }
            )
        except Exception as e:
            logging.error("Erro ao fazer backtest: %s", e)
