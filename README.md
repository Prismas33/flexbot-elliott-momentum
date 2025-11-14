# FlexBot — Setup & Installation

This workspace contains `elliott_momentum_breakout_bot.py` which depends on `ccxt`, `pandas`, `numpy`, and `ta`.

## Recommended installation (Windows PowerShell)

1. Use Python **3.11** (recommended). ccxt currently pulls `coincurve==20.0.0`, which only ships wheels up to Python 3.11 — newer interpreters (3.12+/3.14) will fail compiling it unless you install Visual Studio Build Tools. Install/verify 3.11 with:

```powershell
py -0p                      # list python versions
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements --silent

py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt
```

2. Create and activate an isolated virtual environment (recommended):

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
.\.venv\Scripts\Activate.ps1
# then
pip install -r requirements.txt
```

3. If `pip` is not found, try `py -3 -m pip` or `python -m pip` and ensure Python is added to the PATH.

4. Troubleshooting quick commands:

```powershell
# Check Python
py -3 --version
python --version
# Check pip
py -3 -m pip --version
python -m pip --version
# If pip missing
py -3 -m ensurepip --upgrade
py -3 -m pip install --upgrade pip
```

If you want I can also create a `venv` for you or modify the script to print a friendly message when dependencies are missing.

## Quick start script (Windows)

I've added `start.ps1` to this repo to make it easier to run the bot and backtests in PowerShell.

Examples:

```powershell
# Backtest ETH 15m for 60 days (default)
.\start.ps1 -Action backtest -Symbol 'ETH/USDC:USDC' -Timeframe '15m' -LookbackDays 60

# Backtest usando somente momentum
.\start.ps1 -Action backtest -Symbol 'ETH/USDC:USDC' -Timeframe '15m' -LookbackDays 60 --strategy momentum

# Backtest com estratégia EMA+MACD (EMA 5/21 + MACD 26-55-9)
.\start.ps1 -Action backtest -Symbol 'ETH/USDC:USDC' -Timeframe '15m' -LookbackDays 60 --strategy ema_macd

# Backtest EMA+MACD apenas shorts, aceitando cruzamentos até 16 velas atrás
.\start.ps1 -Action backtest -Symbol 'ETH/USDC:USDC' -Timeframe '15m' -LookbackDays 60 --strategy ema_macd -TradeBias short -CrossLookback 16 -LogLevel DEBUG

# Backtestando apenas shorts com EMA+MACD
.\start.ps1 -Action backtest -Symbol 'ETH/USDC:USDC' -Timeframe '15m' -LookbackDays 60 --strategy ema_macd -TradeBias short

# Backtest EMA+MACD permitindo sinais sem divergência RSI
.\start.ps1 -Action backtest -Symbol 'ETH/USDC:USDC' -Timeframe '15m' -LookbackDays 60 --strategy ema_macd -AllowNoDivergence

# Run live paper trading loop (paper True by default)
.\start.ps1 -Action live -Symbol 'ETH/USDC:USDC'

# Run live real orders (⚠️ be careful!)
.\start.ps1 -Action live -Symbol 'ETH/USDC:USDC' -NoPaper

# Validar rapidamente se um par existe (CLI direto no bot)
python .\elliott_momentum_breakout_bot.py --check-symbol 'ETH/USDC:USDC'

# Validar par usando o script PowerShell
 .\start.ps1 -CheckSymbol 'ETH/USDC:USDC'

# Forçar modo paper mesmo que o config esteja em live
python .\elliott_momentum_breakout_bot.py --paper-mode --live --symbol 'ETH/USDC:USDC'
```

This script will:
- Create a `.venv` (using Python 3.11 if available) if missing
- Activate the virtual environment
- Install dependencies from `requirements.txt` if needed
- Run `elliott_momentum_breakout_bot.py` with the chosen options

## Dashboard (interface simples)

Instale as dependências (`python -m pip install -r requirements.txt`) e rode:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

- Escolha par/timeframe/lookback na barra lateral e clique em "Executar backtest".
- Use os seletores "Estratégia", "Direção" e "Velas para cruzamento EMA/MACD" para alternar entre **Momentum** ou **EMA + MACD (5/21 & 26-55-9)**, escolher o bias e definir quão recentes os cruzamentos precisam ser (por defeito 8 velas).
- O seletor "Ambiente" na barra lateral alterna entre **Teste (paper)** e **Produção (live)**, gravando a preferência em `.flexbot_state/config.json`. O script principal lê essa configuração automaticamente (pode ser sobrescrita com `--paper-mode` ou `--no-paper`).
- O botão "Consultar saldo" consulta o saldo estimado através da API (usa as credenciais configuradas) e a caixa "Validar par" verifica se o instrumento existe na corretora antes de rodar backtests/live.
- O checkbox "Exigir divergência RSI (EMA+MACD)" controla se a confirmação EMA+MACD só aceita setups com divergência; deixe marcado para o comportamento padrão ou desmarque para testar sinais sem esse filtro.
- Quando quiser debugar sinais (especialmente shorts), rode com `-LogLevel DEBUG` (ou `--log-level DEBUG` no script Python) para ver os detalhes de rejeição nas regras.
- A UI mostra métricas resumidas, gráfico do PnL cumulativo e uma tabela com cada trade para inspeção/exportação.
- Este backtest usa dados históricos até o momento da execução; não é streaming em tempo real.
- A aba **Tempo real** lê arquivos `.flexbot_state/runtime.json` e `.flexbot_state/latest_backtest.json` que o bot atualiza automaticamente. Para vê-la viva, deixe `main_loop()` rodando (por exemplo: `.\start.ps1 -Action live`) enquanto o dashboard está aberto e clique em "Atualizar dados agora" quando quiser sincronizar.
- Sempre que alterar par/timeframe/lookback, clique novamente em "Executar backtest" para recalcular os gráficos com os novos parâmetros (o Streamlit reroda o app a cada interação).
- Os backtests agora paginam os candles até ~5000 candles por timeframe. Se pedir janelas maiores (ex.: 5m por 180 dias) o sistema fará múltiplas requisições até preencher esse limite e mostrará na UI quantas velas conseguiu carregar.
- Para manter as simulações alinhadas com uma banca pequena, os backtests utilizam capital inicial fixo de **$100** e arriscam **10% ($10)** por trade ao dimensionar as posições.
- Todas as estratégias respeitam **RR mínimo de 3:1** (recompensa ≥ 3x o risco). Nas estratégias Momentum e EMA+MACD os multiplicadores de ATR foram calibrados (stop 1.5× / TP 4.5× e stop 1.8× / TP 5.4×) para que o alvo natural já seja ≥3R — se nem assim chegar em 3R, o trade é descartado.
- O sizing do backtest usa banca inicial de **$100** e recalcula a cada trade: sempre 10% do capital corrente é arriscado (ex.: se a curva estiver em $124, o risco passa a $12.40 e o alvo padrão fica $37.20).
- Por padrão, a estratégia EMA+MACD exige uma **divergência de RSI** alinhada com o lado do trade (bullish para longs, bearish para shorts). Agora é possível desativar esse filtro na CLI (`--allow-no-divergence`), no script PowerShell (`-AllowNoDivergence`) ou diretamente no dashboard (desmarcando o checkbox) para comparar resultados.

## Managing API keys

Store sensitive keys in a `.env` file (never commit it):

```dotenv
API_KEY=coloque_sua_api_key_aqui
API_SECRET=coloque_sua_api_secret_aqui
```

The bot automatically loads `.env` via `python-dotenv`; you can also export the same variables in the shell before running.
If you accidentally exposed keys (like in screenshots or chat), regenerate them immediately in the exchange dashboard.
