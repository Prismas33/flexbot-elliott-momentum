# Plano de Afinação de Estratégias – FlexBot

Este documento resume recomendações para afinar as estratégias existentes em `elliott_momentum_breakout_bot.py` para aumentar a assertividade (taxa de acerto útil) e melhorar a gestão de risco, sem alterar a filosofia do sistema.

## 1. Estratégia Elliott 2→3

### Objetivo
Aproveitar movimentos impulsivos (onda 3) após correção saudável (onda 2) com confluência de RSI, volume e breakout.

### Situação atual (resumo)
- Detecção de onda 2→3 com:
  - Padrão de máximos/mínimos locais (0–1–2).
  - Correção entre `fib_min=0.38` e `fib_max=0.78` da onda 1.
  - Divergência RSI entre p0 e p2.
  - Breakout do topo da onda 1 (p1).
  - Spike de volume.
- `probability_score` = soma de:
  - divergência RSI
  - breakout
  - volume spike
  - entrada na zona Fib (0.5–0.618)
- Filtro multi-timeframe com `min_tf_agree`.
- RR mínimo (`min_rr_required = 3.0`).
- Entrada em dois estágios (partial + addon em retest).

### Recomendações de afinação

1. **Divergência RSI como bónus forte, não gate em todos os TFs**
   - Manter divergência obrigatória apenas no timeframe de entrada principal.
   - Nos timeframes de confirmação, usá-la como fator de score (bónus), não como filtro rígido.

2. **Refinar a correção da onda 2 em função do regime de tendência**
   - Em tendência forte (TF pai claramente `up`):
     - Apertar zona de correção (ex.: 0.45–0.68) para procurar pullbacks mais limpos.
   - Em tendência mais “flat”:
     - Admitir 0.33–0.8, mas exigir score mais alto (ver ponto seguinte).

3. **Score ponderado em vez de soma simples**
   - Usar pesos diferentes para cada condição, por exemplo:
     - Divergência RSI: 2 pontos
     - Breakout (fecho acima de p1): 2 pontos
     - Volume spike: 1 ponto
     - Onda 2 na zona Fib 0.5–0.618: 1 ponto
   - Exigir `score` mais alto no timeframe de referência (ex.: ≥4) e aceitar `score` mais baixo nos TFs secundários.

4. **Multi-timeframe com prioridade a TFs médios**
   - Dar maior importância a 15m/1h/4h do que a 5m.
   - Ao escolher o timeframe de referência para entrada, preferir TF intermédio (15m ou 1h) quando disponível.

5. **Stop-loss combinado: estrutura + ATR**
   - Em vez de apenas `wave2_bottom_price * 0.995`, considerar:
     - `stop = min(wave2_bottom * 0.995, entry_price - k * ATR)`.
   - Assim, evita-se um SL exageradamente distante quando a wave 2 está muito longe em termos de preço vs. volatilidade atual.

6. **TP dinâmico em função da volatilidade**
   - Manter a referência de 1.618 da onda 1 como alvo ideal.
   - Em períodos de baixa volatilidade ou tendência menos forte, permitir TP parcial mais conservador (1.382–1.5R) com possibilidade de trailing para o restante.

7. **Afinação do addon no retest**
   - Exigir, além do preço na zona de retest:
     - Candle de rejeição claro (pavio longo, corpo pequeno) na zona de antigo topo.
     - Volume do candle de retest ≥ média de volume de N períodos (ex.: 20).

---

## 2. Estratégia Momentum (EMA20/50, ATR e Volume)

### Objetivo
Capturar impulsos de continuação da tendência com confluência de EMAs, volatilidade crescente e volume.

### Situação atual (resumo)
- Filtro de divergência RSI (`has_bullish_rsi_divergence`).
- Confirmação `momentum_confirm`:
  - EMA20 > EMA50.
  - ATR atual > média dos últimos 5 ATR × 1.1.
  - Volume atual > média de volume × 1.2.
  - `score >= 2` para confirmar.
- Usa `momentum_min_tf_agree` para exigir múltiplos timeframes em confluência.
- SL e TP baseados em ATR:
  - Stop: `entry - ATR * momentum_stop_atr` (1.5 ATR).
  - TP: `entry + ATR * momentum_tp_atr` (4.5 ATR → ~3R).

### Recomendações de afinação

1. **Separar dois modos de momentum**
   - **Reversão/Breakout pós-divergência**:
     - Exigir divergência RSI + mudança recente nas EMAs.
   - **Continuação de tendência (trend following)**:
     - EMAs alinhadas (20 > 50) há X candles.
     - ATR alto e estável.
     - RSI não demasiado sobrecomprado (por exemplo, evitar entrar com RSI > 80).
     - Divergência pode ser opcional, usada como bónus de score.

2. **Aperfeiçoar o score de momentum**
   - Peso sugerido:
     - `ema_cross` (especialmente quando recente): 2 pontos.
     - `atr_expansion`: 1 ponto.
     - `vol_spike`: 1 ponto.
     - Divergência RSI: 1–2 pontos extra.
   - Exigir score mais elevado quando o bias em TF superior não é claramente `up`.

3. **Restringir entradas em extremos de RSI**
   - Evitar entradas long quando RSI está muito alto (ex.: >80), mesmo que o score seja alto.
   - Em trade de continuação, preferir que RSI esteja a recuar de uma zona de sobrecompra para um nível mais neutro (ex.: 70→55) antes de entrar.

4. **Refinar multi-timeframe**
   - Definir quais TFs são “core” para momentum (ex.: 15m e 1h).
   - Exigir que pelo menos um TF core esteja confirmado e que qualquer sinal em TF pequeno (5m) não contrarie o TF core.

5. **Gestão de risco específica para momentum**
   - Avaliar SL com ATR do timeframe imediatamente superior para evitar stops demasiado curtos em ruído.
   - Ex.: entrada em 15m, mas SL baseado no ATR do 1h (ou uma combinação).

6. **Parciais e break-even**
   - Realizar parte da posição (ex.: 50%) a ~1.5–2R.
   - Mover SL do restante para break-even após 1R ganho.

---

## 3. Estratégia OMDs

> Nota: OMDs era anteriormente chamada de estratégia EMA+MACD; a lógica permanece igual, apenas o nome exibido para o utilizador foi atualizado.

### Objetivo
Entrar em movimentos em que tendência (EMAs) e momentum (MACD) estão alinhados, com confirmação adicional de RSI e ATR.

### Situação atual (resumo)
- `ema_fast` (5) > `ema_slow` (21).
- MACD > signal, histograma > 0.
- ATR > 0 (para calcular stop/TP em múltiplos de ATR).
- Divergência RSI obrigatória antes de validar sinal.
- Stop e TP baseados em ATR (`ema_macd_stop_atr`, `ema_macd_tp_atr`).

### Recomendações de afinação

1. **Evitar entradas tardias em picos de MACD**
   - Entrar preferencialmente quando:
     - MACD cruza o signal de baixo para cima recentemente, ou
     - histograma passa de negativo para positivo com pouca distância percorrida.
   - Evitar entradas quando o histograma já está muito esticado (pico recente).

2. **Incluir o slope das EMAs**
   - Além de `ema_fast > ema_slow`, checar:
     - Slope positivo da `ema_fast` (comparar últimos 3–5 valores).
     - `ema_slow` pelo menos flat ou ligeiramente ascendente.

3. **RSI em zona saudável**
   - Para entradas long:
     - RSI preferencialmente entre 40 e 65.
     - Divergência RSI pode ser mais importante para reversões; para continuations, pode ser opcional.

4. **Stop baseado em estrutura + ATR**
   - Conjugação de mínimos locais importantes com múltiplos de ATR para definir SL.
   - Ex.: `stop = max(structural_low, entry - k*ATR)` com limitação para não exceder um certo % de preço.

5. **Ajuste dinâmico de parâmetros de ATR**
   - Testar combinações de `ema_macd_stop_atr` e `ema_macd_tp_atr` para encontrar equilibrio entre hit rate e payoff.
   - Exemplo:
     - Stop: 1.5–2 ATR, TP: 3–4.5 ATR.

---

## 4. Gestão de Risco e Portefólio

### Situação atual (resumo)
- `default_risk_per_trade = 0.01` (1% do capital por trade).
- Tamanho de posição calculado por diferença entre entrada e SL.
- RR mínimo global: `min_rr_required = 3.0`.
- Multi-par (`ETH/USDC`, `BTC/USDC`, `SOL/USDC`) e multi-timeframe.
- Backtester integrado com equity dinâmica.

### Recomendações de afinação

1. **Risco por trade dinâmico**
   - Ligar `risk_per_trade` à qualidade do sinal:
     - Setups com score baixo → 0.5%.
     - Setups com score máximo + confluência forte multi-TF → 1–1.5%.
   - Reduzir `risk_per_trade` em períodos de volatilidade extrema (ATR muito alto face à média histórica).

2. **Limites de exposição por ativo e global**
   - Definir:
     - `max_exposure_per_symbol` (risco agregado em % do capital num único par).
     - `max_exposure_total` (risco agregado total em todos os pares/estratégias).
   - Ex.: máximo 3% de risco em ETH, 5% total.

3. **Gestão de série de perdas (drawdown control)**
   - Monitorizar número de trades perdedores consecutivos.
   - Após N perdas seguidas (ex.: 3 ou 4):
     - Reduzir risco por trade pela metade até recuperar parte das perdas.

4. **Filtro de regime de mercado**
   - Usar `determine_trend_direction` para caracterizar regime:
     - "trend" (vários TFs `up`/`down`),
     - "range" (maioria `flat`).
   - Ajustar estratégias ao regime:
     - Elliott 2→3 e Momentum → mais ativos em regime de tendência.
    - OMDs com divergência → pode funcionar melhor em transições.

5. **Uso sistemático do backtester**
   - Explorar diferentes combinações de parâmetros:
     - `fib_min`, `fib_max`, `min_tf_agree`, `momentum_min_tf_agree`, `ema_macd_min_tf_agree`.
     - `momentum_stop_atr`, `momentum_tp_atr`, `ema_macd_stop_atr`, `ema_macd_tp_atr`.
     - `min_rr_required` em intervalos (2.0–3.5).
   - Avaliar não só o `winrate`, mas também:
     - P/L total,
     - drawdown máximo,
     - número de trades (amostra suficiente).

---

## 5. Plano de Experiências de Backtest

Esta secção serve como "caderno de laboratório" para registar experiências de backtest e a respetiva performance. A ideia é ter uma visão rápida do que foi testado e do que funcionou melhor.

### 5.1. Convenções Gerais

- Símbolo base inicial: `ETH/USDC:USDC`.
- Timeframe base inicial: `15m`.
- Período de backtest recomendado: 60–120 dias.
- Comando base (exemplo Elliott, 60 dias):

```powershell
python .\elliott_momentum_breakout_bot.py --symbol "ETH/USDC:USDC" --timeframe 15m --lookback-days 60 --strategy elliott
```

### 5.2. Tabela de Experiências – Elliott 2→3

| Experiência | Symbol          | TF   | min_rr_required | fib_min | fib_max | min_tf_agree | Trades | Winrate | P/L Total | Max DD | Observações |
|------------|-----------------|------|-----------------|---------|---------|--------------|--------|---------|-----------|--------|-------------|
| ELL-01     | ETH/USDC:USDC   | 15m  | 3.0             | 0.38    | 0.78    | 2            |        |         |           |        | baseline    |
| ELL-02     | ETH/USDC:USDC   | 15m  | 2.5             | 0.38    | 0.78    | 2            |        |         |           |        |             |
| ELL-03     | ETH/USDC:USDC   | 15m  | 2.0             | 0.38    | 0.78    | 2            |        |         |           |        |             |
| ELL-04     | ETH/USDC:USDC   | 15m  | 2.5             | 0.45    | 0.68    | 2            |        |         |           |        | tendência forte |

### 5.3. Tabela de Experiências – Momentum

| Experiência | Symbol          | TF   | min_rr_required | momentum_stop_atr | momentum_tp_atr | momentum_min_tf_agree | Trades | Winrate | P/L Total | Max DD | Observações |
|-------------|-----------------|------|-----------------|-------------------|-----------------|-----------------------|--------|---------|-----------|--------|-------------|
| MOM-01      | ETH/USDC:USDC   | 15m  | 3.0             | 1.5               | 4.5             | 2                     |        |         |           |        | baseline    |
| MOM-02      | ETH/USDC:USDC   | 15m  | 2.5             | 1.5               | 4.5             | 2                     |        |         |           |        |             |
| MOM-03      | ETH/USDC:USDC   | 15m  | 2.5             | 1.2               | 3.6             | 2                     |        |         |           |        | stop mais curto |
| MOM-04      | ETH/USDC:USDC   | 15m  | 2.5             | 1.8               | 5.4             | 2                     |        |         |           |        | alvo mais longo |

### 5.4. Tabela de Experiências – OMDs

| Experiência | Symbol          | TF   | min_rr_required | ema_macd_stop_atr | ema_macd_tp_atr | ema_macd_min_tf_agree | Trades | Winrate | P/L Total | Max DD | Observações |
|-------------|-----------------|------|-----------------|-------------------|-----------------|-----------------------|--------|---------|-----------|--------|-------------|
| EMA-01      | ETH/USDC:USDC   | 15m  | 3.0             | 1.8               | 5.4             | 1                     |        |         |           |        | baseline    |
| EMA-02      | ETH/USDC:USDC   | 15m  | 2.5             | 1.8               | 5.4             | 1                     |        |         |           |        |             |
| EMA-03      | ETH/USDC:USDC   | 15m  | 2.5             | 1.5               | 4.5             | 1                     |        |         |           |        | stop mais curto |
| EMA-04      | ETH/USDC:USDC   | 15m  | 2.5             | 2.0               | 6.0             | 1                     |        |         |           |        | alvo mais longo |

Sugestão de uso: após cada backtest, preencher as colunas `Trades`, `Winrate`, `P/L Total`, `Max DD` e um comentário rápido em `Observações`.

---

## 6. Próximos Passos Sugeridos

1. **Criar repositório Git e versionar o projeto FlexBot.**
2. **Executar as experiências ELL-01, MOM-01 e EMA-01** como baselines e registar resultados nas tabelas acima.
3. **Correr as restantes experiências (02–04)** por estratégia e comparar os resultados, escolhendo 1–2 configurações finais por estratégia.
4. **Implementar gradualmente as afinações no código**, começando pelas mudanças de parâmetros (fácil de reverter) e só depois mexendo na lógica interna, se necessário.

Este documento é um guia conceptual e operativo; a aplicação prática será feita passo a passo, testando cada alteração em backtest antes de usar em live.