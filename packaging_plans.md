# Plano de Empacotamento e Gestão de Credenciais

## Objetivo
Documentar a estratégia para distribuir o FlexBot como executável único (Windows) mantendo uma base instalável via `pip`, ao mesmo tempo em que tratamos o armazenamento seguro das credenciais de API dos usuários finais.

---

## Etapa 1 — Organizar o Projeto como Pacote Python
- Criar `pyproject.toml`/`setup.cfg` definindo `flexbot` como pacote instalável.
- Expor entrypoints de console:
  - `flexbot-backtest` → chama `elliott_momentum_breakout_bot.py` (modo CLI).
  - `flexbot-dashboard` → roda `streamlit run` com o módulo adequado.
- Ajustar imports relativos/absolutos se necessário.
- Documentar instalação: `pip install flexbot` ou `pipx install flexbot`.
- Garantir que testes/backtests rodem a partir da instalação (não apenas com o repo clonado).

> Risco: baixo. Mantém a compatibilidade atual (script + Streamlit) e cria base para empacotamento futuro.

---

## Etapa 2 — Empacotar Executáveis com PyInstaller
- Criar arquivos `.spec` separados:
  - `flexbot_bot.spec` para o CLI (backtest/live loop).
  - `flexbot_dashboard.spec` para o dashboard Streamlit.
- Configurar PyInstaller para incluir dependências (`ta`, `ccxt`, etc.) e assets estáticos.
- Habilitar `--noconsole` no dashboard, manter console no CLI.
- Automatizar build com script (`build_exe.ps1`) que:
  1. Cria/usa uma venv limpa.
  2. Instala o pacote local (`pip install .`).
  3. Roda `pyinstaller <spec>` e produz executáveis em `dist/`.
- Opcional: empacotar os executáveis + arquivos auxiliares em `.zip` ou instalador (Inno Setup).

> Risco: médio. Precisamos validar dependências (Streamlit e PyInstaller podem gerar binários grandes). Testar execução em uma VM limpa.

---

## Etapa 3 — Coleta e Armazenamento Seguro de API Keys

### Requisitos
1. Primeiro uso deve solicitar `API_KEY`/`API_SECRET`.
2. Dados devem ser guardados localmente de forma cifrada (não plaintext).
3. Usuário deve poder atualizar chave ou trocar passphrase depois.
4. Caso esqueça a passphrase, deve existir opção de reset (perde as chaves, não o acesso ao app).

### Proposta Técnica
- Armazenar credenciais em `%APPDATA%\FlexBot\credentials.enc` (Windows) e `~/.flexbot/credentials.enc` (Linux/Mac).
- Usar **Fernet (cryptography)** para cifrar, derivando a key de uma passphrase via PBKDF2 + salt aleatório.
- Criar pequeno metadata JSON (`credentials.meta`) com:
  - Versão do formato.
  - Data da última atualização.
  - Informações públicas (ex.: exchange).
- Fluxos necessários:
  - `flexbot config --init`: pergunta passphrase + API keys → salva `credentials.enc`.
  - `flexbot config --edit`: pede passphrase atual, mostra campos para editar, regrava arquivo.
  - `flexbot config --change-passphrase`: descriptografa com passphrase antiga, cifra com nova.
  - `flexbot config --reset`: apaga arquivos cifrados para recomeçar.
- UI/Dashboard: adicionar modal (ou aba) que dispara mesmos comandos, reutilizando lógica do módulo de credenciais.

> Risco: moderado. Exige dependência extra (`cryptography`) e cuidado com UX (passphrase esquecida). Evita exposição de chaves em texto puro.

---

## Etapa 4 — Experiência do Usuário Final
- Primeira execução do executável:
  1. Verifica existência de `credentials.enc`.
  2. Se inexistente, abre wizard: passphrase, API key, API secret.
  3. Salva e prossegue para CLI/dashboard conforme o binário.
- Documentação rápida inclusa no instalador/README (print ou texto) explicando o processo e boas práticas (API com permissões limitadas, guarda local, etc.).
- Fornecer atalho para `flexbot config` via menu ou botão na dashboard para facilitar alterações posteriores.

---

## Próximos Passos
1. Adicionar estrutura de pacote (`pyproject.toml`) e console scripts.
2. Implementar módulo `flexbot.credentials` com fluxos descritos.
3. Criar scripts PyInstaller e validar binários em ambiente limpo.
4. Atualizar documentação (README) com instruções de instalação (`pip`, executável, gestão de credenciais).

---

## Observações
- Durante o desenvolvimento, manter `.flexbot_state` fora do binário; o instalador deve escrever em diretórios de usuário para evitar UAC.
- Testar PyInstaller em Windows 11 (usuário final alvo) e garantir que antivírus não sinalize falsos positivos (assinatura digital opcional).
- Considerar automação de releases (GitHub Actions) para gerar wheels + executáveis a cada tag.
