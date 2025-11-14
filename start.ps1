param(
    [ValidateSet('backtest','live')]
    [string]$Action = 'backtest',
    [string]$Symbol = 'ETH/USDC:USDC',
    [string]$Timeframe = '15m',
    [int]$LookbackDays = 60,
    [ValidateSet('momentum','ema_macd')]
    [string]$Strategy = 'momentum',
    [ValidateSet('long','short','both')]
    [string]$TradeBias = 'long',
    [int]$CrossLookback = 8,
    [switch]$NoPaper
)

$pythonLauncher = "py -3.11"
try {
    & $pythonLauncher --version > $null 2>&1
} catch {
    Write-Warning "py -3.11 não encontrado. A instalar Python 3.11 é recomendado para evitar erros de build. A usar 'py -3' como fallback."
    $pythonLauncher = "py -3"
}

# Create venv if missing
if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Creating virtual environment .venv with $pythonLauncher..."
    Invoke-Expression "$pythonLauncher -m venv .venv"
}

# Activate the venv
Write-Host "Activating .venv..."
.\.venv\Scripts\Activate.ps1

# Install dependencies if missing
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip | Out-Null
Write-Host "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

# Build command
$cmd = "python .\elliott_momentum_breakout_bot.py"
if ($Action -eq 'live') {
    $cmd += " --live --symbol $Symbol"
} else {
    $cmd += " --symbol $Symbol --timeframe $Timeframe --lookback-days $LookbackDays"
}
$cmd += " --strategy $Strategy"
$cmd += " --trade-bias $TradeBias"
$cmd += " --cross-lookback $CrossLookback"
if ($NoPaper) {
    $cmd += " --no-paper"
}

Write-Host "Running: $cmd"
Invoke-Expression $cmd
