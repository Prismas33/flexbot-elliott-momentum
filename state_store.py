import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

STATE_DIR = Path(".flexbot_state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_FILE = STATE_DIR / "runtime.json"
BACKTEST_FILE = STATE_DIR / "latest_backtest.json"
CONFIG_FILE = STATE_DIR / "config.json"
CONTROL_FILE = STATE_DIR / "control.json"

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2))

def _json_default(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def write_runtime_state(open_positions: List[Dict[str, Any]], iteration_summary: List[Dict[str, Any]]) -> None:
    payload = {
        "last_update": datetime.utcnow().isoformat(),
        "open_positions": open_positions,
        "iteration": iteration_summary,
    }
    _write_json(RUNTIME_FILE, payload)

def write_backtest_summary(symbol: str, timeframe: str, lookback_days: int, df_trades, coverage=None, simulation=None) -> None:
    total_trades = len(df_trades) if df_trades is not None else 0
    wins = int(df_trades[df_trades['pnl'] > 0].shape[0]) if df_trades is not None else 0
    losses = int(df_trades[df_trades['pnl'] < 0].shape[0]) if df_trades is not None else 0
    breakevens = int(df_trades[abs(df_trades['pnl']) <= 1e-8].shape[0]) if df_trades is not None else 0
    eligible = wins + losses
    winrate = (wins / eligible) if eligible else 0
    total_pnl = float(df_trades['pnl'].sum()) if df_trades is not None and len(df_trades) else 0.0

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "winrate": winrate,
        "total_pnl": total_pnl,
        "coverage": coverage or {},
    }
    if simulation:
        summary["simulation"] = simulation
    _write_json(BACKTEST_FILE, {
        "summary": summary,
        "trades": df_trades.to_dict(orient="records") if df_trades is not None else [],
    })


def read_user_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}
    except OSError:
        return {}
    return {}


def write_user_config(config: Dict[str, Any]) -> None:
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2))


def update_user_config(**kwargs: Any) -> Dict[str, Any]:
    config = read_user_config()
    config.update(({k: v for k, v in kwargs.items() if v is not None}))
    write_user_config(config)
    return config


def read_control_state() -> Dict[str, Any]:
    if not CONTROL_FILE.exists():
        return {}
    try:
        data = json.loads(CONTROL_FILE.read_text())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}
    except OSError:
        return {}
    return {}


def write_control_state(state: Dict[str, Any]) -> None:
    if state:
        CONTROL_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    elif CONTROL_FILE.exists():
        CONTROL_FILE.unlink()
