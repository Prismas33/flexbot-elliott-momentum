import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

STATE_DIR = Path(".flexbot_state")
STATE_DIR.mkdir(exist_ok=True)
RUNTIME_FILE = STATE_DIR / "runtime.json"
BACKTEST_FILE = STATE_DIR / "latest_backtest.json"

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
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "trades": len(df_trades) if df_trades is not None else 0,
        "winrate": float(df_trades[df_trades['outcome']=='tp'].shape[0]) / len(df_trades) if df_trades is not None and len(df_trades) else 0,
        "total_pnl": float(df_trades['pnl'].sum()) if df_trades is not None and len(df_trades) else 0,
        "coverage": coverage or {},
    }
    if simulation:
        summary["simulation"] = simulation
    _write_json(BACKTEST_FILE, {
        "summary": summary,
        "trades": df_trades.to_dict(orient="records") if df_trades is not None else [],
    })
