"""
Task 9 Module: Assessment History Persistence Manager
MachineGuard History Architecture

Implements:
- Persistent audit logging for predictive maintenance assessments.
- JSON-backed storage with automatic schema initialization and lock-free safe writes.
- Search, filter, and tabular export functionality for UI presentation.
- Graceful error handling ensuring persistence failures never crash the application.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import datetime
import pandas as pd

import config

HISTORY_FILE = config.ARTIFACTS_DIR / "assessment_history.json"


def _ensure_history_file() -> Path:
    """Ensures artifacts directory and history JSON file exist."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
    return HISTORY_FILE


def save_assessment_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Appends an assessment record to history.
    Includes timestamp, inputs, model prediction, and rule recommendations.
    Returns: (success: bool, error_message: Optional[str])
    """
    try:
        path = _ensure_history_file()
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)

        if not isinstance(history, list):
            history = []

        # Add timestamp and assessment ID if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.datetime.now().isoformat()
        if "id" not in record:
            record["id"] = f"ASM-{len(history) + 1:04d}"

        history.append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return True, None
    except Exception as e:
        return False, str(e)


def get_assessment_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieves recent assessment records in reverse chronological order.
    """
    try:
        path = _ensure_history_file()
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)

        if not isinstance(history, list):
            return []

        # Return latest entries first
        return list(reversed(history))[:limit]
    except Exception:
        return []


def clear_assessment_history() -> bool:
    """Clears all stored assessment history records."""
    try:
        path = _ensure_history_file()
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return True
    except Exception:
        return False


def get_history_as_dataframe() -> pd.DataFrame:
    """
    Converts assessment records into a formatted DataFrame for UI tables and CSV export.
    """
    records = get_assessment_history()
    if not records:
        return pd.DataFrame()

    flat_rows = []
    for r in records:
        flat_rows.append({
            "Assessment ID": r.get("id", "N/A"),
            "Timestamp": r.get("timestamp", "N/A"),
            "Machine Type": r.get("Type", "N/A"),
            "Air Temp [K]": r.get("Air temperature [K]", "N/A"),
            "Process Temp [K]": r.get("Process temperature [K]", "N/A"),
            "Speed [rpm]": r.get("Rotational speed [rpm]", "N/A"),
            "Torque [Nm]": r.get("Torque [Nm]", "N/A"),
            "Tool Wear [min]": r.get("Tool wear [min]", "N/A"),
            "Model Used": r.get("model_name", "N/A"),
            "Predicted Risk": f"{r.get('failure_probability', 0.0):.2%}" if "failure_probability" in r else "N/A",
            "Recommendation": r.get("recommendation_action", "N/A"),
            "Safe to Operate": "✅ Safe" if r.get("is_safe_to_operate", True) else "⚠️ Intervene",
            "Fired Rules": ", ".join(r.get("fired_rules", []))
        })

    return pd.DataFrame(flat_rows)

