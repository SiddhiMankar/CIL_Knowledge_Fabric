"""
MachineGuard Assessment History & Persistence Manager
Maintains in-session and persistent logs of operational assessments,
enabling auditing, temporal analysis, and CSV export.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import pandas as pd

import config

HISTORY_FILE = config.ARTIFACTS_DIR / "assessment_history.json"

class AssessmentHistoryManager:
    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file or HISTORY_FILE
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []
        else:
            self.records = []

    def _save(self):
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save assessment history: {e}")

    def add_record(
        self,
        machine_id: str,
        inputs: Dict[str, Any],
        inference_result: Dict[str, Any],
        decision_result: Dict[str, Any],
        source: str = "Manual Input"
    ) -> Dict[str, Any]:
        """Appends a new assessment record with timestamp."""
        record = {
            "assessment_id": f"MG-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(self.records)+1:03d}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": machine_id or f"EQ-{inputs.get('Type', 'M')}-001",
            "source": source,
            "inputs": inputs,
            "failure_probability": round(inference_result.get("failure_probability", 0.0), 4),
            "predicted_failure": int(inference_result.get("prediction", 0)),
            "risk_tier": decision_result.get("risk_tier", "NOMINAL"),
            "primary_action": decision_result.get("primary_action", ""),
            "model_name": inference_result.get("model_name", "Random Forest")
        }
        self.records.insert(0, record)  # Newest first
        self._save()
        return record

    def get_history_dataframe(self) -> pd.DataFrame:
        """Flattens records into a tabular pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()

        flat_rows = []
        for r in self.records:
            row = {
                "Assessment ID": r["assessment_id"],
                "Timestamp": r["timestamp"],
                "Machine ID": r["machine_id"],
                "Source": r["source"],
                "Type": r["inputs"].get("Type", "N/A"),
                "Air Temp [K]": r["inputs"].get("Air temperature [K]", "N/A"),
                "Process Temp [K]": r["inputs"].get("Process temperature [K]", "N/A"),
                "Speed [rpm]": r["inputs"].get("Rotational speed [rpm]", "N/A"),
                "Torque [Nm]": r["inputs"].get("Torque [Nm]", "N/A"),
                "Tool Wear [min]": r["inputs"].get("Tool wear [min]", "N/A"),
                "Failure Prob": r["failure_probability"],
                "Risk Tier": r["risk_tier"],
                "Action": r["primary_action"]
            }
            flat_rows.append(row)
        return pd.DataFrame(flat_rows)

    def clear_history(self):
        """Clears stored history."""
        self.records = []
        if self.history_file.exists():
            try:
                self.history_file.unlink()
            except Exception:
                pass

_history_manager = None

def get_history_manager() -> AssessmentHistoryManager:
    global _history_manager
    if _history_manager is None:
        _history_manager = AssessmentHistoryManager()
    return _history_manager
