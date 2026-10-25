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
                    data = json.load(f)
                    if isinstance(data, list):
                        self.records = data
                    else:
                        print("Warning: Assessment history file did not contain a valid list. Resetting.")
                        self.records = []
            except Exception as e:
                print(f"Warning: Corrupted assessment history file ({e}). Starting with empty history.")
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
        source: str = "Manual Input",
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Appends a new verified assessment record with timestamp, threshold, and model version."""
        th = float(inference_result.get("operating_threshold", config.DEFAULT_OPERATING_THRESHOLD))
        prob = round(float(inference_result.get("failure_probability", 0.0)), 4)
        pred = int(inference_result.get("prediction", 0))
        m_version = str(inference_result.get("model_version", "1.0.0"))
        m_id = str(model_id or inference_result.get("model_id", "random_forest"))
        m_name = str(inference_result.get("model_name", "Random Forest Classifier"))
        m_type = str(inference_result.get("model_type", "ML"))
        risk_tier = str(decision_result.get("risk_tier", "NOMINAL"))
        action = str(decision_result.get("primary_action", ""))
        m_machine_id = str(machine_id or f"EQ-{inputs.get('Type', 'M')}-001")
        expl_type = str(inference_result.get("explanation_type", "tree_attribution"))

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Guard against accidental double-clicks within the exact same second
        if self.records:
            last = self.records[0]
            if (
                last.get("timestamp") == now_str
                and last.get("machine_id") == m_machine_id
                and last.get("inputs") == inputs
                and last.get("failure_probability") == prob
                and last.get("model_id") == m_id
            ):
                return last

        record = {
            "assessment_id": f"MG-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(self.records)+1:03d}",
            "timestamp": now_str,
            "machine_id": m_machine_id,
            "source": source,
            "inputs": inputs,
            "failure_probability": prob,
            "operating_threshold": round(th, 4),
            "predicted_failure": pred,
            "risk_tier": risk_tier,
            "primary_action": action,
            "model_id": m_id,
            "model_name": m_name,
            "model_type": m_type,
            "model_version": m_version,
            "explanation_type": expl_type
        }
        self.records.insert(0, record)  # Newest first
        self._save()
        return record

    def get_history_dataframe(self, model_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Flattens records into a tabular pandas DataFrame for auditing and export.
        Optionally filters records by model display name or model identifier.
        Preserves backward compatibility with legacy records lacking model fields.
        """
        if not self.records:
            return pd.DataFrame()

        flat_rows = []
        for r in self.records:
            inputs = r.get("inputs", {})
            m_id = r.get("model_id")
            m_name = r.get("model_name")
            m_type = r.get("model_type")

            # Determine legacy model identity safely without fabricated claims
            if not m_name:
                m_name = "Legacy (Unknown)"
            if not m_type:
                m_type = "ML" if "random forest" in m_name.lower() else "Unknown"
            if not m_id:
                m_id = "legacy_unknown"

            # Filter if specified
            if model_filter and model_filter != "All Models":
                if model_filter != m_name and model_filter != m_id:
                    continue

            row = {
                "Assessment ID": r.get("assessment_id", ""),
                "Timestamp": r.get("timestamp", ""),
                "Machine ID": r.get("machine_id", ""),
                "Source": r.get("source", ""),
                "Model": m_name,
                "Model Type": m_type,
                "Model ID": m_id,
                "Type": inputs.get("Type", "N/A"),
                "Air Temp [K]": inputs.get("Air temperature [K]", "N/A"),
                "Process Temp [K]": inputs.get("Process temperature [K]", "N/A"),
                "Speed [rpm]": inputs.get("Rotational speed [rpm]", "N/A"),
                "Torque [Nm]": inputs.get("Torque [Nm]", "N/A"),
                "Tool Wear [min]": inputs.get("Tool wear [min]", "N/A"),
                "Failure Prob": r.get("failure_probability", 0.0),
                "Operating Threshold": r.get("operating_threshold", config.DEFAULT_OPERATING_THRESHOLD),
                "Predicted Failure": r.get("predicted_failure", 0),
                "Risk Tier": r.get("risk_tier", "NOMINAL"),
                "Action": r.get("primary_action", ""),
                "Model Version": r.get("model_version", "1.0.0")
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
