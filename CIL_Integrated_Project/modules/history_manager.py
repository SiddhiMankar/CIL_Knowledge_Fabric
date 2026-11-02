"""
Task 9 Module: Assessment History Persistence Manager
MachineGuard History Architecture

Implements:
- Persistent audit logging for predictive maintenance assessments.
- JSON-backed storage with automatic schema initialization and lock-free safe writes.
- Search, filter, and tabular export functionality for UI presentation.
- AssessmentHistoryManager class supporting automated tests and batch workflows.
- Graceful error handling ensuring persistence failures never crash the application.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
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
        history = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []

        if "timestamp" not in record:
            record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "id" not in record:
            record["id"] = f"ASM-{len(history) + 1:04d}"

        # Also populate AssessmentHistoryManager format compatibility fields if missing
        if "inputs" not in record:
            record["inputs"] = {
                "Type": record.get("Type", "M"),
                "Air temperature [K]": record.get("Air temperature [K]", 298.1),
                "Process temperature [K]": record.get("Process temperature [K]", 308.6),
                "Rotational speed [rpm]": record.get("Rotational speed [rpm]", 1500.0),
                "Torque [Nm]": record.get("Torque [Nm]", 40.0),
                "Tool wear [min]": record.get("Tool wear [min]", 0.0)
            }
        if "assessment_id" not in record:
            record["assessment_id"] = record["id"]

        history.append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        global _history_manager
        if _history_manager is not None:
            _history_manager.records = history

        return True, None
    except Exception as e:
        return False, str(e)


def get_assessment_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Reads stored assessment records in reverse chronological order."""
    try:
        path = _ensure_history_file()
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
            if not isinstance(records, list):
                return []
            records.reverse()
            if limit is not None:
                return records[:limit]
            return records
    except Exception:
        return []


def load_assessment_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Alias for get_assessment_history for backward compatibility."""
    return get_assessment_history(limit=limit)


def clear_assessment_history() -> bool:
    """Clears all stored assessment history records."""
    try:
        path = _ensure_history_file()
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        global _history_manager
        if _history_manager is not None:
            _history_manager.records = []
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
        inputs = r.get("inputs", {})
        flat_rows.append({
            "Assessment ID": r.get("id") or r.get("assessment_id", "N/A"),
            "Timestamp": r.get("timestamp", "N/A"),
            "Machine Type": r.get("Type") or inputs.get("Type", "N/A"),
            "Air Temp [K]": r.get("Air temperature [K]") if "Air temperature [K]" in r else inputs.get("Air temperature [K]", "N/A"),
            "Process Temp [K]": r.get("Process temperature [K]") if "Process temperature [K]" in r else inputs.get("Process temperature [K]", "N/A"),
            "Speed [rpm]": r.get("Rotational speed [rpm]") if "Rotational speed [rpm]" in r else inputs.get("Rotational speed [rpm]", "N/A"),
            "Torque [Nm]": r.get("Torque [Nm]") if "Torque [Nm]" in r else inputs.get("Torque [Nm]", "N/A"),
            "Tool Wear [min]": r.get("Tool wear [min]") if "Tool wear [min]" in r else inputs.get("Tool wear [min]", "N/A"),
            "Model Used": r.get("model_name", "N/A"),
            "Predicted Risk": f"{r.get('failure_probability', 0.0):.2%}" if "failure_probability" in r else "N/A",
            "Recommendation": r.get("recommendation_action") or r.get("primary_action", "N/A"),
            "Safe to Operate": "✅ Safe" if r.get("is_safe_to_operate", True) else "⚠️ Intervene",
            "Fired Rules": ", ".join(r.get("fired_rules", [])) if isinstance(r.get("fired_rules"), list) else ""
        })

    return pd.DataFrame(flat_rows)


def export_history_as_dataframe(records: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    """Exports given records or entire history to DataFrame."""
    if records is not None:
        flat_rows = []
        for r in records:
            inputs = r.get("inputs", {})
            flat_rows.append({
                "Assessment ID": r.get("id") or r.get("assessment_id", "N/A"),
                "Timestamp": r.get("timestamp", "N/A"),
                "Machine Type": r.get("Type") or inputs.get("Type", "N/A"),
                "Air Temp [K]": r.get("Air temperature [K]") if "Air temperature [K]" in r else inputs.get("Air temperature [K]", "N/A"),
                "Process Temp [K]": r.get("Process temperature [K]") if "Process temperature [K]" in r else inputs.get("Process temperature [K]", "N/A"),
                "Speed [rpm]": r.get("Rotational speed [rpm]") if "Rotational speed [rpm]" in r else inputs.get("Rotational speed [rpm]", "N/A"),
                "Torque [Nm]": r.get("Torque [Nm]") if "Torque [Nm]" in r else inputs.get("Torque [Nm]", "N/A"),
                "Tool Wear [min]": r.get("Tool wear [min]") if "Tool wear [min]" in r else inputs.get("Tool wear [min]", "N/A"),
                "Model Used": r.get("model_name", "N/A"),
                "Predicted Risk": f"{r.get('failure_probability', 0.0):.2%}" if "failure_probability" in r else "N/A",
                "Recommendation": r.get("recommendation_action") or r.get("primary_action", "N/A"),
                "Safe to Operate": "✅ Safe" if r.get("is_safe_to_operate", True) else "⚠️ Intervene",
                "Fired Rules": ", ".join(r.get("fired_rules", [])) if isinstance(r.get("fired_rules"), list) else ""
            })
        return pd.DataFrame(flat_rows)
    return get_history_as_dataframe()


# ============================================================================
# OBJECT-ORIENTED ASSESSMENT HISTORY MANAGER (MACHINEGUARD CORE)
# ============================================================================

class AssessmentHistoryManager:
    """
    Object-oriented history manager supporting filtering, multi-model tracking,
    and legacy record backward compatibility.
    """

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
                        self.records = []
            except Exception:
                self.records = []
        else:
            self.records = []

    def _save(self):
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        except Exception:
            pass

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
            "id": f"ASM-{len(self.records)+1:04d}",
            "timestamp": now_str,
            "machine_id": m_machine_id,
            "source": source,
            "inputs": inputs,
            "failure_probability": prob,
            "operating_threshold": round(th, 4),
            "predicted_failure": pred,
            "risk_tier": risk_tier,
            "primary_action": action,
            "recommendation_action": action,
            "model_id": m_id,
            "model_name": m_name,
            "model_type": m_type,
            "model_version": m_version,
            "explanation_type": expl_type,
            "is_safe_to_operate": (pred == 0 and risk_tier in ["NOMINAL", "MONITORING"])
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

            if not m_name:
                m_name = "Legacy (Unknown)"
            if not m_type:
                m_type = "ML" if "random forest" in m_name.lower() else "Unknown"
            if not m_id:
                m_id = "legacy_unknown"

            if model_filter and model_filter != "All Models":
                if model_filter != m_name and model_filter != m_id:
                    continue

            row = {
                "Assessment ID": r.get("assessment_id", r.get("id", "")),
                "Timestamp": r.get("timestamp", ""),
                "Machine ID": r.get("machine_id", ""),
                "Source": r.get("source", ""),
                "Model": m_name,
                "Model Type": m_type,
                "Model ID": m_id,
                "Type": inputs.get("Type") or r.get("Type", "N/A"),
                "Air Temp [K]": inputs.get("Air temperature [K]") if "Air temperature [K]" in inputs else r.get("Air temperature [K]", "N/A"),
                "Process Temp [K]": inputs.get("Process temperature [K]") if "Process temperature [K]" in inputs else r.get("Process temperature [K]", "N/A"),
                "Speed [rpm]": inputs.get("Rotational speed [rpm]") if "Rotational speed [rpm]" in inputs else r.get("Rotational speed [rpm]", "N/A"),
                "Torque [Nm]": inputs.get("Torque [Nm]") if "Torque [Nm]" in inputs else r.get("Torque [Nm]", "N/A"),
                "Tool Wear [min]": inputs.get("Tool wear [min]") if "Tool wear [min]" in inputs else r.get("Tool wear [min]", "N/A"),
                "Failure Prob": r.get("failure_probability", 0.0),
                "Operating Threshold": r.get("operating_threshold", config.DEFAULT_OPERATING_THRESHOLD),
                "Predicted Failure": r.get("predicted_failure", 0),
                "Risk Tier": r.get("risk_tier", "NOMINAL"),
                "Action": r.get("primary_action") or r.get("recommendation_action", ""),
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
    """Returns singleton AssessmentHistoryManager instance."""
    global _history_manager
    if _history_manager is None:
        _history_manager = AssessmentHistoryManager()
    return _history_manager
