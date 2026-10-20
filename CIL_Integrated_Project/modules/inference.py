"""
MachineGuard Inference Engine
Handles loading persisted artifacts, rigorous input validation,
consistent feature ordering, preprocessing, and prediction generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib

import config

class MachineGuardInferenceEngine:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (config.MODELS_DIR / "production_model.joblib")
        self.artifact: Optional[Dict[str, Any]] = None
        self.model = None
        self.preprocessor = None
        self.raw_feature_cols = []
        self.transformed_feature_cols = []
        self.operating_threshold = 0.35
        self.model_version = "1.0.0"
        self._load_artifact()

    def _load_artifact(self):
        if not self.model_path.exists():
            # If not yet trained, trigger training
            from modules import model_trainer
            self.artifact, _ = model_trainer.train_and_persist_pipeline()
        else:
            self.artifact = joblib.load(self.model_path)

        self.model = self.artifact["model"]
        self.preprocessor = self.artifact["preprocessor"]
        self.raw_feature_cols = self.artifact.get("feature_names_raw", config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES)
        self.transformed_feature_cols = self.artifact.get("feature_names_transformed", [])
        self.operating_threshold = self.artifact.get("operating_threshold", 0.35)
        self.model_version = self.artifact.get("model_version", "1.0.0")

    def validate_input(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Tuple[bool, Optional[str], pd.DataFrame]:
        """
        Validates input schema, datatypes, missing values, and physical sensor bounds.
        Returns (is_valid, error_message, validated_dataframe).
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            return False, "Input data must be a dictionary or a pandas DataFrame.", pd.DataFrame()

        # Check required columns
        missing_cols = [c for c in self.raw_feature_cols if c not in df.columns]
        if missing_cols:
            return False, f"Missing required input feature(s): {', '.join(missing_cols)}", pd.DataFrame()

        # Check for nulls in required columns
        if df[self.raw_feature_cols].isnull().any().any():
            null_cols = df[self.raw_feature_cols].columns[df[self.raw_feature_cols].isnull().any()].tolist()
            return False, f"Input contains missing (NaN) values in column(s): {', '.join(null_cols)}", pd.DataFrame()

        # Validate categorical 'Type'
        valid_types = {"L", "M", "H"}
        invalid_types = set(df["Type"].astype(str).str.upper()) - valid_types
        if invalid_types:
            return False, f"Invalid machine Type: {invalid_types}. Allowed values are 'L', 'M', or 'H'.", pd.DataFrame()
        df["Type"] = df["Type"].astype(str).str.upper()

        # Physical range checks (warn/reject absurd physical anomalies)
        # Air temp: 250K to 350K (~ -23C to 77C)
        # Process temp: 250K to 360K
        # Rot speed: 0 to 5000 rpm
        # Torque: 0 to 150 Nm
        # Tool wear: 0 to 500 min
        bounds = {
            "Air temperature [K]": (270.0, 330.0),
            "Process temperature [K]": (280.0, 340.0),
            "Rotational speed [rpm]": (500.0, 3500.0),
            "Torque [Nm]": (0.0, 100.0),
            "Tool wear [min]": (0.0, 350.0)
        }

        for col, (low, high) in bounds.items():
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except Exception:
                    return False, f"Column '{col}' must contain valid numeric readings.", pd.DataFrame()

                out_of_bounds = df[(df[col] < low) | (df[col] > high)]
                if not out_of_bounds.empty:
                    val = out_of_bounds[col].iloc[0]
                    return False, f"Sensor value out of plausible operating range for '{col}': {val}. Expected range: [{low}, {high}].", pd.DataFrame()

        # Ensure correct column ordering
        ordered_df = df[self.raw_feature_cols].copy()
        return True, None, ordered_df

    def predict_single(self, input_dict: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Runs validated inference on a single machine record."""
        is_valid, err, df = self.validate_input(input_dict)
        if not is_valid:
            raise ValueError(err)

        th = custom_threshold if custom_threshold is not None else self.operating_threshold
        proc = self.preprocessor.transform(df)
        proba = float(self.model.predict_proba(proc)[0, 1])
        prediction = 1 if proba >= th else 0

        # Calculate exact tree-path local contributions for explainability
        feature_contributions = self._compute_local_contributions(proc[0])

        return {
            "prediction": prediction,
            "failure_probability": proba,
            "operating_threshold": th,
            "model_name": "Random Forest Classifier",
            "model_version": self.model_version,
            "feature_contributions": feature_contributions,
            "input_features": df.iloc[0].to_dict(),
            "transformed_features": dict(zip(self.transformed_feature_cols, proc[0].tolist()))
        }

    def predict_batch(self, df: pd.DataFrame, custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """Runs validated batch inference on a DataFrame."""
        is_valid, err, validated_df = self.validate_input(df)
        if not is_valid:
            raise ValueError(err)

        th = custom_threshold if custom_threshold is not None else self.operating_threshold
        proc = self.preprocessor.transform(validated_df)
        probas = self.model.predict_proba(proc)[:, 1]
        preds = (probas >= th).astype(int)

        result_df = df.copy()
        result_df["Failure_Probability"] = np.round(probas, 4)
        result_df["Predicted_Failure"] = preds
        result_df["Risk_Level"] = [
            "CRITICAL" if p >= 0.70 else "ELEVATED" if p >= th else "NOMINAL"
            for p in probas
        ]
        return result_df

    def _compute_local_contributions(self, x_proc_row: np.ndarray) -> Dict[str, float]:
        """
        Calculates exact tree-based decision path local contributions.
        Accumulates delta in failure class probability along the decision path for each tree.
        """
        n_features = len(self.transformed_feature_cols)
        contribs = np.zeros(n_features)
        
        for tree in self.model.estimators_:
            indicator = tree.decision_path(x_proc_row.reshape(1, -1))
            nodes = indicator.indices
            node_values = tree.tree_.value[:, 0, :]
            # Probability of failure (class 1)
            probs = node_values[:, 1] / node_values.sum(axis=1)
            
            for i in range(len(nodes) - 1):
                curr_node = nodes[i]
                next_node = nodes[i + 1]
                feat = tree.tree_.feature[curr_node]
                if feat >= 0:
                    delta = probs[next_node] - probs[curr_node]
                    contribs[feat] += delta

        mean_contribs = contribs / len(self.model.estimators_)
        return dict(zip(self.transformed_feature_cols, [round(float(c), 4) for c in mean_contribs]))

# Singleton instance
_engine_instance = None

def get_inference_engine() -> MachineGuardInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MachineGuardInferenceEngine()
    return _engine_instance
