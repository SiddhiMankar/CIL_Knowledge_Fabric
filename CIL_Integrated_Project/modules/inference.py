"""
MachineGuard Inference Engine
Handles loading persisted artifacts, rigorous input validation,
consistent feature ordering, preprocessing, and prediction generation across
Machine Learning (Random Forest, Gradient Boosting, Logistic Regression) and
Deep Learning (1D CNN) models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib

import config

class MachineGuardInferenceEngine:
    def __init__(self, model_path: Optional[Path] = None, model_id: Optional[str] = None):
        self._explicit_model_path = model_path  # None means "use default + fallback"
        self.model_id = model_id or "random_forest"

        if model_path is not None:
            self.model_path = model_path
        else:
            # Resolve from model registry if known
            from modules.model_registry import MODEL_CATALOG
            if self.model_id in MODEL_CATALOG:
                self.model_path = MODEL_CATALOG[self.model_id]["artifact_path"]
            else:
                self.model_path = config.PRODUCTION_MODEL_PATH

        self.artifact: Optional[Dict[str, Any]] = None
        self.model = None
        self.preprocessor = None
        self.raw_feature_cols = config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES
        self.transformed_feature_cols: List[str] = []
        self.operating_threshold = config.DEFAULT_OPERATING_THRESHOLD
        self.model_version = "1.0.0"
        self.model_name = "Random Forest Classifier"
        self.model_type = "ML"
        self.explanation_type = "tree_attribution"
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self._load_artifact()

    def _load_artifact(self):
        """
        Loads the persisted model artifact.
        Strict requirement: Does NOT start silent retraining if model is missing.
        Fallback candidate-location search is only performed when no explicit
        model_path was provided by the caller (i.e. _explicit_model_path is None).
        """
        target_path = self.model_path
        if not target_path.exists() and self._explicit_model_path is None:
            # Only attempt candidate-location fallback for the default path
            found, located_path = config.locate_production_model()
            if found and located_path is not None:
                target_path = located_path
                self.model_path = target_path

        if not target_path.exists():
            self.is_loaded = False
            self.load_error = (
                f"Production model artifact not found at '{target_path}'. "
                "Silent retraining during inference is disabled to prevent non-deterministic delays. "
                "To train the production model, run: python -m modules.model_trainer"
            )
            return

        try:
            loaded = joblib.load(target_path)
            if not isinstance(loaded, dict) or "model" not in loaded or "preprocessor" not in loaded:
                self.is_loaded = False
                self.load_error = (
                    f"Corrupt or incompatible model artifact at '{target_path}'. "
                    "Artifact must be a dictionary containing 'model' and 'preprocessor' keys."
                )
                return

            self.artifact = loaded
            self.model = self.artifact["model"]
            self.preprocessor = self.artifact["preprocessor"]
            self.raw_feature_cols = self.artifact.get("feature_names_raw", config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES)
            self.transformed_feature_cols = self.artifact.get("feature_names_transformed", [])
            self.operating_threshold = float(self.artifact.get("operating_threshold", config.DEFAULT_OPERATING_THRESHOLD))
            self.model_version = str(self.artifact.get("model_version", "1.0.0"))
            self.model_id = str(self.artifact.get("model_id", self.model_id))
            self.model_name = str(self.artifact.get("model_name", "Random Forest Classifier"))
            self.model_type = str(self.artifact.get("model_type", "ML"))

            # Determine explanation type based on model architecture
            if self.model_type == "DL":
                self.explanation_type = "conv_activations"
            elif "logistic" in self.model_id.lower() or "logistic" in self.model_name.lower():
                self.explanation_type = "linear_coefficients"
            else:
                self.explanation_type = "tree_attribution"

            # Verify schema compatibility
            expected_features = config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES
            if set(self.raw_feature_cols) != set(expected_features):
                self.is_loaded = False
                self.load_error = (
                    f"Feature schema mismatch in model artifact '{target_path}'. "
                    f"Expected {expected_features}, found {self.raw_feature_cols}."
                )
                return

            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load production model artifact from '{target_path}': {str(e)}"

    def validate_input(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Tuple[bool, Optional[str], pd.DataFrame]:
        """
        Validates input schema, datatypes, missing values, infinite values, and physical sensor bounds.
        Returns (is_valid, error_message, validated_dataframe).
        """
        if isinstance(input_data, dict):
            if not input_data:
                return False, "Input dictionary is empty.", pd.DataFrame()
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            if input_data.empty:
                return False, "Input DataFrame is empty. Please provide at least one machine telemetry record.", pd.DataFrame()
            df = input_data.copy()
        else:
            return False, "Input data must be a dictionary or a pandas DataFrame.", pd.DataFrame()

        # Check required columns
        missing_cols = [c for c in self.raw_feature_cols if c not in df.columns]
        if missing_cols:
            return False, f"Missing required input feature(s): {', '.join(missing_cols)}", pd.DataFrame()

        # Check for nulls/NaN in required columns
        if df[self.raw_feature_cols].isnull().any().any():
            null_cols = df[self.raw_feature_cols].columns[df[self.raw_feature_cols].isnull().any()].tolist()
            return False, f"Input contains missing (NaN) values in column(s): {', '.join(null_cols)}", pd.DataFrame()

        # Validate categorical 'Type'
        valid_types = {"L", "M", "H"}
        invalid_types = set(df["Type"].astype(str).str.upper()) - valid_types
        if invalid_types:
            return False, f"Invalid machine Type: {invalid_types}. Allowed values are 'L', 'M', or 'H'.", pd.DataFrame()
        df["Type"] = df["Type"].astype(str).str.upper()

        # Physical range checks (warn/reject physical anomalies)
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

                # Check for infinite values (np.inf, -np.inf)
                if np.isinf(df[col]).any():
                    return False, f"Input contains infinite (inf or -inf) values in column '{col}'. Readings must be finite physical measurements.", pd.DataFrame()

                out_of_bounds = df[(df[col] < low) | (df[col] > high)]
                if not out_of_bounds.empty:
                    val = out_of_bounds[col].iloc[0]
                    return False, f"Sensor value out of plausible operating range for '{col}': {val}. Expected range: [{low}, {high}].", pd.DataFrame()

        # Ensure correct column ordering
        ordered_df = df[self.raw_feature_cols].copy()
        return True, None, ordered_df

    def predict_single(self, input_dict: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Runs validated inference on a single machine record using the configured model."""
        if not self.is_loaded:
            detail = self.load_error or "Please train the model first."
            raise RuntimeError(f"Production model is not loaded. {detail}")

        if custom_threshold is not None:
            if not (0.0 <= custom_threshold <= 1.0) or np.isnan(custom_threshold):
                raise ValueError(f"Custom threshold must be between 0.0 and 1.0. Got: {custom_threshold}")

        is_valid, err, df = self.validate_input(input_dict)
        if not is_valid:
            raise ValueError(err)

        th = custom_threshold if custom_threshold is not None else self.operating_threshold
        proc = self.preprocessor.transform(df)

        feature_contributions: Optional[Dict[str, float]] = None
        conv_activations: Optional[Dict[str, np.ndarray]] = None

        if self.model_type == "DL":
            # Deep Learning / PyTorch 1D CNN Inference
            import torch
            X_tensor = torch.tensor(proc, dtype=torch.float32).unsqueeze(-1)
            self.model.eval()
            with torch.no_grad():
                proba = float(self.model(X_tensor).item())
            prediction = 1 if proba >= th else 0

            # Extract convolutional activations
            c1, c2 = self.model.extract_conv_activations(X_tensor)
            conv_activations = {"conv1d_1": c1, "conv1d_2": c2}
            # Tree attribution is not applicable to CNNs
            feature_contributions = {}
        else:
            # Classical Machine Learning Inference (Random Forest, Gradient Boosting, Logistic Regression)
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(proc)[0, 1])
            else:
                proba = float(self.model.predict(proc)[0])
            prediction = 1 if proba >= th else 0

            # Compute model-specific feature attribution
            if "random forest" in self.model_name.lower():
                feature_contributions = self._compute_rf_contributions(proc[0])
            elif "gradient boosting" in self.model_name.lower():
                feature_contributions = self._compute_gb_contributions(proc[0])
            elif "logistic" in self.model_name.lower():
                feature_contributions = self._compute_lr_contributions(proc[0])
            else:
                feature_contributions = self._compute_generic_contributions(proc[0])

        return {
            "prediction": prediction,
            "failure_probability": proba,
            "operating_threshold": th,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_contributions": feature_contributions or {},
            "conv_activations": conv_activations,
            "explanation_type": self.explanation_type,
            "input_features": df.iloc[0].to_dict(),
            "transformed_features": dict(zip(self.transformed_feature_cols, proc[0].tolist()))
        }

    def predict_batch(self, df: pd.DataFrame, custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """Runs validated batch inference on a DataFrame."""
        if not self.is_loaded:
            detail = self.load_error or "Please train the model first."
            raise RuntimeError(f"Production model is not loaded. {detail}")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input DataFrame is empty. Please provide at least one machine telemetry record.")

        if custom_threshold is not None:
            if not (0.0 <= custom_threshold <= 1.0) or np.isnan(custom_threshold):
                raise ValueError(f"Custom threshold must be between 0.0 and 1.0. Got: {custom_threshold}")

        is_valid, err, validated_df = self.validate_input(df)
        if not is_valid:
            raise ValueError(err)

        th = custom_threshold if custom_threshold is not None else self.operating_threshold
        proc = self.preprocessor.transform(validated_df)

        if self.model_type == "DL":
            import torch
            X_tensor = torch.tensor(proc, dtype=torch.float32).unsqueeze(-1)
            self.model.eval()
            with torch.no_grad():
                probas = self.model(X_tensor).squeeze(-1).numpy().ravel()
        else:
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(proc)[:, 1]
            else:
                probas = self.model.predict(proc).astype(float)

        preds = (probas >= th).astype(int)

        from modules.decision_layer import classify_risk_tier

        result_df = df.copy()
        result_df["Failure_Probability"] = np.round(probas, 4)
        result_df["Predicted_Failure"] = preds
        result_df["Risk_Level"] = [
            classify_risk_tier(p, th)
            for p in probas
        ]
        result_df["Selected_Model"] = self.model_name
        result_df["Model_Type"] = self.model_type
        result_df["Decision_Threshold"] = th
        result_df["Prediction_Status"] = "SUCCESS"
        return result_df

    def predict_batch_robust(self, df: pd.DataFrame, custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Batch inference with row-level fault tolerance.
        Valid rows receive predictions; invalid rows are marked with error status without crashing the batch.
        """
        if not self.is_loaded:
            detail = self.load_error or "Please train the model first."
            raise RuntimeError(f"Production model is not loaded. {detail}")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input DataFrame is empty. Please provide at least one machine telemetry record.")

        th = custom_threshold if custom_threshold is not None else self.operating_threshold
        from modules.decision_layer import classify_risk_tier

        result_df = df.copy()
        probas: List[Optional[float]] = []
        preds: List[Optional[int]] = []
        risk_levels: List[str] = []
        statuses: List[str] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            is_valid, err, val_df = self.validate_input(row_dict)
            if not is_valid:
                probas.append(None)
                preds.append(None)
                risk_levels.append("INVALID")
                statuses.append(f"Validation Error: {err}")
                continue

            try:
                proc = self.preprocessor.transform(val_df)
                if self.model_type == "DL":
                    import torch
                    X_tensor = torch.tensor(proc, dtype=torch.float32).unsqueeze(-1)
                    self.model.eval()
                    with torch.no_grad():
                        p = float(self.model(X_tensor).item())
                else:
                    p = float(self.model.predict_proba(proc)[0, 1])

                pred = 1 if p >= th else 0
                tier = classify_risk_tier(p, th)

                probas.append(round(p, 4))
                preds.append(pred)
                risk_levels.append(tier)
                statuses.append("SUCCESS")
            except Exception as e:
                probas.append(None)
                preds.append(None)
                risk_levels.append("ERROR")
                statuses.append(f"Inference Error: {str(e)}")

        result_df["Failure_Probability"] = probas
        result_df["Predicted_Failure"] = preds
        result_df["Risk_Level"] = risk_levels
        result_df["Selected_Model"] = self.model_name
        result_df["Model_Type"] = self.model_type
        result_df["Decision_Threshold"] = th
        result_df["Prediction_Status"] = statuses
        return result_df

    def _compute_rf_contributions(self, x_proc_row: np.ndarray) -> Dict[str, float]:
        """Calculates exact tree-based decision path local contributions for Random Forest."""
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

    def _compute_gb_contributions(self, x_proc_row: np.ndarray) -> Dict[str, float]:
        """Calculates exact stagewise tree decision path contributions for Gradient Boosting."""
        n_features = len(self.transformed_feature_cols)
        contribs = np.zeros(n_features)
        lr = float(getattr(self.model, "learning_rate", 0.1))

        for stage in self.model.estimators_:
            tree = stage[0]
            indicator = tree.decision_path(x_proc_row.reshape(1, -1))
            nodes = indicator.indices
            node_values = tree.tree_.value[:, 0, 0]

            for i in range(len(nodes) - 1):
                curr_node = nodes[i]
                next_node = nodes[i + 1]
                feat = tree.tree_.feature[curr_node]
                if feat >= 0:
                    delta = node_values[next_node] - node_values[curr_node]
                    contribs[feat] += lr * delta

        return dict(zip(self.transformed_feature_cols, [round(float(c), 4) for c in contribs]))

    def _compute_lr_contributions(self, x_proc_row: np.ndarray) -> Dict[str, float]:
        """Calculates exact linear log-odds feature contributions for Logistic Regression."""
        coefs = self.model.coef_[0]
        contribs = coefs * x_proc_row
        return dict(zip(self.transformed_feature_cols, [round(float(c), 4) for c in contribs]))

    def _compute_generic_contributions(self, x_proc_row: np.ndarray) -> Dict[str, float]:
        """Fallback feature importance attribution for general tree models."""
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.transformed_feature_cols, [round(float(v), 4) for v in self.model.feature_importances_]))
        return {feat: 0.0 for feat in self.transformed_feature_cols}


# Cache of instantiated inference engines by (model_id, model_path_str)
_engine_cache: Dict[Tuple[str, Optional[str]], MachineGuardInferenceEngine] = {}


def get_inference_engine(
    model_id: str = "random_forest",
    model_path: Optional[Path] = None,
    force_reload: bool = False
) -> MachineGuardInferenceEngine:
    """
    Returns cached inference engine for a given model ID or explicit artifact path.
    Default invocation (no args) returns the primary production Random Forest engine.
    """
    global _engine_cache
    path_key = str(model_path) if model_path is not None else None
    cache_key = (model_id, path_key)

    if force_reload or cache_key not in _engine_cache:
        _engine_cache[cache_key] = MachineGuardInferenceEngine(model_path=model_path, model_id=model_id)

    return _engine_cache[cache_key]
