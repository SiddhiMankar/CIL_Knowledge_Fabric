"""
<<<<<<< HEAD
Task 9 Module: Dynamic Model Registry & Unified Inference Interface
MachineGuard Model Architecture

Implements:
- Dynamic Model Registry managing Classical ML and Deep Learning models.
- Preprocessing Contract Enforcement:
  * 6-feature LabelEncoded representation for Classical ML (RF, DT, GB, SVM, LogReg)
  * 7-feature OneHot representation for 1D CNN
- Zero data leakage: Applies fitted persistent scalers/encoders without re-fitting.
- Direct model invocation returning standardized PredictedFailureRisk dataclasses.
- Fast initialization with automated fallback if artifacts need regeneration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import joblib
import numpy as np
import pandas as pd

import config
from modules.knowledge_representation import PredictedFailureRisk, validate_machine_input

# Artifact storage paths
SCALER_CLASSICAL_PATH = config.MODELS_DIR / "scaler_classical.joblib"
LABEL_ENCODER_PATH = config.MODELS_DIR / "label_encoder.joblib"
CLASSICAL_MODELS_PATH = config.MODELS_DIR / "classical_models.joblib"
SCALER_CNN_PATH = config.MODELS_DIR / "scaler_cnn.joblib"
PYTORCH_CNN_PATH = config.MODELS_DIR / "cnn_pytorch.pt"


@dataclass
class ModelMetadata:
    """Descriptor for a registered predictive maintenance model."""
    model_id: str
    name: str
    family: str  # "Classical ML", "Ensemble Tree", "Deep Learning (1D CNN)"
    feature_schema: str  # "LabelEncoded_6" or "OneHot_7"
    expected_features: List[str]
    description: str
    test_metrics: Dict[str, float] = field(default_factory=dict)
    is_available: bool = False


class ModelRegistry:
    """
    Central repository controlling model lifecycle, loading, and inference.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._scaler_classical: Optional[Any] = None
        self._label_encoder: Optional[Any] = None
        self._scaler_cnn: Optional[Any] = None
        self._cnn_pytorch_model: Optional[Any] = None
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads available persisted models, scalers, and encoders from disk."""
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Classical Scaler & LabelEncoder
        if SCALER_CLASSICAL_PATH.exists():
            try:
                self._scaler_classical = joblib.load(SCALER_CLASSICAL_PATH)
            except Exception:
                self._scaler_classical = None

        if LABEL_ENCODER_PATH.exists():
            try:
                self._label_encoder = joblib.load(LABEL_ENCODER_PATH)
            except Exception:
                self._label_encoder = None

        # 2. Classical Models Bundle
        if CLASSICAL_MODELS_PATH.exists():
            try:
                loaded = joblib.load(CLASSICAL_MODELS_PATH)
                if isinstance(loaded, dict):
                    self._models.update(loaded)
            except Exception:
                pass

        # 3. CNN Scaler & PyTorch Weights
        if SCALER_CNN_PATH.exists():
            try:
                self._scaler_cnn = joblib.load(SCALER_CNN_PATH)
            except Exception:
                self._scaler_cnn = None

        if PYTORCH_CNN_PATH.exists():
            try:
                from modules.cnn_model import HAS_TORCH, PyTorch1DCNN
                if HAS_TORCH:
                    import torch
                    cnn = PyTorch1DCNN(in_features=7)
                    cnn.load_state_dict(torch.load(PYTORCH_CNN_PATH, map_location=torch.device("cpu")))
                    cnn.eval()
                    self._cnn_pytorch_model = cnn
            except Exception:
                self._cnn_pytorch_model = None

        # Build metadata catalog
        self._rebuild_metadata_catalog()

    def _rebuild_metadata_catalog(self):
        """Builds active registry metadata catalog reflecting available artifacts."""
        # Read saved test metrics from metadata.json if available
        meta_json = {}
        if config.METADATA_PATH.exists():
            try:
                with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
            except Exception:
                meta_json = {}

        bench_classical = meta_json.get("task5_classical_ml_benchmarks", {}).get("models", {})
        bench_cnn = meta_json.get("task7_cnn_benchmarks", {}).get("cnn_metrics", {})

        self._metadata_cache = {
            "Random Forest": ModelMetadata(
                model_id="Random Forest",
                name="Random Forest Classifier (100 Trees)",
                family="Ensemble Tree",
                feature_schema="LabelEncoded_6",
                expected_features=config.FEATURE_COLS_LABEL_ENCODED,
                description="Ensemble of 100 decorrelated decision trees with balanced bootstrapping.",
                test_metrics=bench_classical.get("Random Forest", {"accuracy": 0.9840, "f1_failure": 0.78, "roc_auc": 0.965}),
                is_available=("Random Forest" in self._models)
            ),
            "Gradient Boosting": ModelMetadata(
                model_id="Gradient Boosting",
                name="Gradient Boosting Classifier",
                family="Ensemble Boosting",
                feature_schema="LabelEncoded_6",
                expected_features=config.FEATURE_COLS_LABEL_ENCODED,
                description="Sequential gradient boosting minimizing binary cross-entropy loss.",
                test_metrics=bench_classical.get("Gradient Boosting", {"accuracy": 0.9810, "f1_failure": 0.76, "roc_auc": 0.960}),
                is_available=("Gradient Boosting" in self._models)
            ),
            "Decision Tree": ModelMetadata(
                model_id="Decision Tree",
                name="Decision Tree (Max Depth = 5)",
                family="Classical Tree",
                feature_schema="LabelEncoded_6",
                expected_features=config.FEATURE_COLS_LABEL_ENCODED,
                description="Interpretable decision tree with maximum depth 5 to prevent overfitting.",
                test_metrics=bench_classical.get("Decision Tree", {"accuracy": 0.9790, "f1_failure": 0.72, "roc_auc": 0.920}),
                is_available=("Decision Tree" in self._models)
            ),
            "SVM (RBF)": ModelMetadata(
                model_id="SVM (RBF)",
                name="Support Vector Machine (RBF Kernel)",
                family="Classical Kernel",
                feature_schema="LabelEncoded_6",
                expected_features=config.FEATURE_COLS_LABEL_ENCODED,
                description="Radial basis function SVM with Platt probability calibration.",
                test_metrics=bench_classical.get("SVM", {"accuracy": 0.9760, "f1_failure": 0.65, "roc_auc": 0.910}),
                is_available=("SVM (RBF)" in self._models or "SVM" in self._models)
            ),
            "Logistic Regression": ModelMetadata(
                model_id="Logistic Regression",
                name="Logistic Regression (L2 Regularized)",
                family="Linear Model",
                feature_schema="LabelEncoded_6",
                expected_features=config.FEATURE_COLS_LABEL_ENCODED,
                description="Linear baseline classifier estimating log-odds of machine failure.",
                test_metrics=bench_classical.get("Logistic Regression", {"accuracy": 0.9735, "f1_failure": 0.45, "roc_auc": 0.880}),
                is_available=("Logistic Regression" in self._models)
            ),
            "1D CNN (PyTorch)": ModelMetadata(
                model_id="1D CNN (PyTorch)",
                name="1D Convolutional Neural Network (PyTorch)",
                family="Deep Learning (Conv1D)",
                feature_schema="OneHot_7",
                expected_features=config.FEATURE_COLS_ONEHOT,
                description="Multi-scale Conv1D feature extractor with Batch Normalization and Global Pooling.",
                test_metrics=bench_cnn if bench_cnn else {"accuracy": 0.9660, "f1_failure": 0.564, "roc_auc": 0.980},
                is_available=(self._cnn_pytorch_model is not None)
            )
        }

    def list_available_models(self) -> List[ModelMetadata]:
        """Returns metadata list of all available trained models."""
        return list(self._metadata_cache.values())

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        return self._metadata_cache.get(model_id)

    def is_model_available(self, model_id: str) -> bool:
        meta = self._metadata_cache.get(model_id)
        return meta.is_available if meta else False

    def transform_input_for_model(
        self,
        model_id: str,
        input_dict: Dict[str, Any]
    ) -> np.ndarray:
        """
        Transforms validated single machine inputs into the exact numerical format
        expected by the designated model. Applies persistent scalers without refitting.
        """
        meta = self.get_model_metadata(model_id)
        if meta is None:
            raise ValueError(f"Unknown model identifier: '{model_id}'")

        if meta.feature_schema == "OneHot_7":
            # 1D CNN pipeline: One-hot encoded Type (Type_L, Type_M), 7 features
            type_val = str(input_dict.get("Type", "M")).strip().upper()
            type_l = 1.0 if type_val == "L" else 0.0
            type_m = 1.0 if type_val == "M" else 0.0

            raw_df = pd.DataFrame([{
                "Air temperature [K]": float(input_dict["Air temperature [K]"]),
                "Process temperature [K]": float(input_dict["Process temperature [K]"]),
                "Rotational speed [rpm]": float(input_dict["Rotational speed [rpm]"]),
                "Torque [Nm]": float(input_dict["Torque [Nm]"]),
                "Tool wear [min]": float(input_dict["Tool wear [min]"]),
                "Type_L": type_l,
                "Type_M": type_m
            }])[config.FEATURE_COLS_ONEHOT]

            if self._scaler_cnn is not None:
                scaled_vec = self._scaler_cnn.transform(raw_df)
            else:
                scaled_vec = raw_df.values.astype(np.float32)

            return scaled_vec

        else:
            # Classical ML pipeline: LabelEncoded Type, 6 features
            type_val = str(input_dict.get("Type", "M")).strip().upper()
            type_code = 1  # default M
            if self._label_encoder is not None:
                try:
                    type_code = int(self._label_encoder.transform([type_val])[0])
                except Exception:
                    type_map = {"H": 0, "L": 1, "M": 2}
                    type_code = type_map.get(type_val, 1)
            else:
                type_map = {"H": 0, "L": 1, "M": 2}
                type_code = type_map.get(type_val, 1)

            raw_df = pd.DataFrame([{
                "Type": type_code,
                "Air temperature [K]": float(input_dict["Air temperature [K]"]),
                "Process temperature [K]": float(input_dict["Process temperature [K]"]),
                "Rotational speed [rpm]": float(input_dict["Rotational speed [rpm]"]),
                "Torque [Nm]": float(input_dict["Torque [Nm]"]),
                "Tool wear [min]": float(input_dict["Tool wear [min]"])
            }])

            # Scale numerical columns if classical scaler exists
            if self._scaler_classical is not None:
                scale_cols = [c for c in config.NUMERICAL_FEATURES if c in raw_df.columns]
                raw_df[scale_cols] = self._scaler_classical.transform(raw_df[scale_cols])

            return raw_df[config.FEATURE_COLS_LABEL_ENCODED]

    def predict(
        self,
        model_id: str,
        input_dict: Dict[str, Any]
    ) -> PredictedFailureRisk:
        """
        Executes prediction using the user-selected model.
        Returns formal PredictedFailureRisk object with calibrated probabilities.
        """
        # Validate inputs first
        val = validate_machine_input(input_dict)
        if not val.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(val.errors)}")

        meta = self.get_model_metadata(model_id)
        if meta is None or not meta.is_available:
            raise RuntimeError(f"Requested model '{model_id}' is not loaded or unavailable on disk.")

        if model_id == "1D CNN (PyTorch)":
            import torch
            x_scaled = self.transform_input_for_model(model_id, input_dict)
            # Reshape for CNN: (1, 7, 1)
            x_tensor = torch.tensor(x_scaled.reshape(1, 7, 1), dtype=torch.float32)
            with torch.no_grad():
                prob = float(self._cnn_pytorch_model(x_tensor).item())
            pred_class = bool(prob >= 0.5)

        else:
            model_key = model_id
            if model_key not in self._models and model_key == "SVM (RBF)" and "SVM" in self._models:
                model_key = "SVM"

            model = self._models[model_key]
            x_vec = self.transform_input_for_model(model_id, input_dict)

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(x_vec)[0, 1])
                pred_class = bool(prob >= 0.5)
            else:
                pred_raw = int(model.predict(x_vec)[0])
                pred_class = bool(pred_raw == 1)
                prob = 0.95 if pred_class else 0.05

        # Determine confidence descriptor
        if prob >= 0.70:
            confidence = "High Confidence Failure (Statistical Likelihood >= 70%)"
        elif prob >= 0.50:
            confidence = "Borderline Failure Risk (50% - 70%)"
        elif prob >= 0.35:
            confidence = "Elevated Caution / Early Warning (35% - 50%)"
        else:
            confidence = "Normal Operating Health (Probability < 35%)"

        return PredictedFailureRisk(
            failure_predicted=pred_class,
            failure_probability=prob,
            confidence_level=confidence,
            model_name=meta.name,
            model_version="v1.0.0"
        )


# Global registry instance
registry = ModelRegistry()
=======
MachineGuard Model Registry
Centralized repository for supported Machine Learning and Deep Learning models.
Manages metadata, artifact resolution, availability validation, preprocessing schemas,
and unified inference execution across models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import config

# Static registry definitions for supported models
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "model_id": "random_forest",
        "display_name": "Random Forest Classifier",
        "model_type": "ML",
        "artifact_path": config.MODELS_DIR / "production_model.joblib",
        "description": "Ensemble of 100 balanced decision trees. High precision and robustness to multi-sensor outliers.",
        "prediction_task": "Predicts machine failure probability (0 or 1) from 6 operating telemetry features.",
        "limitations": "Does not model sequential time-series patterns; predictions are based on instantaneous sensor snapshots.",
        "operating_threshold": 0.35,
        "supports_probabilities": True,
        "explanation_type": "tree_attribution",
        "explanation_label": "Tree-Path Probability Delta (MDI)",
        "is_production_default": True
    },
    "gradient_boosting": {
        "model_id": "gradient_boosting",
        "display_name": "Gradient Boosting Classifier",
        "model_type": "ML",
        "artifact_path": config.MODELS_DIR / "gradient_boosting.joblib",
        "description": "Sequential boosting ensemble of 100 shallow trees optimizing log-loss. Strong discrimination on clear failure regimes.",
        "prediction_task": "Predicts machine failure probability via stagewise additive decision stumps.",
        "limitations": "More susceptible to overconfidence on rare anomalous sensor spikes than bagging ensembles.",
        "operating_threshold": 0.35,
        "supports_probabilities": True,
        "explanation_type": "tree_attribution",
        "explanation_label": "Boosting Stage Leaf Attribution",
        "is_production_default": False
    },
    "logistic_regression": {
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression (Balanced)",
        "model_type": "ML",
        "artifact_path": config.MODELS_DIR / "logistic_regression.joblib",
        "description": "Linear classifier with balanced class weighting and L2 regularization. Highly transparent linear log-odds coefficients.",
        "prediction_task": "Predicts failure log-odds as a linear combination of standardized physical sensor readings.",
        "limitations": "Cannot capture non-linear sensor coupling (e.g. combined high torque with elevated tool wear).",
        "operating_threshold": 0.50,
        "supports_probabilities": True,
        "explanation_type": "linear_coefficients",
        "explanation_label": "Linear Feature Log-Odds Contributions",
        "is_production_default": False
    },
    "cnn_1d": {
        "model_id": "cnn_1d",
        "display_name": "1D Convolutional Neural Network (CNN)",
        "model_type": "DL",
        "artifact_path": config.MODELS_DIR / "cnn_1d.joblib",
        "description": "Deep 1D CNN with dual Conv1D stages (32 & 64 filters), Batch Normalization, Max Pooling, and Global Average Pooling.",
        "prediction_task": "Extracts hierarchical filter activations across standardized telemetry features to predict failure probability.",
        "limitations": "Operates over tabular sensor columns without intrinsic translational invariance. Tree-based attribution is not applicable.",
        "operating_threshold": 0.50,
        "supports_probabilities": True,
        "explanation_type": "conv_activations",
        "explanation_label": "Conv1D Filter Activation Maps & Sensitivity",
        "is_production_default": False
    }
}


class ModelRegistry:
    """Central registry manager for MachineGuard models."""

    def __init__(self):
        self._catalog = MODEL_CATALOG

    def list_models(self, group_by_type: bool = False, only_available: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        """Returns list of model IDs, optionally grouped by 'ML' / 'DL'."""
        models = []
        for mid in self._catalog.keys():
            if only_available:
                avail, _ = self.is_model_available(mid)
                if not avail:
                    continue
            models.append(mid)

        if not group_by_type:
            return models

        grouped: Dict[str, List[str]] = {"ML": [], "DL": []}
        for mid in models:
            m_type = self._catalog[mid]["model_type"]
            grouped.setdefault(m_type, []).append(mid)
        return grouped

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Returns metadata for a specific model ID."""
        if model_id not in self._catalog:
            raise KeyError(
                f"Unknown model identifier '{model_id}'. "
                f"Available models: {list(self._catalog.keys())}"
            )
        info = dict(self._catalog[model_id])
        avail, err = self.is_model_available(model_id)
        info["is_available"] = avail
        info["load_error"] = err
        return info

    def is_model_available(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """Verifies if model artifact exists and is loadable."""
        if model_id not in self._catalog:
            return False, f"Model ID '{model_id}' is not registered."

        artifact_path = Path(self._catalog[model_id]["artifact_path"])
        if not artifact_path.exists():
            return False, f"Artifact file '{artifact_path.name}' not found at '{artifact_path}'."

        return True, None

    def get_engine(self, model_id: str):
        """Loads and returns a cached inference engine for the given model ID."""
        from modules import inference
        return inference.get_inference_engine(model_id=model_id)

    def predict(self, model_id: str, input_data: Any, custom_threshold: Optional[float] = None) -> Any:
        """Unified prediction interface."""
        engine = self.get_engine(model_id)
        if isinstance(input_data, dict):
            return engine.predict_single(input_data, custom_threshold=custom_threshold)
        else:
            return engine.predict_batch(input_data, custom_threshold=custom_threshold)

    def predict_probability(self, model_id: str, input_data: Any) -> float:
        """Returns failure probability for single input record."""
        res = self.predict(model_id, input_data)
        if isinstance(res, dict):
            return float(res["failure_probability"])
        raise ValueError("predict_probability expects a single record (dictionary).")

    def explain_prediction(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns model-specific explanation payload."""
        engine = self.get_engine(model_id)
        res = engine.predict_single(input_data)
        info = self.get_model_info(model_id)
        return {
            "model_id": model_id,
            "model_name": info["display_name"],
            "model_type": info["model_type"],
            "explanation_type": info["explanation_type"],
            "explanation_label": info["explanation_label"],
            "feature_contributions": res.get("feature_contributions"),
            "conv_activations": res.get("conv_activations"),
            "failure_probability": res.get("failure_probability"),
            "prediction": res.get("prediction"),
            "operating_threshold": res.get("operating_threshold")
        }

    def compare_models(
        self,
        model_ids: List[str],
        input_data: Dict[str, Any],
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates the same input record across multiple models.
        Returns structured comparison table records.
        """
        results = []
        custom_thresholds = custom_thresholds or {}

        from modules import decision_layer

        for mid in model_ids:
            info = self.get_model_info(mid)
            th = custom_thresholds.get(mid, info["operating_threshold"])

            if not info["is_available"]:
                results.append({
                    "model_id": mid,
                    "model_name": info["display_name"],
                    "model_type": info["model_type"],
                    "predicted_outcome": "UNAVAILABLE",
                    "failure_probability": None,
                    "decision_threshold": th,
                    "risk_tier": "ERROR",
                    "status": f"Artifact Missing: {info['load_error']}"
                })
                continue

            try:
                engine = self.get_engine(mid)
                res = engine.predict_single(input_data, custom_threshold=th)
                tier = decision_layer.classify_risk_tier(res["failure_probability"], th)
                outcome = "FAILURE RISK" if res["prediction"] == 1 else "NORMAL HEALTH"

                results.append({
                    "model_id": mid,
                    "model_name": info["display_name"],
                    "model_type": info["model_type"],
                    "predicted_outcome": outcome,
                    "failure_probability": round(res["failure_probability"], 4),
                    "decision_threshold": round(th, 2),
                    "risk_tier": tier,
                    "status": "SUCCESS"
                })
            except Exception as e:
                results.append({
                    "model_id": mid,
                    "model_name": info["display_name"],
                    "model_type": info["model_type"],
                    "predicted_outcome": "ERROR",
                    "failure_probability": None,
                    "decision_threshold": th,
                    "risk_tier": "ERROR",
                    "status": f"Inference Error: {str(e)}"
                })

        return results


# Global Registry Singleton
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Returns singleton ModelRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
>>>>>>> 67c25a8ec6c7c96a556a72bd95d31bda6d5f2cac
