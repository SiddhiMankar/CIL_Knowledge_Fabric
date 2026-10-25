"""
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
