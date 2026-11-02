"""
Unit and integration tests for user-selectable ML/DL model functionality in MachineGuard:
- Model Registry catalog, artifact resolution, and availability validation
- Unified inference execution across Random Forest, Gradient Boosting, Logistic Regression, and 1D CNN
- Verified model invocation (no silent substitution or fallback)
- Model-specific explanations (tree path, linear log-odds, CNN activations)
- Multi-model comparison across identical telemetry inputs
- Batch processing with model selection and row-level fault tolerance
- Assessment history persistence with model metadata and backward compatibility
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules.model_registry import get_model_registry, MODEL_CATALOG
from modules.inference import get_inference_engine, MachineGuardInferenceEngine
from modules.decision_layer import assess_operational_decision
from modules.history_manager import AssessmentHistoryManager

@pytest.fixture
def valid_telemetry():
    return {
        "Type": "L",
        "Air temperature [K]": 300.2,
        "Process temperature [K]": 310.1,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 110.0
    }

@pytest.fixture
def extreme_telemetry():
    """Severe condition that triggers elevated failure probability."""
    return {
        "Type": "L",
        "Air temperature [K]": 303.0,
        "Process temperature [K]": 313.0,
        "Rotational speed [rpm]": 1250.0,
        "Torque [Nm]": 70.0,
        "Tool wear [min]": 240.0
    }


# ============================================================================
# 1. MODEL REGISTRY TESTS
# ============================================================================

def test_registry_lists_all_candidate_models():
    registry = get_model_registry()
    models = registry.list_models()
    expected = ["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"]
    for m in expected:
        assert m in models, f"Model '{m}' missing from registry list."


def test_registry_group_by_type():
    registry = get_model_registry()
    grouped = registry.list_models(group_by_type=True)
    assert "ML" in grouped
    assert "DL" in grouped
    assert "random_forest" in grouped["ML"]
    assert "gradient_boosting" in grouped["ML"]
    assert "logistic_regression" in grouped["ML"]
    assert "cnn_1d" in grouped["DL"]


def test_registry_artifact_resolution():
    registry = get_model_registry()
    for mid in ["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"]:
        info = registry.get_model_info(mid)
        assert info["is_available"] is True
        assert Path(info["artifact_path"]).exists()
        assert info["load_error"] is None


def test_registry_invalid_model_rejection():
    registry = get_model_registry()
    with pytest.raises(KeyError) as exc_info:
        registry.get_model_info("non_existent_model_id")
    assert "Unknown model identifier" in str(exc_info.value)


# ============================================================================
# 2. INFERENCE AND VERIFIED MODEL INVOCATION
# ============================================================================

def test_each_model_predict_single(valid_telemetry):
    registry = get_model_registry()
    for mid in ["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"]:
        engine = registry.get_engine(mid)
        assert engine.is_loaded is True
        assert engine.model_id == mid

        res = engine.predict_single(valid_telemetry)
        assert res["model_id"] == mid
        assert "prediction" in res
        assert res["prediction"] in (0, 1)
        assert 0.0 <= res["failure_probability"] <= 1.0
        assert res["operating_threshold"] > 0.0
        assert "input_features" in res
        assert "transformed_features" in res


def test_verified_model_invocation_underlying_types():
    """Verify that each engine actually invokes the intended model class, not a generic fallback."""
    rf_engine = get_inference_engine(model_id="random_forest")
    assert "RandomForest" in type(rf_engine.model).__name__

    gb_engine = get_inference_engine(model_id="gradient_boosting")
    assert "GradientBoosting" in type(gb_engine.model).__name__

    lr_engine = get_inference_engine(model_id="logistic_regression")
    assert "LogisticRegression" in type(lr_engine.model).__name__

    cnn_engine = get_inference_engine(model_id="cnn_1d")
    assert "PyTorch1DCNN" in type(cnn_engine.model).__name__ or "CNN" in type(cnn_engine.model).__name__


def test_no_silent_fallback_on_unloaded_model(valid_telemetry):
    """A missing model artifact must fail explicitly rather than silently falling back to Random Forest."""
    fake_path = Path("definitely_missing_custom_model_xyz.joblib")
    bad_engine = MachineGuardInferenceEngine(model_path=fake_path)
    assert bad_engine.is_loaded is False
    assert bad_engine.load_error is not None

    with pytest.raises(RuntimeError) as exc_info:
        bad_engine.predict_single(valid_telemetry)
    assert "not loaded" in str(exc_info.value).lower()


# ============================================================================
# 3. MODEL-SPECIFIC EXPLANATIONS
# ============================================================================

def test_model_specific_explanations(valid_telemetry):
    registry = get_model_registry()

    # Tree-based explanation for Random Forest
    rf_expl = registry.explain_prediction("random_forest", valid_telemetry)
    assert rf_expl["explanation_type"] == "tree_attribution"
    assert len(rf_expl["feature_contributions"]) == 7

    # Boosting stage explanation for Gradient Boosting
    gb_expl = registry.explain_prediction("gradient_boosting", valid_telemetry)
    assert gb_expl["explanation_type"] == "tree_attribution"
    assert len(gb_expl["feature_contributions"]) == 7

    # Linear coefficient explanation for Logistic Regression
    lr_expl = registry.explain_prediction("logistic_regression", valid_telemetry)
    assert lr_expl["explanation_type"] == "linear_coefficients"
    assert len(lr_expl["feature_contributions"]) == 7

    # Conv1D activation map explanation for 1D CNN (tree-based not used)
    cnn_expl = registry.explain_prediction("cnn_1d", valid_telemetry)
    assert cnn_expl["explanation_type"] == "conv_activations"
    assert cnn_expl["conv_activations"] is not None
    assert "conv1d_1" in cnn_expl["conv_activations"]


# ============================================================================
# 4. MULTI-MODEL COMPARISON
# ============================================================================

def test_multi_model_comparison(valid_telemetry):
    registry = get_model_registry()
    model_ids = ["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"]
    comparison = registry.compare_models(model_ids, valid_telemetry)

    assert len(comparison) == 4
    for row in comparison:
        assert row["status"] == "SUCCESS"
        assert row["predicted_outcome"] in ("FAILURE RISK", "NORMAL HEALTH")
        assert 0.0 <= row["failure_probability"] <= 1.0
        assert row["risk_tier"] in ("NOMINAL", "MONITORING", "ELEVATED", "CRITICAL")
        assert row["model_type"] in ("ML", "DL")


# ============================================================================
# 5. BATCH INFERENCE WITH MODEL SELECTION & ROBUSTNESS
# ============================================================================

def test_batch_prediction_selected_model_columns(valid_telemetry):
    batch_df = pd.DataFrame([valid_telemetry, valid_telemetry])
    for mid in ["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"]:
        engine = get_inference_engine(model_id=mid)
        res_df = engine.predict_batch(batch_df)

        assert "Selected_Model" in res_df.columns
        assert "Model_Type" in res_df.columns
        assert "Decision_Threshold" in res_df.columns
        assert "Prediction_Status" in res_df.columns
        assert res_df["Selected_Model"].iloc[0] == engine.model_name
        assert (res_df["Prediction_Status"] == "SUCCESS").all()


def test_batch_prediction_robust_row_level_fault_tolerance(valid_telemetry):
    """Invalid row gets an error status without crashing the entire batch."""
    engine = get_inference_engine(model_id="random_forest")
    bad_row = valid_telemetry.copy()
    bad_row["Rotational speed [rpm]"] = 99999.0  # Physically absurd value

    df = pd.DataFrame([valid_telemetry, bad_row])
    robust_res = engine.predict_batch_robust(df)

    assert len(robust_res) == 2
    # Row 0 is valid
    assert robust_res.loc[0, "Prediction_Status"] == "SUCCESS"
    assert robust_res.loc[0, "Failure_Probability"] is not None

    # Row 1 is invalid
    assert "Validation Error" in robust_res.loc[1, "Prediction_Status"]
    assert robust_res.loc[1, "Risk_Level"] == "INVALID"


# ============================================================================
# 6. ASSESSMENT HISTORY PERSISTENCE & FILTERING
# ============================================================================

def test_history_logging_with_model_metadata(tmp_path, valid_telemetry):
    hist_file = tmp_path / "test_history.json"
    mgr = AssessmentHistoryManager(history_file=hist_file)

    for mid in ["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"]:
        engine = get_inference_engine(model_id=mid)
        res = engine.predict_single(valid_telemetry)
        dec = assess_operational_decision(res)
        rec = mgr.add_record(
            machine_id=f"TEST-{mid}",
            inputs=valid_telemetry,
            inference_result=res,
            decision_result=dec,
            source="Automated Test",
            model_id=mid
        )
        assert rec["model_id"] == mid
        assert rec["model_name"] == engine.model_name
        assert rec["model_type"] == engine.model_type

    df = mgr.get_history_dataframe()
    assert len(df) == 4
    assert "Model" in df.columns
    assert "Model Type" in df.columns
    assert "Model ID" in df.columns

    # Test filtering by specific model
    rf_df = mgr.get_history_dataframe(model_filter="random_forest")
    assert len(rf_df) == 1
    assert rf_df["Model ID"].iloc[0] == "random_forest"


def test_history_legacy_records_backward_compatibility(tmp_path):
    """Old records without model_id or model_type load safely as Legacy."""
    import json
    hist_file = tmp_path / "legacy_history.json"
    legacy_data = [
        {
            "assessment_id": "MG-20250101-001",
            "timestamp": "2025-01-01 12:00:00",
            "machine_id": "LEGACY-01",
            "source": "Manual",
            "inputs": {"Type": "M"},
            "failure_probability": 0.15,
            "operating_threshold": 0.35,
            "predicted_failure": 0,
            "risk_tier": "NOMINAL",
            "primary_action": "Normal Operation"
            # No model_id, model_name, model_type
        }
    ]
    hist_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    mgr = AssessmentHistoryManager(history_file=hist_file)
    df = mgr.get_history_dataframe()
    assert len(df) == 1
    assert df["Model"].iloc[0] == "Legacy (Unknown)"
    assert df["Model Type"].iloc[0] == "Unknown"
    assert df["Model ID"].iloc[0] == "legacy_unknown"
