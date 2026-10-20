"""
Unit and integration tests for MachineGuard:
- MachineGuardInferenceEngine (input schema, ranges, single/batch prediction, determinism)
- Decision Layer (risk tiers, threshold sensitivity, economic tradeoff)
- History Manager (logging, persistence, export schema)
- Real Model & Pipeline integrity (leakage prevention, feature order preservation)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import config
from modules.inference import get_inference_engine, MachineGuardInferenceEngine
from modules.decision_layer import assess_operational_decision, classify_risk_tier
from modules.history_manager import get_history_manager, AssessmentHistoryManager
from modules.interpretability import plot_what_if_sensitivity

@pytest.fixture
def valid_sensor_input():
    return {
        "Type": "L",
        "Air temperature [K]": 300.2,
        "Process temperature [K]": 310.1,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 110.0
    }

def test_inference_engine_initialization():
    engine = get_inference_engine()
    assert engine.model is not None
    assert engine.preprocessor is not None
    assert len(engine.raw_feature_cols) == 6
    assert "Type" in engine.raw_feature_cols

def test_valid_single_prediction(valid_sensor_input):
    engine = get_inference_engine()
    res = engine.predict_single(valid_sensor_input)
    assert "prediction" in res
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["failure_probability"] <= 1.0
    assert "feature_contributions" in res
    assert len(res["feature_contributions"]) == len(engine.transformed_feature_cols)

def test_input_validation_missing_column(valid_sensor_input):
    engine = get_inference_engine()
    invalid_input = valid_sensor_input.copy()
    del invalid_input["Torque [Nm]"]
    is_valid, err, _ = engine.validate_input(invalid_input)
    assert not is_valid
    assert "Missing required input feature" in err

def test_input_validation_invalid_type(valid_sensor_input):
    engine = get_inference_engine()
    invalid_input = valid_sensor_input.copy()
    invalid_input["Type"] = "INVALID_TYPE"
    is_valid, err, _ = engine.validate_input(invalid_input)
    assert not is_valid
    assert "Invalid machine Type" in err

def test_input_validation_out_of_bounds(valid_sensor_input):
    engine = get_inference_engine()
    invalid_input = valid_sensor_input.copy()
    invalid_input["Rotational speed [rpm]"] = 15000.0  # Absurdly high
    is_valid, err, _ = engine.validate_input(invalid_input)
    assert not is_valid
    assert "out of plausible operating range" in err

def test_batch_prediction_dataframe(valid_sensor_input):
    engine = get_inference_engine()
    batch_df = pd.DataFrame([
        valid_sensor_input,
        {
            "Type": "M",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 311.5,
            "Rotational speed [rpm]": 1350.0,
            "Torque [Nm]": 62.0,
            "Tool wear [min]": 220.0
        }
    ])
    result_df = engine.predict_batch(batch_df)
    assert "Failure_Probability" in result_df.columns
    assert "Predicted_Failure" in result_df.columns
    assert "Risk_Level" in result_df.columns
    assert len(result_df) == 2

def test_prediction_reproducibility(valid_sensor_input):
    engine = get_inference_engine()
    res1 = engine.predict_single(valid_sensor_input)
    res2 = engine.predict_single(valid_sensor_input)
    assert res1["failure_probability"] == res2["failure_probability"]
    assert res1["prediction"] == res2["prediction"]

def test_decision_layer_mapping(valid_sensor_input):
    engine = get_inference_engine()
    res = engine.predict_single(valid_sensor_input)
    decision = assess_operational_decision(res)
    assert decision["risk_tier"] in ("NOMINAL", "MONITORING", "ELEVATED", "CRITICAL")
    assert "primary_action" in decision
    assert "economic_context" in decision
    assert len(decision["limitations"]) > 0

def test_critical_failure_condition_decision():
    engine = get_inference_engine()
    # High wear + high torque simulates extreme failure conditions
    severe_input = {
        "Type": "L",
        "Air temperature [K]": 303.0,
        "Process temperature [K]": 313.0,
        "Rotational speed [rpm]": 1250.0,
        "Torque [Nm]": 70.0,
        "Tool wear [min]": 240.0
    }
    res = engine.predict_single(severe_input)
    decision = assess_operational_decision(res)
    assert res["failure_probability"] > 0.50
    assert decision["risk_tier"] in ("ELEVATED", "CRITICAL")
    assert "Shutdown" in decision["primary_action"] or "Inspection" in decision["primary_action"]

def test_history_logging_and_export(valid_sensor_input):
    engine = get_inference_engine()
    res = engine.predict_single(valid_sensor_input)
    decision = assess_operational_decision(res)
    
    manager = get_history_manager()
    initial_count = len(manager.records)
    rec = manager.add_record(
        machine_id="TEST-UNIT-01",
        inputs=valid_sensor_input,
        inference_result=res,
        decision_result=decision,
        source="Automated Test"
    )
    assert rec["machine_id"] == "TEST-UNIT-01"
    assert len(manager.records) == initial_count + 1
    
    df = manager.get_history_dataframe()
    assert not df.empty
    assert "Assessment ID" in df.columns
    assert "Operating Threshold" in df.columns
    assert "Model Version" in df.columns


def test_missing_model_controlled_failure(valid_sensor_input):
    """Missing model artifact produces a clear, controlled failure without silent training."""
    fake_path = Path("non_existent_production_model_9999.joblib")
    engine = MachineGuardInferenceEngine(model_path=fake_path)
    assert not engine.is_loaded
    assert engine.load_error is not None
    assert "not found" in engine.load_error.lower()
    
    with pytest.raises(RuntimeError) as exc_info:
        engine.predict_single(valid_sensor_input)
    assert "not loaded" in str(exc_info.value).lower()


def test_corrupt_model_artifact_detection(tmp_path):
    """Incompatible or corrupt artifacts do not produce misleading predictions."""
    corrupt_file = tmp_path / "corrupt_model.joblib"
    joblib.dump({"invalid_key": "not_a_model"}, corrupt_file)
    
    engine = MachineGuardInferenceEngine(model_path=corrupt_file)
    assert not engine.is_loaded
    assert engine.load_error is not None
    assert "corrupt" in engine.load_error.lower()


def test_feature_schema_mismatch_detection(tmp_path):
    """Detects when an artifact was saved with an unexpected feature schema."""
    mismatched_file = tmp_path / "mismatch_model.joblib"
    joblib.dump({
        "model": "dummy",
        "preprocessor": "dummy",
        "feature_names_raw": ["OnlyOneFeature"]
    }, mismatched_file)
    
    engine = MachineGuardInferenceEngine(model_path=mismatched_file)
    assert not engine.is_loaded
    assert "schema mismatch" in engine.load_error.lower()


def test_input_validation_nan_and_infinite_values(valid_sensor_input):
    """NaN and infinite values are explicitly rejected as invalid physical sensor readings."""
    engine = get_inference_engine()
    
    # NaN check
    nan_input = valid_sensor_input.copy()
    nan_input["Torque [Nm]"] = np.nan
    is_valid, err, _ = engine.validate_input(nan_input)
    assert not is_valid
    assert "missing (NaN) values" in err

    # Positive Infinity check
    inf_input = valid_sensor_input.copy()
    inf_input["Torque [Nm]"] = float("inf")
    is_valid, err, _ = engine.validate_input(inf_input)
    assert not is_valid
    assert "infinite" in err.lower()

    # Negative Infinity check
    neginf_input = valid_sensor_input.copy()
    neginf_input["Air temperature [K]"] = float("-inf")
    is_valid, err, _ = engine.validate_input(neginf_input)
    assert not is_valid
    assert "infinite" in err.lower()


def test_invalid_custom_threshold_rejection(valid_sensor_input):
    """Validates custom thresholds and rejects values outside [0.0, 1.0]."""
    engine = get_inference_engine()
    
    for invalid_th in [-0.05, 1.05, float("nan"), None]:
        if invalid_th is None:
            # None should default safely to operating_threshold without error
            res = engine.predict_single(valid_sensor_input, custom_threshold=None)
            assert res["operating_threshold"] == engine.operating_threshold
        else:
            with pytest.raises(ValueError):
                engine.predict_single(valid_sensor_input, custom_threshold=invalid_th)
            with pytest.raises(ValueError):
                batch_df = pd.DataFrame([valid_sensor_input])
                engine.predict_batch(batch_df, custom_threshold=invalid_th)


def test_threshold_boundary_classification():
    """Consistent classification at exact threshold boundaries."""
    th = 0.35
    # Boundary: proba >= 0.70 -> CRITICAL
    assert classify_risk_tier(0.70, th) == "CRITICAL"
    assert classify_risk_tier(0.6999, th) == "ELEVATED"
    
    # Boundary: proba >= 0.35 -> ELEVATED
    assert classify_risk_tier(0.35, th) == "ELEVATED"
    assert classify_risk_tier(0.3499, th) == "MONITORING"
    
    # Boundary: proba >= 0.175 (0.5 * 0.35) -> MONITORING
    assert classify_risk_tier(0.175, th) == "MONITORING"
    assert classify_risk_tier(0.1749, th) == "NOMINAL"
    assert classify_risk_tier(0.0, th) == "NOMINAL"
    assert classify_risk_tier(1.0, th) == "CRITICAL"


def test_batch_prediction_edge_cases(valid_sensor_input):
    """Batch assessment handles empty DataFrames, missing columns, and invalid records safely."""
    engine = get_inference_engine()
    
    # Empty DataFrame
    with pytest.raises(ValueError) as exc_info:
        engine.predict_batch(pd.DataFrame())
    assert "empty" in str(exc_info.value).lower()
    
    # Missing required column in batch
    bad_batch = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 300.0
        # Missing other sensors
    }])
    with pytest.raises(ValueError) as exc_info:
        engine.predict_batch(bad_batch)
    assert "missing required input feature" in str(exc_info.value).lower()

    # Invalid machine type in batch
    bad_type_batch = pd.DataFrame([{
        "Type": "X",  # Invalid
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0
    }])
    with pytest.raises(ValueError) as exc_info:
        engine.predict_batch(bad_type_batch)
    assert "invalid machine type" in str(exc_info.value).lower()


def test_single_and_batch_consistency(valid_sensor_input):
    """Ensures single-machine and batch inference use identical probability and risk classification."""
    engine = get_inference_engine()
    records = [
        valid_sensor_input,
        {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 5.0
        },
        {
            "Type": "L",
            "Air temperature [K]": 302.5,
            "Process temperature [K]": 311.8,
            "Rotational speed [rpm]": 1390.0,
            "Torque [Nm]": 65.0,
            "Tool wear [min]": 225.0
        }
    ]
    batch_df = pd.DataFrame(records)
    batch_res = engine.predict_batch(batch_df)
    
    for i, rec in enumerate(records):
        single_res = engine.predict_single(rec)
        decision = assess_operational_decision(single_res)
        
        # Verify probability agrees within floating tolerance
        assert abs(batch_res.loc[i, "Failure_Probability"] - single_res["failure_probability"]) < 1e-3
        # Verify Risk Level agrees exactly
        assert batch_res.loc[i, "Risk_Level"] == decision["risk_tier"]


def test_what_if_simulation_pipeline(valid_sensor_input):
    """What-if sensitivity simulation uses the exact inference pipeline."""
    engine = get_inference_engine()
    fig, sim_df = plot_what_if_sensitivity(
        engine=engine,
        base_input=valid_sensor_input,
        feature_to_vary="Torque [Nm]",
        min_val=20.0,
        max_val=80.0,
        steps=5
    )
    assert fig is not None
    assert len(sim_df) == 5
    assert "Failure Probability" in sim_df.columns
    assert sim_df["Failure Probability"].between(0.0, 1.0).all()


def test_explainability_causality_disclaimer(valid_sensor_input):
    """Explainability outputs explicitly distinguish model-based statistical evidence from physical causation."""
    engine = get_inference_engine()
    res = engine.predict_single(valid_sensor_input)
    decision = assess_operational_decision(res)
    
    assert "limitations" in decision
    assert len(decision["limitations"]) > 0
    joined_lims = " ".join(decision["limitations"]).lower()
    assert "correlation" in joined_lims or "statistical" in joined_lims
    assert "decision-support" in joined_lims or "decision support" in joined_lims


def test_history_recovery_from_corrupt_file(tmp_path):
    """History manager recovers gracefully if persisted file is corrupted."""
    bad_file = tmp_path / "corrupted_history.json"
    bad_file.write_text("{ this is not valid json : [", encoding="utf-8")
    
    manager = AssessmentHistoryManager(history_file=bad_file)
    assert manager.records == []
    
    # Should safely allow logging a new record
    engine = get_inference_engine()
    res = {"failure_probability": 0.25, "prediction": 0, "operating_threshold": 0.35, "model_version": "1.0.0"}
    dec = {"risk_tier": "MONITORING", "primary_action": "Routine Monitoring"}
    rec = manager.add_record("RECOVER-01", {"Type": "M"}, res, dec)
    assert len(manager.records) == 1
    assert rec["machine_id"] == "RECOVER-01"


def test_history_export_schema(valid_sensor_input):
    """Exported history records match expected column headers."""
    engine = get_inference_engine()
    res = engine.predict_single(valid_sensor_input)
    dec = assess_operational_decision(res)
    
    mgr = get_history_manager()
    mgr.add_record("EXPORT-TEST-01", valid_sensor_input, res, dec)
    df = mgr.get_history_dataframe()
    
    required_export_cols = [
        "Assessment ID", "Timestamp", "Machine ID", "Source", "Type",
        "Air Temp [K]", "Process Temp [K]", "Speed [rpm]", "Torque [Nm]",
        "Tool Wear [min]", "Failure Prob", "Operating Threshold",
        "Predicted Failure", "Risk Tier", "Action", "Model Version"
    ]
    for col in required_export_cols:
        assert col in df.columns, f"Missing expected export column: {col}"

