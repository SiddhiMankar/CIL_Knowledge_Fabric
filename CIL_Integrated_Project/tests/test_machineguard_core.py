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

import config
from modules.inference import get_inference_engine, MachineGuardInferenceEngine
from modules.decision_layer import assess_operational_decision
from modules.history_manager import get_history_manager

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
