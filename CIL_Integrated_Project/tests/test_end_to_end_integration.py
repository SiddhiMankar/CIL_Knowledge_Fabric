"""
Integration tests for Task 9: Complete End-to-End Pipeline.
Exercises:
Input -> Validation -> Preprocessing -> Dynamic Model Selection -> Inference ->
Rule Engine Synthesis -> AI Planning Scenario -> History Persistence.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import (
    knowledge_representation as kr,
    decision_layer as dl,
    model_registry as mr,
    planning,
    history_manager as hm
)


def test_full_pipeline_normal_asset():
    raw_input = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }

    # Step 1 & 2: Input Validation
    val = kr.validate_machine_input(raw_input)
    assert val.is_valid is True

    # Step 3, 4, 5: Model Selection and Real Inference
    assert mr.registry.is_model_available("Random Forest")
    pred_risk = mr.registry.predict("Random Forest", raw_input)
    assert isinstance(pred_risk.failure_probability, float)
    assert 0.0 <= pred_risk.failure_probability <= 1.0

    # Step 6: Rule-Based Decision Making
    engine = dl.RuleEngine()
    trace = engine.infer(raw_input, pred_risk)
    assert trace.recommendation is not None
    assert trace.recommendation.action.action_code in ["ROUTINE_MONITORING", "SCHEDULED_MAINTENANCE"]

    # Step 7: Planning Bridge
    plan_scen = planning.get_planning_scenario_for_recommendation(trace.recommendation.action.action_code)
    assert plan_scen["optimal_plan"]["found"] is True

    # Step 8: Persistence
    record = {
        "Type": raw_input["Type"],
        "Air temperature [K]": raw_input["Air temperature [K]"],
        "Process temperature [K]": raw_input["Process temperature [K]"],
        "Rotational speed [rpm]": raw_input["Rotational speed [rpm]"],
        "Torque [Nm]": raw_input["Torque [Nm]"],
        "Tool wear [min]": raw_input["Tool wear [min]"],
        "model_name": "Random Forest",
        "failure_probability": pred_risk.failure_probability,
        "recommendation_action": trace.recommendation.action.action_code,
        "is_safe_to_operate": trace.recommendation.is_safe_to_operate,
        "fired_rules": [r.rule_id for r in trace.fired_rules]
    }
    saved, err = hm.save_assessment_record(record)
    assert saved is True

    # History retrieval verification
    hist = hm.get_assessment_history(limit=5)
    assert len(hist) > 0
    assert hist[0]["model_name"] == "Random Forest"


def test_full_pipeline_pytorch_cnn_model():
    if not mr.registry.is_model_available("1D CNN (PyTorch)"):
        pytest.skip("PyTorch 1D CNN model not available on this platform.")

    raw_input = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 307.0,
        "Rotational speed [rpm]": 1250.0,
        "Torque [Nm]": 60.0,
        "Tool wear [min]": 180.0
    }

    # Validation
    val = kr.validate_machine_input(raw_input)
    assert val.is_valid is True

    # 1D CNN Inference
    pred = mr.registry.predict("1D CNN (PyTorch)", raw_input)
    assert pred.model_name is not None
    assert isinstance(pred.failure_probability, float)

    # Rule Engine
    engine = dl.RuleEngine()
    trace = engine.infer(raw_input, pred)
    assert trace.recommendation is not None
    assert trace.recommendation.is_safe_to_operate is False

