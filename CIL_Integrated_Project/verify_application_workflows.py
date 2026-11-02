"""
Verification script for Point 1: Testing all application workflows, pages, models, and traces.
"""

import os
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "mpl"))

import pandas as pd
import numpy as np

import config
from modules import (
    knowledge_representation as kr,
    decision_layer as dl,
    planning,
    eda,
    classical_ml,
    regression,
    cnn_model,
    interpretability,
    model_registry,
    history_manager
)


def verify_all():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE VERIFICATION PASS ACROSS ALL PLATFORM WORKFLOWS")
    print("=" * 70)

    # 1. Dataset Loading
    print("[1/9] Dataset Loading Check...")
    found, p = config.locate_dataset()
    assert found and p is not None, "Dataset not located"
    df = eda.load_dataset()
    assert df.shape == (10000, 14), f"Unexpected shape {df.shape}"
    summary = eda.get_dataset_summary(df)
    print(f"      OK: Shape={df.shape}, Missing={summary['total_missing']}, Duplicates={summary['duplicate_rows']}")

    # 2. Dynamic Model Selection
    print("[2/9] Dynamic Model Selection Check...")
    reg = model_registry.registry
    active_models = [m.model_id for m in reg.list_available_models() if m.is_available]
    print(f"      Active models in registry ({len(active_models)}): {active_models}")
    assert "Random Forest" in active_models
    assert "Gradient Boosting" in active_models
    assert "1D CNN (PyTorch)" in active_models

    # 3. Input Validation
    print("[3/9] Prediction Input Validation Check...")
    valid_sample = {
        "Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0, "Torque [Nm]": 42.8, "Tool wear [min]": 0.0
    }
    v_ok = kr.validate_machine_input(valid_sample)
    assert v_ok.is_valid is True, "Valid sample failed validation"

    invalid_sample = dict(valid_sample)
    invalid_sample["Air temperature [K]"] = 450.0  # out of bounds
    v_bad = kr.validate_machine_input(invalid_sample)
    assert v_bad.is_valid is False, "Invalid sample passed validation"
    print("      OK: Valid input accepted, out-of-bounds input rejected with clear diagnostic.")

    # 4. ML and CNN Inference
    print("[4/9] ML & CNN Inference Check...")
    # Classical ML (Random Forest)
    rf_pred = reg.predict("Random Forest", valid_sample)
    assert isinstance(rf_pred.failure_probability, float)
    print(f"      Random Forest: P(Failure)={rf_pred.failure_probability:.4f}, Class={rf_pred.failure_predicted}, Conf={rf_pred.confidence_level}")

    # Deep Learning (1D CNN PyTorch)
    cnn_pred = reg.predict("1D CNN (PyTorch)", valid_sample)
    assert isinstance(cnn_pred.failure_probability, float)
    print(f"      1D CNN (PyTorch): P(Failure)={cnn_pred.failure_probability:.4f}, Class={cnn_pred.failure_predicted}, Conf={cnn_pred.confidence_level}")

    # 5. Rule-Engine Execution & Inference Traces
    print("[5/9] Rule Engine & Inference Trace Check...")
    engine = dl.RuleEngine()
    trace_rf = engine.infer(valid_sample, rf_pred)
    assert trace_rf.winning_rule is not None
    assert trace_rf.recommendation is not None
    assert trace_rf.epistemic_disclaimer is not None
    print(f"      Rule Fired: {trace_rf.winning_rule.rule_id} -> Action: {trace_rf.recommendation.action.action_code}")
    print(f"      Conflict Rationale: {trace_rf.conflict_resolution_rationale}")

    # Test Guard Rule on Invalid Data
    trace_guard = engine.infer(invalid_sample)
    assert trace_guard.winning_rule.rule_id == "RULE_GUARD_01"
    assert trace_guard.recommendation.action.action_code == "REJECT_INPUT"
    print("      Guard Rule Fired correctly: REJECT_INPUT on invalid telemetry.")

    # 6. Maintenance Planning Bridge
    print("[6/9] Maintenance Planning Bridge Check...")
    plan_scen = planning.get_planning_scenario_for_recommendation(trace_rf.recommendation.action.action_code)
    astar_plan = plan_scen["optimal_plan"]
    assert astar_plan["found"] is True
    print(f"      Plan Generated: Algorithm={astar_plan['algorithm']}, Cost={astar_plan['path_cost']}, Actions={len(astar_plan['actions'])}, Nodes={astar_plan['nodes_expanded']}")

    # 7. Explainability & What-If Functionality
    print("[7/9] Explainability & What-If Simulation Check...")
    what_if = interpretability.simulate_what_if(
        baseline_record=valid_sample,
        modifications={"Torque [Nm]": 68.0, "Tool wear [min]": 215.0},
        model_id="Random Forest"
    )
    print(f"      Baseline P(Fail)={what_if['baseline_risk']['probability']:.2%} ({what_if['baseline_risk']['action']})")
    print(f"      Modified P(Fail)={what_if['modified_risk']['probability']:.2%} ({what_if['modified_risk']['action']})")
    print(f"      Risk Delta={what_if['delta']['probability_change']:+.2%}")
    assert what_if['delta']['probability_change'] > 0.5, "Sensitivity delta did not increase failure risk as expected"

    # CNN Activations
    c1, c2 = interpretability.extract_real_cnn_activations(valid_sample)
    assert c1 is not None and c2 is not None
    print(f"      Extracted Real Conv1D Activations: Conv1={c1.shape}, Conv2={c2.shape}")

    # 8. Assessment History Persistence
    print("[8/9] Assessment History Persistence Check...")
    test_record = {
        "Type": valid_sample["Type"],
        "Air temperature [K]": valid_sample["Air temperature [K]"],
        "Process temperature [K]": valid_sample["Process temperature [K]"],
        "Rotational speed [rpm]": valid_sample["Rotational speed [rpm]"],
        "Torque [Nm]": valid_sample["Torque [Nm]"],
        "Tool wear [min]": valid_sample["Tool wear [min]"],
        "model_name": "Random Forest",
        "failure_probability": rf_pred.failure_probability,
        "recommendation_action": trace_rf.recommendation.action.action_code,
        "is_safe_to_operate": trace_rf.recommendation.is_safe_to_operate,
        "fired_rules": [r.rule_id for r in trace_rf.fired_rules]
    }
    saved, err = history_manager.save_assessment_record(test_record)
    assert saved is True
    hist_df = history_manager.get_history_as_dataframe()
    assert not hist_df.empty
    print(f"      OK: History successfully saved and retrieved ({len(hist_df)} records logged).")

    # 9. Validation & Ethics Document Check
    print("[9/9] Validation & Ethics Document Check...")
    val_doc = config.PROJECT_ROOT / "docs" / "VALIDATION_AND_ETHICS.md"
    assert val_doc.exists(), "docs/VALIDATION_AND_ETHICS.md does not exist"
    print(f"      OK: {val_doc} exists and contains {val_doc.stat().st_size} bytes.")

    print("=" * 70)
    print("ALL 9 CORE APPLICATION WORKFLOW CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    verify_all()

