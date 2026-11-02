"""
Unit tests for Task 2: Rule-Based Inference Engine.
Tests rule firing, conflict resolution, trace generation, and epistemic distinction.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import decision_layer as dl
from modules import knowledge_representation as kr


@pytest.fixture
def rule_engine():
    return dl.RuleEngine()


def test_rule_fires_normal_baseline(rule_engine):
    facts = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    trace = rule_engine.infer(facts)
    assert trace.winning_rule is not None
    assert trace.winning_rule.rule_id == "RULE_STAT_LOW_01"
    assert trace.recommendation.action.action_code == "ROUTINE_MONITORING"
    assert trace.recommendation.is_safe_to_operate is True


def test_rule_fires_critical_risk(rule_engine):
    facts = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 10.0
    }
    risk = kr.PredictedFailureRisk(
        failure_predicted=True,
        failure_probability=0.85,
        confidence_level="High Confidence",
        model_name="Random Forest"
    )
    trace = rule_engine.infer(facts, risk)
    assert trace.winning_rule is not None
    assert trace.winning_rule.rule_id == "RULE_STAT_CRIT_01"
    assert trace.recommendation.action.action_code == "URGENT_INSPECTION"
    assert trace.recommendation.is_safe_to_operate is False


def test_rule_conflict_resolution_priority(rule_engine):
    # Triggers both Tool Wear (priority 85) and Critical Risk (priority 95)
    facts = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 220.0  # >= 200 min triggers RULE_PHYS_TWF_01
    }
    risk = kr.PredictedFailureRisk(
        failure_predicted=True,
        failure_probability=0.90,  # >= 0.70 triggers RULE_STAT_CRIT_01 (priority 95)
        confidence_level="High Confidence",
        model_name="Random Forest"
    )
    trace = rule_engine.infer(facts, risk)
    # Both rules fired
    fired_ids = [r.rule_id for r in trace.fired_rules]
    assert "RULE_STAT_CRIT_01" in fired_ids
    assert "RULE_PHYS_TWF_01" in fired_ids
    # Highest priority wins (95 > 85)
    assert trace.winning_rule.rule_id == "RULE_STAT_CRIT_01"
    assert "Conflict Resolution" in trace.conflict_resolution_rationale


def test_invalid_input_guard_rule_fires(rule_engine):
    # Out of bounds Air Temperature triggers guard rule
    invalid_facts = {
        "Type": "M",
        "Air temperature [K]": 999.0,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    trace = rule_engine.infer(invalid_facts)
    assert trace.winning_rule.rule_id == "RULE_GUARD_01"
    assert trace.recommendation.action.action_code == "REJECT_INPUT"
    assert trace.recommendation.is_safe_to_operate is False


def test_inference_trace_integrity(rule_engine):
    facts = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 307.0,
        "Rotational speed [rpm]": 1250.0,  # Triggers HDF
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 10.0
    }
    trace = rule_engine.infer(facts)
    assert len(trace.evaluations) == len(rule_engine.rules)
    assert trace.epistemic_disclaimer is not None
    assert len(trace.epistemic_disclaimer) > 20
    d = trace.to_dict()
    assert "timestamp" in d
    assert "fired_rules" in d
    assert d["fired_rules_count"] >= 1

