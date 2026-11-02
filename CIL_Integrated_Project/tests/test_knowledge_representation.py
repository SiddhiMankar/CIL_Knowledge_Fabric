"""
Unit tests for Task 1: Formal Knowledge Representation.
Tests entities, attributes, physical constraints, and validation logic.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import knowledge_representation as kr


def test_schema_metadata_introspection():
    schema = kr.get_schema_metadata()
    assert "entities" in schema
    assert "Machine" in schema["entities"]
    assert "SensorReading" in schema["entities"]
    assert "OperatingCondition" in schema["entities"]
    assert "PredictedFailureRisk" in schema["entities"]
    assert "MaintenanceAction" in schema["entities"]
    assert "MaintenanceRecommendation" in schema["entities"]
    assert len(schema["relationships_graph"]) >= 6


def test_sensor_reading_derived_properties():
    reading = kr.SensorReading(
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=100.0
    )
    assert abs(reading.temperature_difference_k - 10.5) < 1e-4
    assert reading.mechanical_power_w > 0
    assert reading.overstrain_index == 4000.0


def test_valid_machine_input():
    valid_record = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    res = kr.validate_machine_input(valid_record)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.validated_data["Type"] == "M"


def test_missing_required_fields():
    incomplete = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        # Process temperature missing
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    res = kr.validate_machine_input(incomplete)
    assert res.is_valid is False
    assert any("Process temperature" in err for err in res.errors)


def test_invalid_categorical_type():
    invalid_type = {
        "Type": "Z_INVALID",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    res = kr.validate_machine_input(invalid_type)
    assert res.is_valid is False
    assert any("Type" in err for err in res.errors)


def test_out_of_bounds_numerical():
    # Air temperature out of [295.0, 305.0]
    out_of_bounds = {
        "Type": "L",
        "Air temperature [K]": 350.0,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    res = kr.validate_machine_input(out_of_bounds)
    assert res.is_valid is False
    assert any("Air temperature [K]" in err for err in res.errors)


def test_recommendation_consistency_check():
    risk = kr.PredictedFailureRisk(
        failure_predicted=True,
        failure_probability=0.95,
        confidence_level="High Confidence",
        model_name="Random Forest"
    )
    # Contradictory recommendation: high failure risk declared safe
    action = kr.MaintenanceAction(
        action_code="ROUTINE_MONITORING",
        title="Continue",
        description="Routine",
        estimated_duration_min=0,
        severity=kr.ActionSeverity.INFO
    )
    rec = kr.MaintenanceRecommendation(
        action=action,
        reasoning="Test",
        triggered_rules=[],
        is_safe_to_operate=True,  # Inconsistent with 95% failure probability!
        recommended_timeframe="Next cycle",
        model_risk_score=0.95
    )
    consistent, msg = kr.validate_recommendation_consistency(risk, rec)
    assert consistent is False
    assert "Inconsistent" in msg

