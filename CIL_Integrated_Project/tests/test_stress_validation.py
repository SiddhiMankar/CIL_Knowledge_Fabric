"""
Validation and stress tests for Task 10: Robustness, Input Stress Testing & Guard Boundaries.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import knowledge_representation as kr
from modules import decision_layer as dl


@pytest.mark.parametrize("invalid_input,expected_error_keyword", [
    (
        {"Type": "M", "Process temp": 308.0},  # Missing required keys
        "Missing required attribute"
    ),
    (
        {"Type": "X", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
         "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 0.0},
        "Type"
    ),
    (
        {"Type": "M", "Air temperature [K]": "NOT_A_FLOAT", "Process temperature [K]": 308.0,
         "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 0.0},
        "numeric"
    ),
    (
        {"Type": "M", "Air temperature [K]": 550.0, "Process temperature [K]": 308.0,
         "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 0.0},
        "out of permitted range"
    ),
    (
        {"Type": "M", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
         "Rotational speed [rpm]": 5000.0, "Torque [Nm]": 40.0, "Tool wear [min]": 0.0},
        "out of permitted range"
    ),
    (
        {"Type": "M", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
         "Rotational speed [rpm]": 1500.0, "Torque [Nm]": -20.0, "Tool wear [min]": 0.0},
        "out of permitted range"
    ),
    (
        {"Type": "M", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
         "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 450.0},
        "out of permitted range"
    )
])
def test_input_validation_boundaries(invalid_input, expected_error_keyword):
    val_res = kr.validate_machine_input(invalid_input)
    assert val_res.is_valid is False
    assert any(expected_error_keyword in err for err in val_res.errors)


def test_guard_rule_prevents_unsupported_decision():
    engine = dl.RuleEngine()
    corrupt_input = {
        "Type": "CORRUPT",
        "Air temperature [K]": 9999.0
    }
    trace = engine.infer(corrupt_input)
    assert trace.winning_rule.rule_id == "RULE_GUARD_01"
    assert trace.recommendation.action.action_code == "REJECT_INPUT"
    assert trace.recommendation.is_safe_to_operate is False

