"""
Task 2 Module: Rule-Based Inference Engine
MachineGuard Decision Architecture

Implements:
- Structured Forward-Chaining Rule-Based Inference Engine.
- Declarative Rule definitions with Rule ID, Name, Condition Predicate,
  Action/Conclusion, Priority, Explanation, and Domain Category.
- Transparent Inference Trace recording all evaluated facts, fired rules,
  conflict resolution steps, and final recommendations.
- Explicit epistemic distinction between statistical model probability
  and deterministic physical failure certainty.
- Dynamic scenario testing and conflict resolution for UI and automated verification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import datetime

from modules.knowledge_representation import (
    MachineType,
    FailureMode,
    ActionSeverity,
    MaintenanceAction,
    MaintenanceRecommendation,
    PredictedFailureRisk,
    validate_machine_input
)


# ============================================================================
# 1. STRUCTURED RULE DATA STRUCTURE
# ============================================================================

@dataclass
class Rule:
    """
    Structured Rule definition for forward chaining.
    """
    rule_id: str
    name: str
    category: str
    priority: int
    condition: Callable[[Dict[str, Any]], bool]
    action_code: str
    action_title: str
    action_severity: ActionSeverity
    recommended_timeframe: str
    is_safe_to_operate: bool
    estimated_duration_min: int
    explanation: str
    condition_description: str

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """Evaluates rule condition against the working memory facts."""
        try:
            return bool(self.condition(facts))
        except Exception:
            return False


# ============================================================================
# 2. INFERENCE TRACE RECORD
# ============================================================================

@dataclass
class RuleEvaluationRecord:
    """Audit record for an individual rule evaluation."""
    rule_id: str
    name: str
    category: str
    priority: int
    fired: bool
    explanation: str
    condition_description: str


@dataclass
class InferenceTrace:
    """
    Transparent execution log documenting the complete reasoning chain.
    Ensures zero black-box decision-making.
    """
    timestamp: str
    input_facts: Dict[str, Any]
    model_prediction: Optional[Dict[str, Any]]
    epistemic_disclaimer: str
    evaluations: List[RuleEvaluationRecord] = field(default_factory=list)
    fired_rules: List[RuleEvaluationRecord] = field(default_factory=list)
    winning_rule: Optional[RuleEvaluationRecord] = None
    conflict_resolution_rationale: str = ""
    recommendation: Optional[MaintenanceRecommendation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "input_facts": self.input_facts,
            "model_prediction": self.model_prediction,
            "epistemic_disclaimer": self.epistemic_disclaimer,
            "total_rules_evaluated": len(self.evaluations),
            "fired_rules_count": len(self.fired_rules),
            "fired_rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "priority": r.priority,
                    "explanation": r.explanation
                }
                for r in self.fired_rules
            ],
            "winning_rule_id": self.winning_rule.rule_id if self.winning_rule else None,
            "conflict_resolution_rationale": self.conflict_resolution_rationale,
            "recommendation": self.recommendation.to_dict() if self.recommendation else None
        }


# ============================================================================
# 3. RULE BASE SPECIFICATION
# ============================================================================

def build_default_rule_base() -> List[Rule]:
    """
    Constructs the standard industrial predictive-maintenance rule base.
    Derives condition thresholds from the AI4I 2020 domain documentation:
    - Temperature difference <= 8.6 K and Speed < 1380 rpm -> Heat Dissipation Failure (HDF)
    - Power P = (2*pi/60)*Speed*Torque outside [3500, 9000] W -> Power Failure (PWF)
    - Torque * Tool wear >= 11000 (L), 12000 (M), 13000 (H) -> Overstrain Failure (OSF)
    - Tool wear >= 200 min -> Tool Wear Failure (TWF)
    - Statistical risk thresholds: High (>=0.70), Moderate (0.35 - 0.70), Low (<0.35).
    """
    rules: List[Rule] = [
        # Guard Rule: Invalid Data
        Rule(
            rule_id="RULE_GUARD_01",
            name="Input Data Integrity Guard",
            category="Input Validation",
            priority=999,
            condition=lambda f: not f.get("is_valid_input", True),
            action_code="REJECT_INPUT",
            action_title="Reject Assessment and Request Correct Sensor Telemetry",
            action_severity=ActionSeverity.CRITICAL,
            recommended_timeframe="Immediate (Sensor Recalibration)",
            is_safe_to_operate=False,
            estimated_duration_min=0,
            explanation="Input telemetry violates physical bounds or contains missing attributes; automated recommendation is aborted to prevent unsafe advice.",
            condition_description="Input data validation check failed (is_valid_input == False)"
        ),

        # Physical Rule: Heat Dissipation Strain (HDF Mechanism)
        Rule(
            rule_id="RULE_PHYS_HDF_01",
            name="Heat Dissipation Thermal Strain",
            category="Thermodynamic Conditions",
            priority=90,
            condition=lambda f: (
                f.get("temperature_difference_k", 99.0) <= 8.6 and
                f.get("Rotational speed [rpm]", 9999.0) < 1380.0
            ),
            action_code="INSPECT_COOLING",
            action_title="Inspect Coolant Flow, Heat Exchanger and Thermal Boundary",
            action_severity=ActionSeverity.HIGH,
            recommended_timeframe="Within Current Shift (Within 2 Hours)",
            is_safe_to_operate=False,
            estimated_duration_min=45,
            explanation="Low rotational speed combined with minimal thermal dissipation delta (<= 8.6 K) induces thermal saturation and motor coil breakdown.",
            condition_description="(Process Temp - Air Temp <= 8.6 K) AND (Rotational Speed < 1380 rpm)"
        ),

        # Physical Rule: Mechanical Overstrain (OSF Mechanism)
        Rule(
            rule_id="RULE_PHYS_OSF_01",
            name="Mechanical Overstrain Threshold Exceeded",
            category="Mechanical Stress",
            priority=88,
            condition=lambda f: (
                f.get("Torque [Nm]", 0.0) * f.get("Tool wear [min]", 0.0) >= (
                    11000.0 if f.get("Type") == "L" else (
                        12000.0 if f.get("Type") == "M" else 13000.0
                    )
                )
            ),
            action_code="REPLACE_TOOL_OVERSTRAIN",
            action_title="Halt Machining & Replace Severely Strained Tool Insert",
            action_severity=ActionSeverity.HIGH,
            recommended_timeframe="Immediate Tool Change",
            is_safe_to_operate=False,
            estimated_duration_min=30,
            explanation="The product of spindle torque and cutting duration exceeds structural load limits for this material variant, indicating impending tool fracture.",
            condition_description="Torque * Tool Wear >= [11000 (L) | 12000 (M) | 13000 (H)]"
        ),

        # Physical Rule: Extreme Tool Wear (TWF Mechanism)
        Rule(
            rule_id="RULE_PHYS_TWF_01",
            name="Tool Wear Life Limit Reached",
            category="Tool Wear",
            priority=85,
            condition=lambda f: f.get("Tool wear [min]", 0.0) >= 200.0,
            action_code="REPLACE_TOOL_WEAR",
            action_title="Scheduled Replacement of Worn Cutting Tool",
            action_severity=ActionSeverity.HIGH,
            recommended_timeframe="Before Starting Next Workpiece",
            is_safe_to_operate=False,
            estimated_duration_min=20,
            explanation="Tool wear has exceeded 200 minutes, operating within the historical tool fracture zone (200-240 min).",
            condition_description="Tool wear [min] >= 200.0 min"
        ),

        # Physical Rule: Power Envelope Strain (PWF Mechanism)
        Rule(
            rule_id="RULE_PHYS_PWF_01",
            name="Spindle Power Out of Safe Envelope",
            category="Electrical Power",
            priority=86,
            condition=lambda f: (
                f.get("mechanical_power_w", 5000.0) < 3500.0 or
                f.get("mechanical_power_w", 5000.0) > 9000.0
            ),
            action_code="INSPECT_DRIVE_MOTOR",
            action_title="Inspect Spindle Drive Motor and Electrical Inverter",
            action_severity=ActionSeverity.HIGH,
            recommended_timeframe="Inspect Within 4 Hours",
            is_safe_to_operate=False,
            estimated_duration_min=60,
            explanation="Calculated mechanical spindle power falls outside the safe operating envelope (<3500 W or >9000 W), indicating drive overload or decoupling.",
            condition_description="Mechanical Power < 3500 W OR Mechanical Power > 9000 W"
        ),

        # Physical Rule: High Speed Runaway / Torque Loss
        Rule(
            rule_id="RULE_PHYS_RUNAWAY_01",
            name="High Spindle Velocity Under Minimal Load",
            category="Mechanical Stress",
            priority=75,
            condition=lambda f: (
                f.get("Rotational speed [rpm]", 0.0) > 2500.0 and
                f.get("Torque [Nm]", 99.0) < 15.0
            ),
            action_code="CHECK_LUBRICATION",
            action_title="Check Spindle Bearings and Lubricant Level",
            action_severity=ActionSeverity.MEDIUM,
            recommended_timeframe="Inspect at End of Shift",
            is_safe_to_operate=True,
            estimated_duration_min=25,
            explanation="Spindle is operating at elevated velocity under near zero load, risking bearing overheating and lubricant starvation.",
            condition_description="Rotational speed > 2500 rpm AND Torque < 15 Nm"
        ),

        # Risk Rule: Critical Predictive Failure Risk
        Rule(
            rule_id="RULE_STAT_CRIT_01",
            name="Critical ML Failure Probability Alert",
            category="Statistical Risk",
            priority=95,
            condition=lambda f: f.get("failure_probability", 0.0) >= 0.70,
            action_code="URGENT_INSPECTION",
            action_title="Emergency Machine Halt & Diagnostic Health Audit",
            action_severity=ActionSeverity.CRITICAL,
            recommended_timeframe="Immediate (Halt Current Program)",
            is_safe_to_operate=False,
            estimated_duration_min=90,
            explanation="The trained predictive model reports a high failure probability (>= 70%). Immediate physical inspection is required before resuming machining.",
            condition_description="Model Failure Probability >= 0.70"
        ),

        # Risk Rule: Moderate Predictive Failure Risk
        Rule(
            rule_id="RULE_STAT_MOD_01",
            name="Moderate ML Failure Probability Warning",
            category="Statistical Risk",
            priority=60,
            condition=lambda f: 0.35 <= f.get("failure_probability", 0.0) < 0.70,
            action_code="SCHEDULED_MAINTENANCE",
            action_title="Schedule Preventative Inspection on Next Shift",
            action_severity=ActionSeverity.MEDIUM,
            recommended_timeframe="Schedule Within 12 Hours",
            is_safe_to_operate=True,
            estimated_duration_min=45,
            explanation="Model indicates elevated failure risk (35% to 70%). Machine may complete the current batch but requires scheduled preventative maintenance.",
            condition_description="0.35 <= Model Failure Probability < 0.70"
        ),

        # Baseline Rule: Normal Health State
        Rule(
            rule_id="RULE_STAT_LOW_01",
            name="Normal Equipment Operating Health",
            category="Baseline Health",
            priority=10,
            condition=lambda f: f.get("failure_probability", 0.0) < 0.35,
            action_code="ROUTINE_MONITORING",
            action_title="Continue Active Machining Under Routine Telemetry",
            action_severity=ActionSeverity.INFO,
            recommended_timeframe="Next Standard Scheduled PM Cycle",
            is_safe_to_operate=True,
            estimated_duration_min=0,
            explanation="All physical sensor parameters and predictive models confirm normal operating baseline. No immediate intervention required.",
            condition_description="Model Failure Probability < 0.35 and no physical trip conditions"
        )
    ]
    return rules


# ============================================================================
# 4. INFERENCE ENGINE CLASS
# ============================================================================

class RuleEngine:
    """
    Transparent Forward-Chaining Inference Engine.
    Evaluates domain rules, resolves conflicts by priority and specificity,
    and returns a traceable recommendation.
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = rules if rules is not None else build_default_rule_base()
        # Sort rule base descending by priority for deterministic evaluation
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def infer(
        self,
        sensor_data: Dict[str, Any],
        model_risk: Optional[PredictedFailureRisk] = None
    ) -> InferenceTrace:
        """
        Executes forward chaining inference:
        1. Validates input facts.
        2. Computes derived physical parameters.
        3. Integrates model prediction (with epistemic distinction).
        4. Evaluates all rules.
        5. Identifies fired rules and applies deterministic conflict resolution.
        6. Constructs and returns structured InferenceTrace.
        """
        timestamp = datetime.datetime.now().isoformat()

        # Step 1: Validate input
        val_result = validate_machine_input(sensor_data)
        facts = dict(sensor_data)
        facts["is_valid_input"] = val_result.is_valid

        # Step 2: Compute derived physical parameters if valid
        if val_result.is_valid:
            air_t = float(sensor_data["Air temperature [K]"])
            proc_t = float(sensor_data["Process temperature [K]"])
            speed = float(sensor_data["Rotational speed [rpm]"])
            torque = float(sensor_data["Torque [Nm]"])
            wear = float(sensor_data["Tool wear [min]"])

            facts["temperature_difference_k"] = round(proc_t - air_t, 2)
            facts["mechanical_power_w"] = round((2.0 * 3.141592653589793 / 60.0) * speed * torque, 2)
            facts["overstrain_index"] = round(torque * wear, 2)

        # Step 3: Integrate model risk if provided
        if model_risk is not None:
            facts["failure_probability"] = float(model_risk.failure_probability)
            facts["failure_predicted"] = bool(model_risk.failure_predicted)
            model_info = {
                "model_name": model_risk.model_name,
                "failure_predicted": model_risk.failure_predicted,
                "failure_probability": round(model_risk.failure_probability, 4),
                "confidence_level": model_risk.confidence_level
            }
        else:
            facts["failure_probability"] = 0.0
            facts["failure_predicted"] = False
            model_info = None

        epistemic_disclaimer = (
            "Epistemic Note: The ML probability is a statistical estimate of failure likelihood "
            "based on historical training distributions. It does NOT constitute physical certainty of failure. "
            "Rule-based recommendations combine model risk with physical engineering constraints."
        )

        # Step 4: Evaluate rules
        evaluations: List[RuleEvaluationRecord] = []
        fired_rules: List[Rule] = []

        for rule in self.rules:
            fired = rule.evaluate(facts)
            eval_record = RuleEvaluationRecord(
                rule_id=rule.rule_id,
                name=rule.name,
                category=rule.category,
                priority=rule.priority,
                fired=fired,
                explanation=rule.explanation,
                condition_description=rule.condition_description
            )
            evaluations.append(eval_record)
            if fired:
                fired_rules.append(rule)

        # Step 5: Conflict resolution
        # Fired rules are already sorted by priority (descending).
        # The highest priority fired rule is selected as the primary winning conclusion.
        fired_records = [e for e in evaluations if e.fired]

        if not fired_rules:
            # Fallback if no rule fired
            winning_rule = None
            conflict_rationale = "No rule fired. Fallback to default monitoring."
            rec_action = MaintenanceAction(
                action_code="ROUTINE_MONITORING",
                title="Continue Active Machining Under Routine Monitoring",
                description="No specific rule condition triggered.",
                estimated_duration_min=0,
                severity=ActionSeverity.INFO
            )
            recommendation = MaintenanceRecommendation(
                action=rec_action,
                reasoning="Standard default baseline: no failure conditions detected.",
                triggered_rules=[],
                is_safe_to_operate=True,
                recommended_timeframe="Standard PM Cycle",
                model_risk_score=facts["failure_probability"]
            )
            winning_record = None
        else:
            winning_rule = fired_rules[0]
            winning_record = fired_records[0]

            if len(fired_rules) > 1:
                conflict_rationale = (
                    f"Conflict Resolution: {len(fired_rules)} rules fired simultaneously. "
                    f"Selected '{winning_rule.rule_id}: {winning_rule.name}' based on highest priority "
                    f"(Priority {winning_rule.priority} vs {', '.join(str(r.priority) for r in fired_rules[1:])})."
                )
            else:
                conflict_rationale = f"Single rule fired unambiguously: {winning_rule.rule_id}."

            rec_action = MaintenanceAction(
                action_code=winning_rule.action_code,
                title=winning_rule.action_title,
                description=winning_rule.explanation,
                estimated_duration_min=winning_rule.estimated_duration_min,
                severity=winning_rule.action_severity
            )

            all_fired_ids = [r.rule_id for r in fired_rules]
            synthesized_reasoning = (
                f"{winning_rule.explanation} "
                + (f"Additional supporting conditions observed from: {', '.join(all_fired_ids[1:])}." if len(all_fired_ids) > 1 else "")
            )

            recommendation = MaintenanceRecommendation(
                action=rec_action,
                reasoning=synthesized_reasoning,
                triggered_rules=all_fired_ids,
                is_safe_to_operate=winning_rule.is_safe_to_operate,
                recommended_timeframe=winning_rule.recommended_timeframe,
                model_risk_score=facts["failure_probability"]
            )

        return InferenceTrace(
            timestamp=timestamp,
            input_facts=facts,
            model_prediction=model_info,
            epistemic_disclaimer=epistemic_disclaimer,
            evaluations=evaluations,
            fired_rules=fired_records,
            winning_rule=winning_record,
            conflict_resolution_rationale=conflict_rationale,
            recommendation=recommendation
        )

