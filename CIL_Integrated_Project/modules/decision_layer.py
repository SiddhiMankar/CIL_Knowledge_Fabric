"""
<<<<<<< HEAD
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
=======
MachineGuard Decision Layer & Maintenance Recommendations
Translates raw model probabilities and local feature attributions into
transparent operational risk tiers, evidentiary justifications, and actionable maintenance protocols.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import config

def classify_risk_tier(
    failure_probability: float,
    threshold: float = config.DEFAULT_OPERATING_THRESHOLD
) -> str:
    """
    Authoritative risk tier classification function used across all single and batch workflows.
    Four-tier scheme:
    - CRITICAL:   P(Fail) >= 0.70
    - ELEVATED:   threshold <= P(Fail) < 0.70
    - MONITORING: 0.5 * threshold <= P(Fail) < threshold
    - NOMINAL:    P(Fail) < 0.5 * threshold
    """
    if threshold is None or not (0.0 <= threshold <= 1.0) or np.isnan(threshold):
        raise ValueError(f"Operating threshold must be between 0.0 and 1.0. Got: {threshold}")

    if failure_probability is None or np.isnan(failure_probability) or np.isinf(failure_probability):
        raise ValueError(f"Invalid failure probability: {failure_probability}")

    p = float(failure_probability)
    # Clamp to [0.0, 1.0] to protect against floating precision artifacts
    p = max(0.0, min(1.0, p))

    if p >= config.CRITICAL_PROBABILITY_THRESHOLD:
        return "CRITICAL"
    elif p >= threshold:
        return "ELEVATED"
    elif p >= (threshold * 0.5):
        return "MONITORING"
    else:
        return "NOMINAL"


def get_risk_tier_metadata(risk_tier: str) -> Dict[str, str]:
    """Returns color, urgency level, and primary action for an operational risk tier."""
    tier_meta = {
        "CRITICAL": {
            "risk_color": "#DC2626",  # Red
            "urgency": "IMMEDIATE (Within 2 Hours)",
            "primary_action": "Initiate Controlled Equipment Shutdown and Urgent Mechanical Inspection"
        },
        "ELEVATED": {
            "risk_color": "#F59E0B",  # Amber
            "urgency": "HIGH (Within 24 Hours / Next Scheduled Shift)",
            "primary_action": "Schedule Targeted Sensor Diagnostic & Component Wear Inspection"
        },
        "MONITORING": {
            "risk_color": "#3B82F6",  # Blue
            "urgency": "ROUTINE (Next Maintenance Cycle)",
            "primary_action": "Maintain Normal Operation with Enhanced Telemetry Logging"
        },
        "NOMINAL": {
            "risk_color": "#10B981",  # Green
            "urgency": "NORMAL",
            "primary_action": "Maintain Standard Operating Schedule"
        }
    }
    return tier_meta.get(risk_tier, tier_meta["NOMINAL"])


def assess_operational_decision(
    inference_result: Dict[str, Any],
    cost_fp: float = 150.0,    # Estimated cost of unnecessary manual inspection ($)
    cost_fn: float = 3500.0   # Estimated cost of catastrophic in-service equipment failure ($)
) -> Dict[str, Any]:
    """
    Transforms model prediction into a complete industrial maintenance recommendation:
    - Risk Tier (NOMINAL, MONITORING, ELEVATED, CRITICAL)
    - Primary Evidence / Contributing Features
    - Actionable Inspection Protocol
    - Safety Caveats & Limitations
    """
    proba = inference_result["failure_probability"]
    threshold = inference_result.get("operating_threshold", config.DEFAULT_OPERATING_THRESHOLD)
    inputs = inference_result["input_features"]
    contribs = inference_result.get("feature_contributions", {})

    # 1. Determine Risk Tier via authoritative classification
    risk_tier = classify_risk_tier(proba, threshold)
    meta = get_risk_tier_metadata(risk_tier)
    risk_color = meta["risk_color"]
    urgency = meta["urgency"]
    primary_action = meta["primary_action"]

    # 2. Extract Leading Risk Factors based on exact local feature contributions
    # Positive contribution increases failure probability
    positive_contribs = {k: v for k, v in contribs.items() if v > 0}
    sorted_factors = sorted(positive_contribs.items(), key=lambda x: x[1], reverse=True)
    
    top_factors = []
    for feat, delta in sorted_factors[:3]:
        raw_val = inputs.get(feat, inputs.get(feat.split("_")[0], "N/A"))
        top_factors.append({
            "feature": feat,
            "contribution_delta": delta,
            "current_value": raw_val,
            "impact": f"+{delta*100:.1f}% to failure risk score"
        })

    # 3. Physics & Domain Reasoning (Connecting sensor readings to mechanical failure modes)
    evidence_notes = []
    recommended_checks = []

    tool_wear = inputs.get("Tool wear [min]", 0)
    torque = inputs.get("Torque [Nm]", 0)
    rpm = inputs.get("Rotational speed [rpm]", 0)
    air_temp = inputs.get("Air temperature [K]", 0)
    proc_temp = inputs.get("Process temperature [K]", 0)
    temp_diff = proc_temp - air_temp

    if tool_wear >= 200:
        evidence_notes.append(f"Tool wear time is critically high ({tool_wear} min, nearing mechanical fatigue limit).")
        recommended_checks.append("Inspect cutting tool tip/bit for chipping, catastrophic wear, or plastic deformation (TWF risk).")
    elif tool_wear >= 160:
        evidence_notes.append(f"Tool wear time is elevated ({tool_wear} min).")
        recommended_checks.append("Verify tool sharpness and gauge remaining usable cutting life.")

    if torque >= 60.0 or rpm <= 1300:
        evidence_notes.append(f"Abnormal power/load profile: High torque ({torque:.1f} Nm) and/or low RPM ({rpm:.0f} rpm).")
        recommended_checks.append("Check spindle motor load and gearbox lubrication; test for overstrain (OSF) or power failure (PWF).")
    elif rpm >= 2300 and torque <= 20.0:
        evidence_notes.append(f"High rotational speed ({rpm:.0f} rpm) under very low load ({torque:.1f} Nm).")
        recommended_checks.append("Check drive belt tension and rotational speed stability.")

    if temp_diff <= 8.6 and rpm <= 1380:
        evidence_notes.append(f"Heat dissipation deficiency: Low differential between process and ambient air ({temp_diff:.1f} K).")
        recommended_checks.append("Check heat exchanger coolant flow and radiator surfaces for heat dissipation failure (HDF risk).")

    if not recommended_checks:
        recommended_checks.append("Perform standard visual and lubrication inspection per SOP.")

    # 4. Synthesize AI Planning Connection
    # Connects with modules.planning if bearing/tool replacement is indicated
    suggested_planning_plan = None
    if risk_tier in ("CRITICAL", "ELEVATED"):
        suggested_planning_plan = [
            "1. Lock out / Tag out machine electrical feed.",
            "2. Dismantle tool housing and inspect spindle / bearing wear.",
            "3. Replace worn tooling / bearing components.",
            "4. Execute cold spindle rotation and vibration calibration test.",
            "5. Monitor thermal equilibration at nominal load.",
            "6. Sign off work order and restore equipment to operational line."
        ]

    # 5. Calculate Cost-Benefit Tradeoff
    expected_cost_inaction = proba * cost_fn
    expected_cost_action = cost_fp

    m_name = inference_result.get("model_name", "Random Forest Classifier")
    m_type = inference_result.get("model_type", "ML")

    return {
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "urgency": urgency,
        "primary_action": primary_action,
        "model_name": m_name,
        "model_type": m_type,
        "top_contributing_factors": top_factors,
        "domain_evidence": evidence_notes,
        "recommended_checks": recommended_checks,
        "suggested_maintenance_plan": suggested_planning_plan,
        "economic_context": {
            "expected_cost_inaction": round(expected_cost_inaction, 2),
            "cost_preventive_inspection": round(expected_cost_action, 2),
            "recommendation": "Perform preventive inspection" if expected_cost_inaction > expected_cost_action else "Routine monitoring"
        },
        "limitations": [
            f"Predictions generated by {m_name} ({m_type}) based on AI4I synthetic dataset patterns.",
            "Feature importance and attributions indicate statistical model influence, not physical mechanical causation.",
            "All recommendations serve as engineering decision-support heuristics and require physical verification prior to maintenance execution."
        ]
    }
>>>>>>> 67c25a8ec6c7c96a556a72bd95d31bda6d5f2cac

