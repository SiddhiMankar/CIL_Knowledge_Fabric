"""
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

    return {
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "urgency": urgency,
        "primary_action": primary_action,
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
            "Predictions are statistical estimates derived from historical AI4I synthetic dataset patterns.",
            "Model feature importance indicates predictive correlation, not definitive mechanical causality.",
            "All recommendations serve as engineering decision-support and must be verified by a qualified technician prior to executing maintenance."
        ]
    }
