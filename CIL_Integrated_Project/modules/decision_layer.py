"""
MachineGuard Decision Layer & Maintenance Recommendations
Translates raw model probabilities and local feature attributions into
transparent operational risk tiers, evidentiary justifications, and actionable maintenance protocols.
"""

from typing import Dict, List, Optional, Any
import numpy as np

def assess_operational_decision(
    inference_result: Dict[str, Any],
    cost_fp: float = 150.0,    # Estimated cost of unnecessary manual inspection ($)
    cost_fn: float = 3500.0   # Estimated cost of catastrophic in-service equipment failure ($)
) -> Dict[str, Any]:
    """
    Transforms model prediction into a complete industrial maintenance recommendation:
    - Risk Tier (NOMINAL, ELEVATED, CRITICAL)
    - Primary Evidence / Contributing Features
    - Actionable Inspection Protocol
    - Safety Caveats & Limitations
    """
    proba = inference_result["failure_probability"]
    threshold = inference_result["operating_threshold"]
    inputs = inference_result["input_features"]
    contribs = inference_result.get("feature_contributions", {})

    # 1. Determine Risk Tier
    if proba >= 0.70:
        risk_tier = "CRITICAL"
        risk_color = "#DC2626"  # Red
        urgency = "IMMEDIATE (Within 2 Hours)"
        primary_action = "Initiate Controlled Equipment Shutdown and Urgent Mechanical Inspection"
    elif proba >= threshold:
        risk_tier = "ELEVATED"
        risk_color = "#F59E0B"  # Amber
        urgency = "HIGH (Within 24 Hours / Next Scheduled Shift)"
        primary_action = "Schedule Targeted Sensor Diagnostic & Component Wear Inspection"
    elif proba >= (threshold * 0.5):
        risk_tier = "MONITORING"
        risk_color = "#3B82F6"  # Blue
        urgency = "ROUTINE (Next Maintenance Cycle)"
        primary_action = "Maintain Normal Operation with Enhanced Telemetry Logging"
    else:
        risk_tier = "NOMINAL"
        risk_color = "#10B981"  # Green
        urgency = "NORMAL"
        primary_action = "Maintain Standard Operating Schedule"

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
