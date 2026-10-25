"""
MachineGuard — Predictive Maintenance Intelligence
Industrial decision-support platform with user-selectable Machine Learning and Deep Learning models.
Built upon CIL Integrated Lab Tasks 3-8 with full artifact persistence and leakage exclusion.

Main Navigation Structure:
- 🛡️ Machine Assessment (Single telemetry, demonstration profiles, multi-model comparison, CSV batch upload)
- 📋 Assessment History (Auditable logs, model filtering, CSV export)
- 📊 Model Performance (Rigorous held-out evaluation across RF, GB, LR, and 1D CNN)
- 🔬 Experiments (Tasks 3–8) (Preserved interactive laboratory tasks: Planning, EDA, Classical ML, Regression, 1D CNN, Interpretability)
- ⚖️ About, Ethics & Limitations (Class imbalance, safety oversight, domain shift, synthetic data provenance)
"""

import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive Agg backend; Streamlit uses st.pyplot()
import matplotlib.pyplot as plt
import seaborn as sns

import config
from modules import (
    planning,
    eda,
    classical_ml,
    regression,
    cnn_model,
    interpretability,
    inference,
    decision_layer,
    history_manager,
    model_registry
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MachineGuard — Predictive Maintenance Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load metadata
@st.cache_data
def load_metadata():
    if config.METADATA_PATH.exists():
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Load raw dataset if present
@st.cache_data
def load_raw_dataset():
    found, path = config.locate_dataset()
    if found and path is not None:
        return pd.read_csv(path)
    return None

metadata = load_metadata()
raw_df = load_raw_dataset()
registry = model_registry.get_model_registry()
hist_mgr = history_manager.get_history_manager()

# Sidebar Brand & Navigation
st.sidebar.markdown(
    """
    <div style="padding: 10px 0; border-bottom: 1px solid #334155;">
        <h2 style="margin: 0; color: #38BDF8; font-size: 1.4rem;">🛡️ MachineGuard</h2>
        <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.8rem; font-weight: 500;">
            Predictive Maintenance Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🛡️ Machine Assessment",
        "📋 Assessment History",
        "📊 Model Performance",
        "🔬 Experiments (Tasks 3–8)",
        "⚖️ About, Ethics & Limitations"
    ],
    index=0
)

st.sidebar.markdown("---")
# Quick System Status in Sidebar
dataset_found, dataset_path = config.locate_dataset()
available_models = registry.list_models(only_available=True)
all_models = registry.list_models()

st.sidebar.markdown("**System Health & Model Registry:**")
st.sidebar.markdown(f"- **Available Models:** `{len(available_models)} / {len(all_models)} active`")
st.sidebar.markdown(f"- **Primary Default:** Random Forest (`v1.0.0`)")
st.sidebar.markdown(f"- **Telemetry Dataset:** `{'ai4i2020.csv (10k rows)' if dataset_found else 'Missing'}`")
st.sidebar.markdown(f"- **Host Runtime:** Python `{sys.version.split()[0]}`")

st.sidebar.markdown("---")
st.sidebar.caption("MachineGuard Intelligence • Decision Support v2.0.0")


# ============================================================================
# 1. MACHINE ASSESSMENT (PRIMARY WORKFLOW)
# ============================================================================
if nav_selection == "🛡️ Machine Assessment":
    st.title("🛡️ Machine Assessment & Risk Intelligence")
    st.markdown(
        "Evaluate operating machine sensor telemetry in real time. **Explicitly select which ML or DL model to use**, "
        "review physical inputs, execute inference against persisted artifacts without retraining, and view model-specific "
        "probabilities, risk tiers, and validated explanations."
    )

    # Model selection catalog options
    model_display_map = {
        "Random Forest Classifier (ML — Production Default)": "random_forest",
        "Gradient Boosting Classifier (ML)": "gradient_boosting",
        "Logistic Regression (ML — Balanced)": "logistic_regression",
        "1D Convolutional Neural Network (DL — PyTorch)": "cnn_1d"
    }

    # Model Selection UI Section
    st.markdown("### 🤖 Select Prediction Model")
    st.caption("Choose the machine learning or deep learning architecture that will execute inference for this assessment:")

    col_m_select, col_m_card = st.columns([1, 1])

    with col_m_select:
        selected_display_label = st.selectbox(
            "Active Prediction Model:",
            options=list(model_display_map.keys()),
            index=0,
            help="Select the model architecture to perform actual telemetry inference."
        )
        selected_model_id = model_display_map[selected_display_label]
        model_info = registry.get_model_info(selected_model_id)

    with col_m_card:
        # Informational Model Card
        badge_color = "#3B82F6" if model_info["model_type"] == "ML" else "#8B5CF6"
        st.markdown(
            f"""
            <div style="background-color: #0F172A; border-left: 4px solid {badge_color}; padding: 12px; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #F8FAFC; font-size: 1.05rem;">{model_info['display_name']}</strong>
                    <span style="background-color: {badge_color}; color: #FFFFFF; font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: bold;">
                        {model_info['model_type']}
                    </span>
                </div>
                <p style="margin: 6px 0 4px 0; color: #CBD5E1; font-size: 0.85rem;">
                    <strong>Architecture:</strong> {model_info['description']}
                </p>
                <p style="margin: 2px 0 4px 0; color: #94A3B8; font-size: 0.82rem;">
                    <strong>Predicts:</strong> {model_info['prediction_task']}
                </p>
                <p style="margin: 2px 0 0 0; color: #F59E0B; font-size: 0.80rem;">
                    <strong>⚠️ Known Limitation:</strong> {model_info['limitations']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Check if selected model is available
    if not model_info["is_available"]:
        st.error(f"❌ Model Artifact Unavailable: {model_info['load_error']}")
        if dataset_found:
            if st.button(f"🚀 Train & Persist Artifacts for {model_info['display_name']}"):
                with st.spinner("Training models on data/raw/ai4i2020.csv..."):
                    from modules import model_trainer
                    model_trainer.train_and_persist_pipeline()
                    st.success("Artifacts generated! Reloading page...")
                    st.rerun()
        else:
            st.info("Place `ai4i2020.csv` in `data/raw/` and run `python -m modules.model_trainer` to build model artifacts.")

    # Instantiate the selected model's inference engine
    active_engine = registry.get_engine(selected_model_id)

    tab_single, tab_batch = st.tabs(["⚙️ Single Machine Assessment", "📂 Batch Assessment (CSV Upload)"])

    # --- TAB 1: SINGLE MACHINE ASSESSMENT ---
    with tab_single:
        col_form, col_quick = st.columns([2, 1])

        # Preset Demonstration Profiles
        with col_quick:
            st.markdown("#### 🧪 Demonstration Profiles")
            st.caption("Load verified test-set conditions to evaluate model decision dynamics:")
            preset_choice = st.selectbox(
                "Select Preset Profile:",
                [
                    "Custom Input",
                    "Normal Operation (Baseline Nominal)",
                    "Tool Wear Failure (Critical Wear)",
                    "Overstrain Risk (High Torque + Wear)",
                    "Heat Dissipation Risk (Thermal Compression)"
                ]
            )

            preset_data = {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 60.0
            }

            if preset_choice == "Normal Operation (Baseline Nominal)":
                preset_data = {"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551.0, "Torque [Nm]": 42.8, "Tool wear [min]": 5.0}
            elif preset_choice == "Tool Wear Failure (Critical Wear)":
                preset_data = {"Type": "L", "Air temperature [K]": 302.5, "Process temperature [K]": 311.8, "Rotational speed [rpm]": 1390.0, "Torque [Nm]": 52.0, "Tool wear [min]": 225.0}
            elif preset_choice == "Overstrain Risk (High Torque + Wear)":
                preset_data = {"Type": "L", "Air temperature [K]": 301.0, "Process temperature [K]": 310.5, "Rotational speed [rpm]": 1300.0, "Torque [Nm]": 68.0, "Tool wear [min]": 210.0}
            elif preset_choice == "Heat Dissipation Risk (Thermal Compression)":
                preset_data = {"Type": "H", "Air temperature [K]": 304.0, "Process temperature [K]": 311.5, "Rotational speed [rpm]": 1340.0, "Torque [Nm]": 58.0, "Tool wear [min]": 180.0}

            if preset_choice != "Custom Input":
                st.info(f"Loaded demo profile: **{preset_choice}**.")

        with col_form:
            st.markdown("#### 📥 Machine Telemetry Inputs")
            with st.form("machine_input_form"):
                machine_id_input = st.text_input("Machine Identifier / Asset Tag", value=f"CNC-CELL-{preset_data.get('Type', 'L')}01")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    type_val = st.selectbox(
                        "Machine Quality Variant (Type)",
                        options=["L", "M", "H"],
                        index=["L", "M", "H"].index(preset_data.get("Type", "L")),
                        help="Product variant: L = Low (60%), M = Medium (30%), H = High (10%)"
                    )
                    air_temp_val = st.number_input(
                        "Air Temperature [K]",
                        min_value=275.0, max_value=325.0,
                        value=float(preset_data.get("Air temperature [K]", 300.0)),
                        step=0.1,
                        help="Ambient temperature in Kelvin (Typical: 295 - 305 K)"
                    )
                    process_temp_val = st.number_input(
                        "Process Temperature [K]",
                        min_value=285.0, max_value=335.0,
                        value=float(preset_data.get("Process temperature [K]", 310.0)),
                        step=0.1,
                        help="Operating temperature in Kelvin (Typical: 305 - 315 K)"
                    )

                with col_f2:
                    speed_val = st.number_input(
                        "Rotational Speed [rpm]",
                        min_value=600.0, max_value=3200.0,
                        value=float(preset_data.get("Rotational speed [rpm]", 1500.0)),
                        step=10.0,
                        help="Spindle rotational speed (Typical: 1100 - 2900 rpm)"
                    )
                    torque_val = st.number_input(
                        "Torque [Nm]",
                        min_value=0.0, max_value=90.0,
                        value=float(preset_data.get("Torque [Nm]", 40.0)),
                        step=0.5,
                        help="Spindle torque in Newton-meters (Typical: 3 - 77 Nm)"
                    )
                    wear_val = st.number_input(
                        "Tool Wear [min]",
                        min_value=0.0, max_value=300.0,
                        value=float(preset_data.get("Tool wear [min]", 60.0)),
                        step=1.0,
                        help="Accumulated cutting tool service time in minutes (Typical: 0 - 253 min)"
                    )

                threshold_slider = st.slider(
                    f"Decision Threshold Sensitivity ({model_info['display_name']})",
                    min_value=0.10, max_value=0.90,
                    value=float(active_engine.operating_threshold),
                    step=0.05,
                    help=f"Configured operational threshold for {model_info['display_name']} is {model_info['operating_threshold']:.2f}."
                )

                submit_btn = st.form_submit_button(f"⚡ Run Prediction with {model_info['display_name']}", use_container_width=True)

        if submit_btn:
            raw_input_dict = {
                "Type": type_val,
                "Air temperature [K]": air_temp_val,
                "Process temperature [K]": process_temp_val,
                "Rotational speed [rpm]": speed_val,
                "Torque [Nm]": torque_val,
                "Tool wear [min]": wear_val
            }

            # 1. Validate
            is_valid, err_msg, validated_df = active_engine.validate_input(raw_input_dict)
            if not active_engine.is_loaded:
                st.error(f"❌ Cannot assess machine: {model_info['display_name']} is not loaded. {active_engine.load_error}")
            elif not is_valid:
                st.error(f"❌ Input Validation Error: {err_msg}")
            else:
                # 2. Run Inference using the explicitly selected model
                try:
                    res = active_engine.predict_single(raw_input_dict, custom_threshold=threshold_slider)
                    decision = decision_layer.assess_operational_decision(res)

                    # 3. Log into History with full model identification
                    hist_mgr.add_record(
                        machine_id=machine_id_input,
                        inputs=raw_input_dict,
                        inference_result=res,
                        decision_result=decision,
                        source=preset_choice if preset_choice != "Custom Input" else "Manual Form",
                        model_id=selected_model_id
                    )

                    # Store in session state for comparison and simulation continuity
                    st.session_state["last_assessment_input"] = raw_input_dict
                    st.session_state["last_assessment_result"] = res
                    st.session_state["last_assessment_decision"] = decision
                    st.session_state["last_machine_id"] = machine_id_input

                except Exception as ex:
                    st.error(f"Inference Failure for {model_info['display_name']}: {str(ex)}")

        # Display results if available in session state
        if "last_assessment_result" in st.session_state:
            res = st.session_state["last_assessment_result"]
            decision = st.session_state["last_assessment_decision"]
            raw_input_dict = st.session_state["last_assessment_input"]
            m_id_display = st.session_state.get("last_machine_id", "CNC-UNIT")

            st.markdown("---")
            st.subheader(f"📊 Assessment Outcome — {m_id_display}")

            # Active Model Callout Bar
            st.info(
                f"Inference Model: **{res['model_name']}** (`{res['model_type']}`) • "
                f"Model ID: `{res['model_id']}` • Operating Threshold: `{res['operating_threshold']:.2f}`"
            )

            # Header KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric(
                    label="Predicted Condition",
                    value="FAILURE RISK" if res["prediction"] == 1 else "NORMAL HEALTH",
                    delta=f"P(Fail): {res['failure_probability']*100:.1f}%",
                    delta_color="inverse" if res["prediction"] == 1 else "normal"
                )
            with k2:
                st.metric(
                    label="Risk Tier",
                    value=decision["risk_tier"],
                    delta=f"Urgency: {decision['urgency']}"
                )
            with k3:
                st.metric(
                    label="Operating Threshold",
                    value=f"{res['operating_threshold']:.2f}",
                    delta=f"Type: {res['model_type']}"
                )
            with k4:
                st.metric(
                    label="Cost Recommendation",
                    value=decision["economic_context"]["recommendation"],
                    delta=f"Exp Risk: ${decision['economic_context']['expected_cost_inaction']:.0f}"
                )

            # Primary Recommended Action Box
            if decision["risk_tier"] in ("CRITICAL", "ELEVATED"):
                st.error(f"🚨 **Required Action:** {decision['primary_action']}")
            elif decision["risk_tier"] == "MONITORING":
                st.warning(f"⚠️ **Recommended Action:** {decision['primary_action']}")
            else:
                st.success(f"✅ **Recommended Action:** {decision['primary_action']}")

            # Deep Dive Columns: Model-Specific Explainability
            col_expl, col_action = st.columns([1, 1])

            with col_expl:
                st.markdown(f"#### 🔍 Model-Specific Explainability ({res['model_name']})")

                if res.get("explanation_type") == "conv_activations":
                    st.caption("Intermediate Conv1D Layer 1 feature activations across feature dimensions:")
                    if res.get("conv_activations"):
                        c1 = res["conv_activations"]["conv1d_1"]
                        fig_act = interpretability.plot_activation_heatmap(c1, "Conv1D Layer 1 (32 Filters)")
                        st.pyplot(fig_act)
                    st.info(
                        "ℹ️ **Architecture Limitation:** Tree-path attribution does not apply to Convolutional Neural Networks. "
                        "Hierarchical layer activation maps and counterfactual sensitivity are presented instead."
                    )
                else:
                    expl_title = "Local Feature Contributions (Delta in P(Fail))" if "forest" in res['model_name'].lower() else "Feature Contributions (Log-Odds Impact)"
                    st.caption(f"{expl_title} for {res['model_name']}:")
                    fig_contrib = interpretability.plot_local_feature_contributions(
                        res["feature_contributions"],
                        title=f"Feature Contributions — {res['model_name']}"
                    )
                    st.pyplot(fig_contrib)

                    if decision["top_contributing_factors"]:
                        st.markdown("**Leading Failure Drivers:**")
                        for factor in decision["top_contributing_factors"]:
                            st.markdown(f"- **{factor['feature']}** (`{factor['current_value']}`): {factor['impact']}")

            with col_action:
                st.markdown("#### 🛠️ Physical Diagnosis & Maintenance Protocol")
                if decision["domain_evidence"]:
                    st.markdown("**Physical Anomalies Observed:**")
                    for note in decision["domain_evidence"]:
                        st.markdown(f"- {note}")

                st.markdown("**Actionable Inspection Protocol:**")
                for check in decision["recommended_checks"]:
                    st.markdown(f"- 🔧 {check}")

                if decision["suggested_maintenance_plan"]:
                    with st.expander("📋 View Structured Maintenance Execution Plan"):
                        for step in decision["suggested_maintenance_plan"]:
                            st.markdown(step)

                st.markdown("**⚠️ Model Confidence & Limitations:**")
                for lim in decision["limitations"]:
                    st.caption(f"• {lim}")

            # Counterfactual What-If Simulator (Using Active Selected Model)
            st.markdown("---")
            st.markdown(f"### 🎛️ Counterfactual What-If Simulator ({res['model_name']})")
            st.caption(f"Simulate sensitivity: Recompute inference dynamically through **{res['model_name']}** as physical sensor readings vary.")

            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                feature_to_sim = st.selectbox(
                    "Select Sensor to Vary:",
                    [
                        "Tool wear [min]",
                        "Torque [Nm]",
                        "Rotational speed [rpm]",
                        "Air temperature [K]",
                        "Process temperature [K]"
                    ],
                    key="what_if_sensor_select"
                )
                bounds_map = {
                    "Tool wear [min]": (0.0, 260.0),
                    "Torque [Nm]": (10.0, 80.0),
                    "Rotational speed [rpm]": (1100.0, 2800.0),
                    "Air temperature [K]": (295.0, 305.0),
                    "Process temperature [K]": (305.0, 315.0)
                }
                low_b, high_b = bounds_map[feature_to_sim]

            with sim_col2:
                fig_sim, _ = interpretability.plot_what_if_sensitivity(
                    engine=active_engine,
                    base_input=raw_input_dict,
                    feature_to_vary=feature_to_sim,
                    min_val=low_b,
                    max_val=high_b
                )
                st.pyplot(fig_sim)

            # Optional Multi-Model Comparison Section
            st.markdown("---")
            st.markdown("### 🔬 Multi-Model Comparison (Optional)")
            st.caption("Evaluate this exact telemetry profile against all available ML and DL architectures simultaneously to compare perspectives.")

            compare_enabled = st.checkbox("🔍 Run Side-by-Side Comparison Across All Models", value=False)
            if compare_enabled:
                with st.spinner("Evaluating telemetry across all candidate models..."):
                    comp_results = registry.compare_models(
                        model_ids=["random_forest", "gradient_boosting", "logistic_regression", "cnn_1d"],
                        input_data=raw_input_dict
                    )
                comp_df = pd.DataFrame(comp_results)
                st.dataframe(
                    comp_df[[
                        "model_name", "model_type", "predicted_outcome",
                        "failure_probability", "decision_threshold", "risk_tier", "status"
                    ]],
                    use_container_width=True
                )

                st.markdown(
                    """
                    **💡 Understanding Model Disagreements:**
                    - **Tree Ensembles (Random Forest & Gradient Boosting):** Capable of capturing sharp, non-linear interaction thresholds (e.g. torque spikes combined with high tool wear).
                    - **Linear Model (Logistic Regression):** Constrained to linear decision boundaries; useful for baseline interpretability but may miss compound mechanical faults.
                    - **Deep Learning (1D CNN):** Learns localized filter representations; does not assume monotonicity. Differences reflect architecture inductive biases on small tabular feature sets.
                    *No single model is universally correct on an isolated sample; offline held-out PR-AUC on rare failures (3.39%) provides the authoritative statistical benchmark.*
                    """
                )

    # --- TAB 2: BATCH CSV ASSESSMENT ---
    with tab_batch:
        st.markdown("#### 📂 Batch Telemetry Upload & Evaluation")
        st.markdown(
            "Upload a CSV containing operational machine records. MachineGuard validates each row, applies "
            "fitted preprocessing without retraining, and outputs risk tiers and probability scores using your chosen model."
        )

        col_b_m1, col_b_m2 = st.columns([1, 1])
        with col_b_m1:
            batch_model_label = st.selectbox(
                "Select Model for Batch Assessment:",
                options=list(model_display_map.keys()),
                index=0,
                key="batch_model_selector"
            )
            batch_model_id = model_display_map[batch_model_label]
            batch_model_info = registry.get_model_info(batch_model_id)
            batch_engine = registry.get_engine(batch_model_id)

        with col_b_m2:
            batch_th_slider = st.slider(
                f"Batch Operating Threshold ({batch_model_info['display_name']}):",
                min_value=0.10, max_value=0.90,
                value=float(batch_engine.operating_threshold),
                step=0.05,
                key="batch_threshold_slider"
            )

        uploaded_batch = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_uploader")

        col_sample, _ = st.columns([1, 2])
        with col_sample:
            if st.button("📥 Load 25 Unseen Held-Out Test Records for Demonstration"):
                test_artifact_path = config.MODELS_DIR / "test_set_split.joblib"
                if test_artifact_path.exists():
                    test_split = joblib.load(test_artifact_path)
                    sample_batch_df = test_split["X_test"].head(25).copy()
                    st.session_state["demo_batch_df"] = sample_batch_df
                else:
                    st.warning("Test split artifact not found. Please run training pipeline.")

        batch_to_process = None
        if uploaded_batch is not None:
            try:
                batch_to_process = pd.read_csv(uploaded_batch)
                st.success(f"Uploaded `{uploaded_batch.name}` ({len(batch_to_process)} records).")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
        elif "demo_batch_df" in st.session_state:
            batch_to_process = st.session_state["demo_batch_df"]
            st.info(f"Loaded {len(batch_to_process)} held-out demonstration records.")

        if batch_to_process is not None:
            st.dataframe(batch_to_process.head(5), use_container_width=True)

            if st.button(f"🚀 Execute Batch Inference with {batch_model_info['display_name']}"):
                if not batch_engine.is_loaded:
                    st.error(f"❌ Cannot execute batch inference: {batch_model_info['display_name']} artifact is not loaded.")
                else:
                    try:
                        with st.spinner(f"Processing batch inference using {batch_model_info['display_name']}..."):
                            batch_results = batch_engine.predict_batch_robust(batch_to_process, custom_threshold=batch_th_slider)

                        st.markdown("### 📈 Batch Assessment Summary")
                        b1, b2, b3, b4 = st.columns(4)
                        critical_count = (batch_results["Risk_Level"] == "CRITICAL").sum()
                        elevated_count = (batch_results["Risk_Level"] == "ELEVATED").sum()
                        monitoring_count = (batch_results["Risk_Level"] == "MONITORING").sum()
                        nominal_count = (batch_results["Risk_Level"] == "NOMINAL").sum()

                        with b1:
                            st.metric("Critical Risk", int(critical_count))
                        with b2:
                            st.metric("Elevated Risk", int(elevated_count))
                        with b3:
                            st.metric("Monitoring Needed", int(monitoring_count))
                        with b4:
                            st.metric("Nominal Health", int(nominal_count))

                        display_cols = [
                            c for c in [
                                "Selected_Model", "Model_Type", "Type", "Air temperature [K]", "Process temperature [K]",
                                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                                "Failure_Probability", "Risk_Level", "Prediction_Status"
                            ] if c in batch_results.columns
                        ]
                        st.dataframe(batch_results[display_cols], use_container_width=True)

                        # CSV Export
                        csv_data = batch_results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"📥 Export Batch Results for {batch_model_info['display_name']} (CSV)",
                            data=csv_data,
                            file_name=f"machineguard_batch_{batch_model_id}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Batch Processing Error: {str(e)}")


# ============================================================================
# 2. ASSESSMENT HISTORY
# ============================================================================
elif nav_selection == "📋 Assessment History":
    st.title("📋 Operational Assessment History")
    st.markdown(
        "Chronological log of machine assessments evaluated during active sessions and persisted to disk. "
        "Enables auditing, accountability, and temporal pattern tracking across different models."
    )

    col_filter, _ = st.columns([1, 1])
    with col_filter:
        model_filter_options = [
            "All Models",
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
            "Logistic Regression (Balanced)",
            "1D Convolutional Neural Network (CNN)"
        ]
        chosen_filter = st.selectbox("Filter History by Model:", model_filter_options, index=0)

    history_df = hist_mgr.get_history_dataframe(model_filter=chosen_filter)

    if history_df.empty:
        st.info("No assessments recorded matching the selected filter. Run an assessment on the **Machine Assessment** tab to populate.")
    else:
        col_stats1, col_stats2, col_stats3, col_clear = st.columns([1, 1, 1, 1])
        tier_col = "Risk Tier" if "Risk Tier" in history_df.columns else "Risk_Tier"
        with col_stats1:
            st.metric("Assessments Displayed", len(history_df))
        with col_stats2:
            critical_logged = (history_df[tier_col] == "CRITICAL").sum()
            st.metric("Critical Alerts", int(critical_logged))
        with col_stats3:
            elevated_logged = (history_df[tier_col] == "ELEVATED").sum()
            st.metric("Elevated Alerts", int(elevated_logged))
        with col_clear:
            if st.button("🗑️ Clear Stored History"):
                hist_mgr.clear_history()
                st.rerun()

        st.markdown("---")
        st.dataframe(history_df, use_container_width=True)

        # CSV Download
        csv_hist = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Audit History (CSV)",
            data=csv_hist,
            file_name="machineguard_audit_history.csv",
            mime="text/csv"
        )


# ============================================================================
# 3. MODEL PERFORMANCE & VALIDATION
# ============================================================================
elif nav_selection == "📊 Model Performance":
    st.title("📊 Model Performance & Leakage-Free Validation")
    st.markdown(
        "Rigorous held-out test evaluation on the AI4I 2020 predictive maintenance benchmark. "
        "All metrics reported are calculated strictly from an untouched test partition (2,000 samples, 20% stratified split) "
        "with complete target leakage exclusion."
    )

    mg_meta = metadata.get("machineguard_production", {})
    held_out = mg_meta.get("held_out_metrics", {})
    op_metrics = mg_meta.get("metrics_at_operating_threshold", {})
    all_models = mg_meta.get("all_model_evaluations", {})

    st.markdown("### 🔍 Select Model to Inspect")
    perf_model_choice = st.selectbox(
        "Select Model Architecture for Performance Details:",
        [
            "Random Forest Classifier (Production Default)",
            "Gradient Boosting Classifier",
            "Logistic Regression (Balanced)",
            "1D Convolutional Neural Network (CNN)",
            "Compare All Models (Benchmark Matrix)"
        ],
        index=0
    )

    name_mapping = {
        "Random Forest Classifier (Production Default)": "Random Forest",
        "Gradient Boosting Classifier": "Gradient Boosting",
        "Logistic Regression (Balanced)": "Logistic Regression (Balanced)",
        "1D Convolutional Neural Network (CNN)": "1D CNN"
    }

    if perf_model_choice in name_mapping:
        target_name = name_mapping[perf_model_choice]
        m_eval = all_models.get(target_name, {})

        st.subheader(f"Performance Overview: {perf_model_choice}")
        st.caption("Evaluated on 2,000 held-out test samples (68 failures, 1,932 normal units, stratified 80/20 split):")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("ROC-AUC", f"{m_eval.get('roc_auc', 0.0):.4f}")
        with c2:
            st.metric("PR-AUC", f"{m_eval.get('pr_auc', 0.0):.4f}")
        with c3:
            st.metric("F1-Score", f"{m_eval.get('f1_score', 0.0):.4f}")
        with c4:
            st.metric("Recall (Sensitivity)", f"{m_eval.get('recall', 0.0):.4f}")
        with c5:
            st.metric("Precision", f"{m_eval.get('precision', 0.0):.4f}")

        col_cm, col_desc = st.columns(2)
        with col_cm:
            st.markdown("##### Confusion Matrix (Held-Out Test Set)")
            cm_dict = m_eval.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
            cm_matrix = np.array([
                [cm_dict.get("tn", 0), cm_dict.get("fp", 0)],
                [cm_dict.get("fn", 0), cm_dict.get("tp", 0)]
            ])
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                        xticklabels=["Normal (0)", "Failure (1)"], yticklabels=["Normal (0)", "Failure (1)"])
            ax_cm.set_xlabel("Predicted Class", fontsize=9, fontweight="bold")
            ax_cm.set_ylabel("True Class", fontsize=9, fontweight="bold")
            ax_cm.set_title(f"Confusion Matrix — {target_name}", fontsize=10, fontweight="bold")
            fig_cm.tight_layout()
            st.pyplot(fig_cm)

        with col_desc:
            st.markdown("##### Architecture Evaluation Notes")
            if target_name == "Random Forest":
                st.markdown(
                    """
                    - **Production Default:** Chosen as production model because of its balanced PR-AUC (`0.7305`) on the imbalanced failure class.
                    - **Threshold Optimization:** At threshold `0.35`, recall increases to `76.5%` reducing missed breakdowns from 25 to 16.
                    - **Interpretability:** Fully supports exact tree-path probability attribution.
                    """
                )
            elif target_name == "Gradient Boosting":
                st.markdown(
                    """
                    - **Highest PR-AUC:** Achieves highest PR-AUC (`0.8080`) and highest precision (`90.6%`) at standard threshold.
                    - **Sequential Boosting:** 100 shallow trees minimize residual classification loss.
                    - **Consideration:** Can be slightly more sensitive to extreme sensor outliers than bagging ensembles.
                    """
                )
            elif target_name == "Logistic Regression (Balanced)":
                st.markdown(
                    """
                    - **High Sensitivity Baseline:** Balanced class weighting gives high recall (`82.4%`) but lower precision (`14.1%`).
                    - **Linear Separation:** Provides clear linear odds ratios, but fails to capture non-linear sensor coupling.
                    - **Recommended Role:** Baseline transparent comparator for non-linear models.
                    """
                )
            elif target_name == "1D CNN":
                st.markdown(
                    """
                    - **Deep Learning Benchmark:** Achieves high ROC-AUC (`0.9604`) and high sensitivity (`92.7%` recall).
                    - **Convolutional Abstractions:** 2 Conv1D layers extract multi-channel filter representations.
                    - **Tabular Constraint:** Without spatial locality in tabular columns, tree ensembles retain superior precision/PR-AUC.
                    """
                )

    else:
        # Comparison Matrix across All Models
        st.subheader("Benchmark Comparison: All Candidate Models (Held-Out Test Set)")
        st.caption("All models trained and tested on the exact same stratified 80/20 train/test split (8,000 train / 2,000 test):")

        if all_models:
            comp_rows = []
            for name, m_metrics in all_models.items():
                cm = m_metrics.get("confusion_matrix", {})
                comp_rows.append({
                    "Model": name,
                    "ROC-AUC": m_metrics.get("roc_auc"),
                    "PR-AUC": m_metrics.get("pr_auc"),
                    "F1-Score": m_metrics.get("f1_score"),
                    "Recall": m_metrics.get("recall"),
                    "Precision": m_metrics.get("precision"),
                    "Accuracy": m_metrics.get("accuracy"),
                    "TP": cm.get("tp"),
                    "FP": cm.get("fp"),
                    "FN": cm.get("fn"),
                    "TN": cm.get("tn")
                })
            df_comp = pd.DataFrame(comp_rows).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
            st.dataframe(df_comp, use_container_width=True)

            # Bar plot of PR-AUC and ROC-AUC
            fig_bar, ax_bar = plt.subplots(figsize=(8, 3.8))
            x = np.arange(len(df_comp))
            width = 0.35
            ax_bar.bar(x - width/2, df_comp["PR-AUC"], width, label="PR-AUC (Precision-Recall)", color="#2563EB")
            ax_bar.bar(x + width/2, df_comp["ROC-AUC"], width, label="ROC-AUC", color="#10B981")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(df_comp["Model"], rotation=15, ha="right", fontsize=9)
            ax_bar.set_ylabel("Metric Value", fontsize=10)
            ax_bar.set_title("PR-AUC vs ROC-AUC Across Architectures (Held-Out Test Set)", fontsize=11, fontweight="bold")
            ax_bar.set_ylim(0.0, 1.05)
            ax_bar.legend()
            ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)

    st.markdown("---")
    st.subheader("Subgroup Reliability across Machine Quality Types (L, M, H)")
    st.caption("Verifying performance consistency across product variants in held-out test data:")
    subgroups = mg_meta.get("subgroup_performance", {})
    if subgroups:
        sub_rows = []
        for grp, gm in subgroups.items():
            sub_rows.append({
                "Machine Variant": grp,
                "Sample Count": gm["sample_count"],
                "Actual Failures": gm["failure_count"],
                "Failure Rate": f"{gm['failure_rate']*100:.2f}%",
                "Recall (Safety)": gm["recall"],
                "Precision": gm["precision"],
                "F1 Score": gm["f1_score"]
            })
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)


# ============================================================================
# 4. EXPERIMENTS (PRESERVED TASKS 3–8)
# ============================================================================
elif nav_selection == "🔬 Experiments (Tasks 3–8)":
    st.title("🔬 Laboratory Experiments & Research Tasks")
    st.markdown(
        "Interactive exploration of foundational laboratory experiments from coursework Tasks 3 through 8. "
        "These experimental modules are retained for academic evaluation, research reproduction, and algorithmic comparison."
    )

    exp_choice = st.selectbox(
        "Select Experiment Domain:",
        [
            "🧭 Task 3: AI Maintenance Planner (State Space Search)",
            "📊 Task 4: Exploratory Data Analysis & Preprocessing",
            "🤖 Task 5: Classical ML & Unsupervised Clustering",
            "📈 Task 6: Tool Wear Regression Diagnostics",
            "🧠 Task 7: 1D CNN Deep Learning Architecture",
            "🔍 Task 8: Model Interpretability & Layer Activations"
        ]
    )

    # TASK 3
    if "Task 3" in exp_choice:
        st.subheader("🧭 Task 3: State Space Search & Maintenance Planning")
        st.markdown(
            "Synthesizes discrete action planning sequences for industrial equipment maintenance (Pump P-101 bearing failure). "
            "Evaluates state transitions and compares BFS, DFS, and A* search algorithms."
        )

        col_scen, col_run = st.columns([2, 1])
        with col_scen:
            scenario_name = st.selectbox("Select Maintenance Scenario:", list(planning.SCENARIOS.keys()))
            scenario = planning.SCENARIOS[scenario_name]
            st.info(f"**Equipment:** {scenario['equipment']} | **Fault:** {scenario['fault']}\n\n{scenario['description']}")

        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)
            run_planner_btn = st.button("🚀 Run Planning Benchmark Suite", use_container_width=True)

        if run_planner_btn:
            results_table, details = planning.run_benchmarks_on_scenario(scenario)
            st.dataframe(results_table, use_container_width=True)

            tab_a, tab_b, tab_d = st.tabs(["A* Optimal Plan", "BFS Plan", "DFS Plan"])
            with tab_a:
                st.markdown("**A* Search Optimal Execution Plan:**")
                for i, step in enumerate(details["A*"]["path"], 1):
                    st.markdown(f"{i}. `{step}`")
            with tab_b:
                st.markdown("**BFS Execution Plan:**")
                for i, step in enumerate(details["BFS"]["path"], 1):
                    st.markdown(f"{i}. `{step}`")
            with tab_d:
                st.markdown("**DFS Execution Plan:**")
                for i, step in enumerate(details["DFS"]["path"], 1):
                    st.markdown(f"{i}. `{step}`")

    # TASK 4
    elif "Task 4" in exp_choice:
        st.subheader("📊 Task 4: Exploratory Data Analysis & Preprocessing")
        if raw_df is not None:
            summary = eda.get_dataset_summary(raw_df)
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                st.metric("Total Records", summary["rows"])
            with col_e2:
                st.metric("Feature Count", summary["columns"])
            with col_e3:
                st.metric("Missing Values", summary["total_missing"])
            with col_e4:
                st.metric("Duplicates", summary["duplicates"])

            st.markdown("#### Physical Sensor Feature Distributions")
            col_hist, col_box = st.columns(2)
            with col_hist:
                fig_hist = eda.plot_feature_histograms(raw_df)
                st.pyplot(fig_hist)
            with col_box:
                fig_box = eda.plot_boxplots(raw_df)
                st.pyplot(fig_box)

            st.markdown("#### Sensor Correlation Heatmap")
            fig_corr = eda.plot_correlation_matrix(raw_df)
            st.pyplot(fig_corr)
        else:
            st.warning("ai4i2020.csv not found. Place dataset in data/raw/ to view live EDA.")

    # TASK 5
    elif "Task 5" in exp_choice:
        st.subheader("🤖 Task 5: Classical Machine Learning & Clustering")
        if raw_df is not None:
            tab_ml1, tab_ml2 = st.tabs(["Supervised Classification", "Unsupervised Clustering"])
            with tab_ml1:
                st.markdown("#### Random Forest Feature Importance (MDI)")
                X, y, le = classical_ml.prepare_classification_data(raw_df)
                imp_df, fig_imp = classical_ml.compute_feature_importance(X, y)
                col_i1, col_i2 = st.columns([1, 1])
                with col_i1:
                    st.pyplot(fig_imp)
                with col_i2:
                    st.dataframe(imp_df, use_container_width=True)

            with tab_ml2:
                st.markdown("#### K-Means Clustering (Torque vs Rotational Speed)")
                X_clust = X[["Rotational speed [rpm]", "Torque [Nm]"]].copy()
                clusters, kmeans_obj, fig_k = classical_ml.run_kmeans_clustering(X_clust, n_clusters=3)
                st.pyplot(fig_k)
        else:
            st.warning("ai4i2020.csv required to run Task 5 interactively.")

    # TASK 6
    elif "Task 6" in exp_choice:
        st.subheader("📈 Task 6: Tool Wear Linear Regression Diagnostics")
        if raw_df is not None:
            X_reg, y_reg = regression.prepare_regression_data(raw_df)
            X_tr, X_te, y_tr, y_te = regression.split_regression_data(X_reg, y_reg)
            res_reg = regression.train_and_evaluate_regression(X_tr, y_tr, X_te, y_te)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("MSE", f"{res_reg['mse']:.4f}")
                st.metric("R² Score", f"{res_reg['r2']:.4f}")
                st.info(
                    "💡 **Engineering Finding:** R² is near 0.0, confirming that tool wear accumulation cannot "
                    "be predicted linearly from instantaneous operating states alone without cumulative time-series data."
                )
            with col_r2:
                fig_res = regression.plot_regression_residuals(y_te, res_reg["predictions"])
                st.pyplot(fig_res)
        else:
            st.warning("ai4i2020.csv required to run Task 6 interactively.")

    # TASK 7
    elif "Task 7" in exp_choice:
        st.subheader("🧠 Task 7: 1D Convolutional Neural Network (CNN)")
        st.markdown(
            "Evaluates deep 1D CNN architecture on tabular physical sequences vs baseline Logistic Regression. "
            "Architecture: `Conv1D(32) -> BatchNorm -> MaxPool -> Conv1D(64) -> BatchNorm -> GlobalAvgPool -> Dense(64) -> Dropout -> Dense(1)`."
        )

        task7_meta = metadata.get("task7_cnn_benchmarks", {})
        cnn_metrics = task7_meta.get("cnn_metrics", {})
        lr_metrics = task7_meta.get("baseline_logistic_regression_metrics", {})

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 1D CNN Test Metrics")
            st.dataframe(pd.DataFrame([cnn_metrics]), use_container_width=True)
        with col_c2:
            st.markdown("#### Baseline Logistic Regression Metrics")
            st.dataframe(pd.DataFrame([lr_metrics]), use_container_width=True)

        st.info(
            "🔬 **Architecture Assessment:** On tabular data with 7 physical features, tree ensembles (Random Forest / Gradient Boosting) "
            "outperform 1D CNN in precision and training efficiency, avoiding arbitrary spatial filter assumptions over non-sequential sensor metrics."
        )

    # TASK 8
    elif "Task 8" in exp_choice:
        st.subheader("🔍 Task 8: Interpretability & Layer Activations")
        comp_text = interpretability.get_interpretability_comparison_text()
        col_t8_1, col_t8_2 = st.columns(2)
        with col_t8_1:
            st.markdown(f"#### {comp_text['baseline_ml']['title']}")
            st.markdown(f"**Method:** {comp_text['baseline_ml']['method']}")
            st.markdown(f"**Strengths:** {comp_text['baseline_ml']['strengths']}")
            st.markdown(f"**Limitations:** {comp_text['baseline_ml']['limitations']}")
        with col_t8_2:
            st.markdown(f"#### {comp_text['cnn']['title']}")
            st.markdown(f"**Method:** {comp_text['cnn']['method']}")
            st.markdown(f"**Strengths:** {comp_text['cnn']['strengths']}")
            st.markdown(f"**Limitations:** {comp_text['cnn']['limitations']}")


# ============================================================================
# 5. ABOUT, ETHICS & LIMITATIONS
# ============================================================================
elif nav_selection == "⚖️ About, Ethics & Limitations":
    st.title("⚖️ Engineering Ethics, Reliability & System Limitations")
    st.markdown(
        "MachineGuard is an industrial decision-support system. Understanding model boundaries, statistical "
        "assumptions, and operational safety protocols is essential for responsible deployment."
    )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### ⚠️ Class Imbalance & Asymmetric Error Costs")
        st.markdown(
            """
            - **Extreme Imbalance:** In the AI4I 2020 dataset, machine failures account for only **3.39%** of records (339 failures vs 9,661 normal operations).
            - **Accuracy Paradox:** A naive model predicting 'Normal' 100% of the time achieves **96.61% accuracy** while missing every single equipment breakdown. Accuracy is therefore fundamentally misleading.
            - **Cost Asymmetry:**
              * **False Negative (Missed Failure):** Unplanned catastrophic equipment breakdown, costly downtime, worker safety hazard (~$3,500+).
              * **False Positive (False Alarm):** Premature tool inspection or unnecessary scheduled maintenance check (~$150).
            - **Operating Policy:** MachineGuard operates at an optimized threshold (`0.35`) to deliberately bias towards recall/safety.
            """
        )

        st.markdown("### 🔍 Target Leakage & Predictor Integrity")
        st.markdown(
            """
            - **The Hazard:** The original AI4I dataset contains failure mode columns: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
            - **Strict Exclusion:** These indicators encode post-hoc failure diagnoses. Including them as model inputs yields artificial 99.9% accuracy but constitutes complete target leakage.
            - **Enforcement:** MachineGuard strictly strips all failure mode columns and identifier keys (`UDI`, `Product ID`) prior to preprocessing and training.
            """
        )

    with col_e2:
        st.markdown("### 🏭 Domain Shift & Synthetic Data Provenance")
        st.markdown(
            """
            - **Synthetic Provenance:** The AI4I 2020 dataset is synthetically generated using physical equations with added noise.
            - **Real-World Domain Shift:** Real manufacturing environments involve sensor drift, vibration harmonics, variable ambient humidity, and multi-component degradation not captured in synthetic benchmarks.
            - **Generalization Boundary:** Performance on synthetic test data does not guarantee zero-shot transfer to new physical plant machinery without domain calibration.
            """
        )

        st.markdown("### 👷 Human-in-the-Loop & Safety Accountability")
        st.markdown(
            """
            - **Decision Support Only:** MachineGuard generates recommendations, not autonomous authorization to operate or shut down equipment.
            - **Human Oversight:** Qualified maintenance engineers must inspect physical equipment and verify sensor readings prior to executing high-cost interventions.
            - **Explainability as Evidence, Not Proof:** Feature attributions highlight statistical correlations in the model's decision path; they do not replace formal root-cause failure analysis (RCFA).
            """
        )

    st.markdown("---")
    st.caption("MachineGuard • Built for Industrial Reliability, Transparent Explainability, and Responsible AI Practices.")
