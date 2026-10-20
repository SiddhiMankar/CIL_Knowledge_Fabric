# MachineGuard — Predictive Maintenance Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.66.0-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-30%20passed-brightgreen.svg)](tests/)

**MachineGuard** is an industrial decision-support prototype built by transforming and unifying laboratory tasks 3 through 8 of the CIL Integrated Project into an auditable, explainable predictive maintenance platform.

It directly answers four essential operational questions:
1. **What is the predicted failure risk** for the supplied machine readings?
2. **Which features influenced the model's prediction** (exact tree-path local attributions)?
3. **What action should the user consider**, and what physical evidence supports it?
4. **How reliable is the system**, and what are its operational boundaries and limitations?

---

## 1. Core Architecture & Main Workflow

The application is structured into a streamlined industrial workflow:

* **🛡️ Machine Assessment (Primary Workflow):**
  - **Validated Telemetry Form:** Real-time sensor inputs (`Air temp`, `Process temp`, `Rotational speed`, `Torque`, `Tool wear`, and machine variant `Type`).
  - **Demonstration Profiles:** Verified operating points from test data (Nominal, Critical Wear, Overstrain, Heat Dissipation).
  - **Batch Upload:** Upload any operational CSV for bulk assessment and export results.
  - **Instant Inference:** Evaluates the persisted Random Forest model (`artifacts/models/production_model.joblib`) with **zero retraining during runtime**.
  - **Real Local Feature Attributions:** Exact tree-path probability deltas showing which sensors increased or mitigated risk.
  - **What-If Counterfactual Simulator:** Interactive sensor sensitivity curves showing how varying a reading alters failure probability.
* **📋 Assessment History:**
  - Auditable in-session and persistent logs of all machine assessments (`artifacts/assessment_history.json`).
  - One-click CSV audit trail export.
* **📊 Model Performance & Leakage-Free Validation:**
  - Evaluated strictly on an untouched held-out test split (2,000 samples, 20% stratified split).
  - Reports Precision, Recall, F1, ROC-AUC, and PR-AUC.
  - Full confusion matrix and operating threshold sensitivity analysis (`0.20` to `0.50`).
  - Subgroup reliability checks across machine quality variants (`L`, `M`, `H`).
* **🔬 Experiments (Tasks 3–8):**
  - Preserved academic and laboratory modules:
    * **Task 3:** State Space Search AI Planner (BFS, DFS, A* for Pump P-101).
    * **Task 4:** EDA, outlier boxplots, and correlation heatmaps.
    * **Task 5:** Classical classifiers (RF, DT, SVM, LogReg) and K-Means clustering.
    * **Task 6:** Tool Wear linear regression residual diagnostics.
    * **Task 7:** 1D CNN deep learning architecture benchmark.
    * **Task 8:** Conv1D activation feature maps and LIME explainability.
* **⚖️ About, Ethics & Limitations:**
  - Class imbalance consequences (3.39% failure rate).
  - Target leakage prevention (exclusion of `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
  - Synthetic data provenance and domain shift caveats.
  - Human-in-the-loop safety oversight.

---

## 2. Directory Structure

```text
CIL_Integrated_Project/
├── app.py                         # Unified MachineGuard Streamlit application
├── config.py                      # Centralized path, feature, and schema configuration
├── notebooks/                     # Preserved, read-only copies of original notebooks
│   ├── CIL3.ipynb                 # Task 3 notebook
│   ├── CIL456.ipynb               # Tasks 4-6 notebook
│   └── CIL78.ipynb                # Tasks 7-8 notebook
├── modules/                       # Reusable analytical modules
│   ├── __init__.py
│   ├── model_trainer.py           # Leakage-free training and artifact persistence pipeline
│   ├── inference.py               # Production inference engine with schema and bounds validation
│   ├── decision_layer.py          # Operational risk tier mapping and inspection recommendations
│   ├── history_manager.py         # Assessment history logger and CSV exporter
│   ├── interpretability.py        # Real-model tree attributions, what-if sensitivity & activations
│   ├── planning.py                # Task 3: State-space planning (BFS, DFS, A*)
│   ├── eda.py                     # Task 4: EDA and preprocessing
│   ├── classical_ml.py            # Task 5: Classical ML and clustering
│   ├── regression.py              # Task 6: Tool wear regression
│   └── cnn_model.py               # Task 7: 1D CNN architecture
├── data/
│   ├── raw/ai4i2020.csv           # Original AI4I 2020 predictive maintenance dataset
│   └── processed/                 # Scaled and preprocessed datasets
├── artifacts/
│   ├── models/
│   │   ├── production_model.joblib # Persisted Random Forest + ColumnTransformer artifact
│   │   └── test_set_split.joblib   # Persisted held-out test split for demonstration
│   ├── assessment_history.json    # Persisted assessment audit log
│   └── metadata.json              # Verified test metrics, schema, and threshold curves
├── tests/                         # Automated test suite (30 passed tests)
│   ├── test_machineguard_core.py  # Inference, decision layer, validation, and history tests
│   ├── test_app_startup.py        # Streamlit app compilation smoke test
│   ├── test_config.py             # Configuration contract tests
│   ├── test_modules_contracts.py  # Modular transformation contract tests
│   ├── test_planning.py           # AI planner BFS/DFS/A* optimality tests
│   └── test_missing_dataset_behavior.py # Graceful degradation tests
├── requirements.txt
├── PROJECT_AUDIT.md               # Baseline audit and post-transformation report
└── README.md
```

---

## 3. Quick Start & Execution (Windows PowerShell)

### Step 1: Activate Environment
```powershell
cd .\CIL_Integrated_Project\
.\.venv\Scripts\Activate.ps1
```

### Step 2: Run Automated Tests
```powershell
pytest
```
*Expected: 30 passed tests.*

### Step 3: Launch MachineGuard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 4. Retraining & Generating Artifacts

To retrain models from scratch on `data/raw/ai4i2020.csv` and regenerate artifacts:
```powershell
.\.venv\Scripts\python.exe -m modules.model_trainer
```
This strictly enforces:
- Exclusion of post-failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent target leakage.
- Stratified 80/20 train/test split.
- `StandardScaler` fitted exclusively on training data.
- Atomic persistence to `artifacts/models/production_model.joblib` and `artifacts/metadata.json`.

---

## 5. Model Performance Summary (Held-Out Test Set)

| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Production)** | **0.9597** | **0.7814** | **0.7121** | **0.7344** | **0.6912** | **Primary Decision Engine** |
| Gradient Boosting | 0.9699 | 0.8022 | 0.7667 | 0.8846 | 0.6765 | Benchmark Ensemble |
| Logistic Regression (Balanced) | 0.9069 | 0.3817 | 0.2419 | 0.1418 | 0.8235 | Linear Baseline |
| 1D CNN | 0.9800 | 0.5800 | 0.5640 | 0.5000 | 0.6470 | Experimental Deep Learning |
