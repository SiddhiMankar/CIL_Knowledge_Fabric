# CIL Knowledge Fabric — MachineGuard Predictive Maintenance Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
<<<<<<< HEAD
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.0-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-orange.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-green.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-42%20passed-brightgreen.svg)](CIL_Integrated_Project/tests/)
[![Dataset](https://img.shields.io/badge/Dataset-AI4I%202020%20(UCI)-blueviolet.svg)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

An enterprise-grade, explainable industrial intelligence platform combining formal knowledge representation, classical machine learning, deep 1D convolutional neural networks, forward-chaining rule inference, heuristic state-space planning, and comprehensive validation.

Built on the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository), this system unifies all ten syllabus tasks into a single coherent, working web application.
=======
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.58.0%2B-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-55%20passed-brightgreen.svg)](tests/)

An enterprise-grade, modular Streamlit predictive maintenance intelligence platform (**MachineGuard**) integrating user-selectable ML/DL models (Random Forest, Gradient Boosting, Logistic Regression, 1D CNN), computational maintenance planning, statistical data profiling, and explainability (Tasks 3 through 8).
>>>>>>> 67c25a8ec6c7c96a556a72bd95d31bda6d5f2cac

---

## 1. Syllabus Tasks Compliance Matrix

<<<<<<< HEAD
| Task | Syllabus Requirement | Implementation Evidence | Verification / Results | Limitations & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Knowledge Representation** | `modules/knowledge_representation.py`, Page 3 of `app.py` | Frame-based schema with entities (`Machine`, `SensorReading`, `OperatingCondition`, `FailureType`, `PredictedFailureRisk`, `MaintenanceAction`, `MaintenanceRecommendation`), physical ranges, constraint validation engine, and relationship graph. | Validation ranges derived from AI4I dataset bounds; assumes continuous sensor telemetry. |
| **Task 2** | **Rule-Based Inference Engine** | `modules/decision_layer.py`, Page 4 of `app.py` | Declarative forward-chaining rule engine (`Rule`, `RuleEngine`, `InferenceTrace`), priority-based conflict resolution, full audit trace, explicit distinction between statistical likelihood and physical certainty. | Thresholds derived from domain physics (HDF temp diff $\le 8.6\text{K}$, OSF strain limits, wear limit 200 min). |
| **Task 3** | **AI Search & State-Space Planning** | `modules/planning.py`, Page 7 of `app.py` | 8-element boolean state vector, 9 maintenance actions and costs, BFS, DFS, A* with admissible heuristic ($h(s) = \sum \text{cost}$), interactive path validator, recommendation-to-plan bridge. | Domain model represents discrete maintenance tasks for industrial spindle/pump assemblies. |
| **Task 4** | **Exploratory Data Analysis & Preprocessing** | `modules/eda.py`, Page 2 of `app.py` | Shape, datatypes, missing/duplicate audits, outlier boxplots, KDE histograms, numerical correlation heatmap, extreme class imbalance visualization (~3.39% failures), leakage-free scaling. | AI4I 2020 has 0 missing values and 0 duplicate rows in raw format. |
| **Task 5** | **Feature Engineering & Selection** | `modules/classical_ml.py`, Page 8 of `app.py` | Domain feature derivation (Temperature difference, Mechanical power, Tool wear torque strain), Random Forest feature importance, K-Means clustering ($k=3$), Hierarchical Ward dendrogram. | Categorical encoding supports both LabelEncoding (6 features) and One-Hot encoding (7 features). |
| **Task 6** | **Classical ML & Evaluation** | `modules/classical_ml.py`, `modules/regression.py`, Page 8 of `app.py` | 5 classifiers (Random Forest, Gradient Boosting, Decision Tree, SVM RBF, Logistic Regression), stratified 80/20 holdout evaluation, failure-class recall/precision, Tool Wear Linear Regression ($R^2 \approx 0$). | Demonstrates why raw accuracy is misleading on imbalanced failure targets. |
| **Task 7** | **Deep Learning Architecture** | `modules/cnn_model.py`, `train_all_models.py`, Page 8 of `app.py` | 1D CNN in PyTorch (Conv1D 32 $\rightarrow$ BatchNorm $\rightarrow$ MaxPool $\rightarrow$ Conv1D 64 $\rightarrow$ BatchNorm $\rightarrow$ GlobalAvgPool $\rightarrow$ Dense 64 $\rightarrow$ Dropout $\rightarrow$ Sigmoid), balanced class weighting. | Evaluated on 2,000 holdout test samples: ROC-AUC = 0.9785, Failure Recall = 91.18% (capturing 62 of 68 failures). |
| **Task 8** | **Model Interpretability & Explainability** | `modules/interpretability.py`, Page 6 of `app.py` | Real Conv1D filter activation heatmaps from trained PyTorch weights, LIME local surrogate explanations, diagnostic profiles (TP, TN, FP, FN), and interactive What-If parameter sensitivity simulator. | Activation maps reflect feature representations, not root physical failure causes. |
| **Task 9** | **Complete End-to-End Integration** | `modules/model_registry.py`, `modules/history_manager.py`, Page 5 of `app.py` | Unified 10-step pipeline: Input $\rightarrow$ Validation $\rightarrow$ Preprocessing $\rightarrow$ Dynamic Model Selection $\rightarrow$ Inference $\rightarrow$ Rule Engine $\rightarrow$ A* Plan $\rightarrow$ Result $\rightarrow$ Persistent History $\rightarrow$ Evaluation. | Models loaded dynamically from registry without refitting scalers. |
| **Task 10** | **Validation, Ethics, Fairness & Subgroup Analysis** | `docs/VALIDATION_AND_ETHICS.md`, Page 10 of `app.py` | Holdout validation on 2,000 samples, Subgroup fairness audit across Types L, M, H, 8-case boundary stress test suite, industrial ethics (asymmetric error costs, human-in-the-loop, privacy). | Type H has smaller sample volume (214 test samples, 5 failures), impacting recall variance. |

---

## 2. Project Architecture & Directory Structure

```text
CIL_KF/
├── docs/
│   ├── IMPLEMENTATION_AUDIT.md       # Complete requirements-to-implementation audit
│   └── VALIDATION_AND_ETHICS.md      # Holdout validation, fairness analysis, & ethics report
├── CIL_Integrated_Project/
│   ├── app.py                         # Streamlit multi-page platform entry point
│   ├── config.py                      # Centralized path, schema, and environment configuration
│   ├── train_all_models.py            # Reproducible model training & artifact generation script
│   ├── generate_validation_report.py  # Automated fairness audit & stress testing script
│   ├── requirements.txt               # Pinned Python package dependencies
│   ├── modules/                       # Reusable analytical modules
│   │   ├── __init__.py
│   │   ├── knowledge_representation.py# Task 1: Frames, entities, schema, physical constraints
│   │   ├── decision_layer.py          # Task 2: Forward-chaining rule engine & inference trace
│   │   ├── planning.py                # Task 3: BFS, DFS, A*, path validator, recommendation bridge
│   │   ├── eda.py                     # Task 4: EDA distributions, boxplots, correlation, scaling
│   │   ├── classical_ml.py            # Task 5 & 6: Feature importance, 5 classifiers, clustering
│   │   ├── regression.py              # Task 6: Tool wear linear regression & diagnostics
│   │   ├── cnn_model.py               # Task 7: PyTorch 1D CNN architecture & evaluation
│   │   ├── interpretability.py        # Task 8: Conv1D activations, LIME, What-If simulator
│   │   ├── model_registry.py          # Task 9: Dynamic model discovery & schema bridge
│   │   └── history_manager.py         # Task 9: Persistent JSON assessment audit logger
│   ├── data/
│   │   ├── raw/
│   │   │   └── ai4i2020.csv           # AI4I 2020 Predictive Maintenance dataset (10,000 rows)
│   │   └── processed/                 # Preprocessed/scaled dataset exports
│   ├── artifacts/
│   │   ├── metadata.json              # Measured test metrics, architectures, & hyperparameters
│   │   ├── assessment_history.json    # Persistent log of user assessments
│   │   └── models/                    # Serialized model weights & preprocessors
│   │       ├── classical_models.joblib# Serialized bundle (RF, GB, DT, SVM, LogReg)
│   │       ├── scaler_classical.joblib# Fitted StandardScaler for 6-feature pipeline
│   │       ├── label_encoder.joblib   # Fitted LabelEncoder for Type
│   │       ├── cnn_pytorch.pt         # Trained PyTorch 1D CNN state dictionary
│   │       └── scaler_cnn.joblib      # Fitted StandardScaler for 7-feature OneHot pipeline
│   └── tests/                         # Automated test suite (42 passed tests)
│       ├── conftest.py                # Pytest configuration & Matplotlib acceleration
│       ├── test_app_startup.py        # Streamlit syntax and py_compile smoke test
│       ├── test_config.py             # Path resolution and feature list integrity
│       ├── test_knowledge_representation.py # Task 1 entity, constraint & schema tests
│       ├── test_rule_engine.py        # Task 2 rule firing, conflict resolution & trace tests
│       ├── test_planning.py           # Task 3 BFS, DFS, A* optimality & path tests
│       ├── test_modules_contracts.py  # Synthetic contract tests for analytical pipelines
│       ├── test_missing_dataset_behavior.py # Missing dataset exception handling tests
│       ├── test_end_to_end_integration.py # Task 9 complete pipeline integration tests
│       └── test_stress_validation.py  # Task 10 input boundaries & guard tests
├── notebooks/                         # Preserved original coursework notebooks
│   ├── CIL3.ipynb                     # Original Task 3 notebook
│   ├── CIL456.ipynb                   # Original Tasks 4-6 notebook
│   └── CIL78.ipynb                    # Original Tasks 7-8 notebook
└── README.md                          # This root documentation
=======
This project consolidates and refactors work from three standalone Jupyter notebooks into a deployable, interactive system:
* **`CIL3.ipynb`** $\rightarrow$ **Task 3:** State Space Search & AI Maintenance Planning (BFS, DFS, A*).
* **`CIL456.ipynb`** $\rightarrow$ **Tasks 4, 5, 6:** Exploratory Data Analysis, Supervised Classification (4 models), Clustering (K-Means, Dendrogram), and Tool Wear Regression.
* **`CIL78 (1).ipynb`** $\rightarrow$ **Tasks 7, 8:** 1D CNN Architecture, Imbalance Handling, Early Stopping, ROC/PR Evaluation, Conv1D Feature Map Activations, and Explainability.

Original notebooks remain preserved in `notebooks/` and in the repository root.

---

## 2. Confirmed Task Coverage

| Task | Title | Key Implementation Details | Status |
| :--- | :--- | :--- | :--- |
| **Task 3** | **AI Maintenance Planning** | 8-element state space, 9 maintenance actions, BFS, DFS, A* with admissible heuristic, interactive path validator. | **100% Operational** (Runs independently of dataset) |
| **Task 4** | **Exploratory Data Analysis** | Outlier boxplots, KDE histograms, numerical correlation heatmap, `StandardScaler` pipeline. | **Ready** (Live when data present; schema audited) |
| **Task 5** | **Supervised & Unsupervised ML** | Random Forest feature importance, 4 classifiers (RF, DT, SVM, LogReg), K-Means clustering, Ward hierarchical dendrogram. | **Ready** (Live when data present; model registry active) |
| **Task 6** | **Tool Wear Regression** | Linear regression on continuous tool degradation, MSE / $R^2$ metrics, Actual vs. Predicted scatter plot, residual diagnostics. | **Ready** (Live when data present; historical audit available) |
| **Task 7** | **1D CNN Deep Learning** | Conv1D feature extractor with Batch Normalization, Max/Avg pooling, Class Weights for imbalance, ROC/PR curves, baseline Logistic Regression comparison. | **Ready** (PyTorch native on Win/Py3.14 + Keras support) |
| **Task 8** | **Model Interpretability** | Conv1D layer filter activation heatmaps across 4 diagnostic cases (TP, TN, FP, FN), tree-path attributions, linear log-odds analysis. | **Ready** (Interactive exploration profiles active) |

---

## 3. Project Directory Structure

```text
CIL_Integrated_Project/
├── app.py                         # MachineGuard Streamlit application
├── config.py                      # Centralized path, feature, and schema configuration
├── notebooks/                     # Preserved, read-only copies of original notebooks
│   ├── CIL3.ipynb                 # Original Task 3 notebook
│   ├── CIL456.ipynb               # Original Tasks 4-6 notebook
│   └── CIL78.ipynb                # Original Tasks 7-8 notebook
├── modules/                       # Reusable analytical modules
│   ├── __init__.py
│   ├── model_registry.py          # Centralized Model Registry (catalog, metadata, comparison)
│   ├── model_trainer.py           # Leakage-free training & multi-model artifact persistence pipeline
│   ├── inference.py               # Unified multi-model inference engine with schema validation
│   ├── decision_layer.py          # Risk tier mapping & maintenance recommendations
│   ├── history_manager.py         # Assessment history logger with model metadata & filtering
│   ├── interpretability.py        # Tree attributions, linear log-odds, CNN activations & what-if simulator
│   ├── planning.py                # Task 3: BFS, DFS, A*, validator
│   ├── eda.py                     # Task 4: EDA, outlier plots, correlation, scaling
│   ├── classical_ml.py            # Task 5: 4 Classifiers, feature importance, clustering
│   ├── regression.py              # Task 6: Tool wear linear regression & diagnostics
│   └── cnn_model.py               # Task 7: 1D CNN architecture & evaluation
├── data/
│   ├── raw/ai4i2020.csv           # AI4I 2020 Predictive Maintenance dataset
│   └── processed/                 # Scaled and preprocessed datasets
├── artifacts/
│   ├── models/
│   │   ├── production_model.joblib # Persisted Random Forest + ColumnTransformer
│   │   ├── random_forest.joblib    # Random Forest classifier artifact
│   │   ├── gradient_boosting.joblib # Gradient Boosting classifier artifact
│   │   ├── logistic_regression.joblib # Logistic Regression classifier artifact
│   │   ├── cnn_1d.joblib           # PyTorch 1D CNN model artifact
│   │   └── test_set_split.joblib   # Common held-out test split for demonstration
│   ├── assessment_history.json    # Persisted assessment audit log with model metadata
│   ├── figures/                   # Exported charts and graphs
│   └── metadata.json              # Verified metrics, schema, and threshold curves
├── tests/                         # Automated test suite (55 tests)
│   ├── test_model_selection.py    # Model registry, multi-model inference & comparison tests
│   ├── test_machineguard_core.py  # Inference, decision layer, history, validation tests
│   ├── test_app_startup.py        # App syntax and compilation smoke test
│   ├── test_config.py             # Path resolution and feature list tests
│   ├── test_planning.py           # BFS/DFS/A* correctness and optimality tests
│   ├── test_missing_dataset_behavior.py  # Graceful degradation tests
│   └── test_modules_contracts.py  # Modular interface contract tests
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules for Python, IDEs, and artifacts
├── README.md                      # This comprehensive documentation
└── PROJECT_AUDIT.md               # Engineering audit & transformation report
>>>>>>> 67c25a8ec6c7c96a556a72bd95d31bda6d5f2cac
```

---

## 3. Measured Holdout Performance Results

All models were evaluated on the **2,000-sample unseen holdout test set** (stratified 80/20 train/test split):

| Model | Paradigm | Accuracy | Failure Recall | Failure Precision | Failure F1 | ROC-AUC | False Negatives (Missed) | False Positives (Alarms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Ensemble (100 Trees) | 98.50% | 64.71% | 88.00% | 0.7458 | 0.9645 | 24 | 6 |
| **Gradient Boosting** | Ensemble (100 Trees) | 98.50% | 66.18% | 86.54% | 0.7500 | 0.9684 | 23 | 7 |
| **1D CNN (PyTorch)** | Conv1D Deep Learning | 92.50% | **91.18%** | 30.10% | 0.4526 | **0.9785** | **6** | 144 |
| **Decision Tree** | Tree (Max Depth = 5) | 97.30% | 29.41% | 76.92% | 0.4255 | 0.9147 | 48 | 6 |
| **SVM (RBF)** | Kernel Machine | 97.10% | 17.65% | 85.71% | 0.2927 | 0.9503 | 56 | 2 |
| **Logistic Regression**| Linear Baseline | 96.80% | 11.76% | 66.67% | 0.2000 | 0.9005 | 60 | 4 |

### Tool Wear Linear Regression (Task 6)
- **Target:** `Tool wear [min]`
- **Holdout Metrics:** $\text{MSE} = 1.0073$, $\text{RMSE} = 1.0036$, $\text{MAE} = 0.8540$, $R^2 = -0.0009$.
- **Finding:** Near-zero $R^2$ demonstrates that cumulative cutting edge wear cannot be modeled purely from instantaneous physical parameters without operating hour history.

---

## 4. Installation & Execution Guide

### Prerequisites
- Python 3.10 – 3.14 (Verified on Python 3.14 on Windows 11).
- Windows PowerShell or command prompt.

### 1. Setup Virtual Environment
```powershell
# Navigate to workspace root
cd C:\Projects\CIL_KF

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r CIL_Integrated_Project\requirements.txt
```

### 3. Verify Dataset
The AI4I 2020 dataset is already provisioned at:
`CIL_Integrated_Project\data\raw\ai4i2020.csv`

<<<<<<< HEAD
### 4. Run Automated Test Suite (42 Tests)
```powershell
$env:MPLCONFIGDIR = "$env:TEMP\mpl"
.\.venv\Scripts\python.exe -m pytest CIL_Integrated_Project\tests -v
```

### 5. Launch Streamlit Application
```powershell
$env:MPLCONFIGDIR = "$env:TEMP\mpl"
.\.venv\Scripts\streamlit.exe run CIL_Integrated_Project\app.py
=======
---

## 6. Dataset Requirements & Placement

* **Expected Filename:** `ai4i2020.csv` (AI4I 2020 Predictive Maintenance Dataset).
* **Expected Location:** `data/raw/ai4i2020.csv` (also automatically detected if placed in project root or parent folder).
* **Expected Dimensions:** 10,000 rows × 14 columns.
* **Expected Schema:**
  * `UDI` (Integer ID)
  * `Product ID` (Categorical String, e.g., 'M14860', 'L47181')
  * `Type` (Categorical: 'L', 'M', 'H')
  * `Air temperature [K]` (Float)
  * `Process temperature [K]` (Float)
  * `Rotational speed [rpm]` (Integer/Float)
  * `Torque [Nm]` (Float)
  * `Tool wear [min]` (Integer/Float)
  * `Machine failure` (Binary Target: 0 = Normal, 1 = Failure)
  * Failure Modes (Excluded during feature engineering): `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

---

## 7. Running the Automated Test Suite

Run the full pytest suite from the **repository root**:
```powershell
python -m pytest CIL_Integrated_Project/tests -v
```

### Verified Test Results (42 tests, 0 failures):
```text
CIL_Integrated_Project/tests/test_app_startup.py::test_app_compiles_cleanly PASSED
CIL_Integrated_Project/tests/test_config.py::test_paths_exist_or_can_be_created PASSED
CIL_Integrated_Project/tests/test_config.py::test_feature_lists_integrity PASSED
CIL_Integrated_Project/tests/test_config.py::test_locate_dataset_contract PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_inference_engine_initialization PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_valid_single_prediction PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_input_validation_missing_column PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_input_validation_invalid_type PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_input_validation_out_of_bounds PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_batch_prediction_dataframe PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_decision_layer_mapping PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_critical_failure_condition_decision PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_history_logging_and_export PASSED
CIL_Integrated_Project/tests/test_machineguard_core.py::test_missing_model_controlled_failure PASSED
... (22 machineguard_core tests) ...
CIL_Integrated_Project/tests/test_planning.py::test_bfs_search PASSED
CIL_Integrated_Project/tests/test_planning.py::test_dfs_search PASSED
CIL_Integrated_Project/tests/test_planning.py::test_astar_search PASSED
CIL_Integrated_Project/tests/test_planning.py::test_path_validation PASSED
CIL_Integrated_Project/tests/test_planning.py::test_run_all_algorithms PASSED
CIL_Integrated_Project/tests/test_modules_contracts.py::test_classical_ml_pipeline_contract PASSED
CIL_Integrated_Project/tests/test_missing_dataset_behavior.py::test_eda_load_dataset_missing_file_raises PASSED

============================= 42 passed in 9.45s ==============================
>>>>>>> 67c25a8ec6c7c96a556a72bd95d31bda6d5f2cac
```
Open your browser to: **`http://localhost:8501`**

---

## 5. Step-by-Step Demonstration Sequence (Viva & Practical Demo)

<<<<<<< HEAD
Follow this 5-minute demonstration sequence during practical evaluation:

1. **Overview & Architecture (Page 1):**
   - Show the 10/10 task compliance matrix and confirm dataset status (10,000 rows × 14 columns).
   - Point out active models in the dynamic registry.
2. **Data Explorer (Page 2):**
   - Show the **Class Imbalance Chart** (~96.6% Normal vs ~3.4% Failure).
   - Explain why high accuracy alone is deceptive and why **Failure Recall** is critical.
3. **Knowledge Representation (Page 3):**
   - Show the **Frame-Based Schema** (Entities, Slots, Relationships).
   - Open the **Interactive Validation Sandbox**, enter an out-of-range temperature (e.g., 520 K), and demonstrate that the schema flags the violation.
4. **Rule-Based Inference (Page 4):**
   - Review the declarative rule catalog.
   - Select the **Severe Heat Dissipation Failure Risk (HDF)** preset.
   - Show the **Transparent Inference Trace**: explain how the forward chaining engine evaluates physical conditions ($T_{proc} - T_{air} \le 8.6\text{K}$) and triggers `INSPECT_COOLING` while resolving conflicts by rule priority.
5. **Predictive Assessment & End-to-End Workflow (Page 5):**
   - Select **Random Forest** or **1D CNN (PyTorch)** from the dynamic model dropdown.
   - Set parameters: Machine Type = `L`, Speed = `1300 rpm`, Torque = `65 Nm`, Tool Wear = `210 min`.
   - Click **Run Complete Predictive Assessment**.
   - Review the integrated output: Model Probability + Rule Recommendation (`REPLACE_TOOL_OVERSTRAIN`) + Safety Halting Flag + AI Maintenance Plan + History persistence confirmation.
6. **What-If Simulator (Page 6):**
   - Move the **Spindle Torque** slider from 40 Nm to 70 Nm.
   - Observe the live probability delta and how the recommendation changes dynamically.
7. **AI Maintenance Planner (Page 7):**
   - Click **Run Planning Benchmark** to compare **BFS vs DFS vs A***.
   - Show that A* achieves the optimal path cost (14 units) while expanding minimal nodes using the admissible heuristic.
8. **Model Comparison & Subgroup Fairness (Pages 8 & 10):**
   - Highlight the trade-off table: 1D CNN provides 91.18% failure recall vs Logistic Regression's 11.76%.
   - Show the Subgroup Fairness table across Types L, M, and H.

---

## 6. Software Quality & Security

- **Data Leakage Elimination:** Scalers and encoders are strictly fitted on training splits (`X_train`) and only transform holdout test and inference records.
- **Defensive Error Handling:** Malformed inputs or missing files trigger informative warnings rather than unhandled application crashes.
- **Safe Persistence:** Local assessment records are written in standard JSON format without executable object deserialization.
- **Zero API Key / Cloud Dependency:** The application runs 100% locally on the host machine without external network requirements.
=======
This repository is prepared for zero-configuration deployment to **Streamlit Community Cloud**:
1. Push to GitHub (already done at `SiddhiMankar/CIL_Knowledge_Fabric`).
2. In [share.streamlit.io](https://share.streamlit.io/), click **New app**.
3. Select repository and branch `main`.
4. Set Main file path: `CIL_Integrated_Project/app.py`.
5. Deploy. The application starts immediately — Task 3 AI Planner, Machine Assessment, and Model Performance pages are all fully functional.

---

## 9. Production Model & Current Status

The dataset (`ai4i2020.csv`) is present and the production model is trained and persisted:

| Item | Status |
| :--- | :--- |
| Dataset `data/raw/ai4i2020.csv` | Present (10,000 rows × 14 cols) |
| `artifacts/models/production_model.joblib` | Trained, persisted, version 1.0.0 |
| `artifacts/models/test_set_split.joblib` | Held-out test split (2,000 samples) |
| `artifacts/metadata.json` | Populated with verified held-out metrics |
| Automated test suite | 42 passed, 0 failed |

To retrain the production model from scratch:
```powershell
cd CIL_Integrated_Project
python -m modules.model_trainer
```
>>>>>>> 67c25a8ec6c7c96a556a72bd95d31bda6d5f2cac
