# CIL Integrated Project: Predictive Maintenance & AI Planning (Tasks 3–8)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.66.0-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-42%20passed-brightgreen.svg)](tests/)

An enterprise-grade, modular Streamlit application integrating computational maintenance planning, statistical data profiling, classical machine learning, 1D convolutional neural networks, and model interpretability (Tasks 3 through 8).

---

## 1. Project Overview & Scope

This project consolidates and refactors work from three standalone Jupyter notebooks into a deployable, interactive system:
* **`CIL3.ipynb`** $\rightarrow$ **Task 3:** State Space Search & AI Maintenance Planning (BFS, DFS, A*).
* **`CIL456.ipynb`** $\rightarrow$ **Tasks 4, 5, 6:** Exploratory Data Analysis, Supervised Classification (4 models), Clustering (K-Means, Dendrogram), and Tool Wear Regression.
* **`CIL78 (1).ipynb`** $\rightarrow$ **Tasks 7, 8:** 1D CNN Architecture, Imbalance Handling, Early Stopping, ROC/PR Evaluation, Conv1D Feature Map Activations, and LIME Explainability.

Original notebooks remain preserved in `notebooks/` and in the repository root.

---

## 2. Confirmed Task Coverage

| Task | Title | Key Implementation Details | Status |
| :--- | :--- | :--- | :--- |
| **Task 3** | **AI Maintenance Planning** | 8-element state space, 9 maintenance actions, BFS, DFS, A* with admissible heuristic, interactive path validator. | **100% Operational** (Runs independently of dataset) |
| **Task 4** | **Exploratory Data Analysis** | Outlier boxplots, KDE histograms, numerical correlation heatmap, `StandardScaler` pipeline. | **Ready** (Live when data present; schema audited) |
| **Task 5** | **Supervised & Unsupervised ML** | Random Forest feature importance, 4 classifiers (RF, DT, SVM, LogReg), K-Means clustering, Ward hierarchical dendrogram. | **Ready** (Live when data present; historical audit available) |
| **Task 6** | **Tool Wear Regression** | Linear regression on continuous tool degradation, MSE / $R^2$ metrics, Actual vs. Predicted scatter plot, residual diagnostics. | **Ready** (Live when data present; historical audit available) |
| **Task 7** | **1D CNN Deep Learning** | Conv1D feature extractor with Batch Normalization, Max/Avg pooling, Class Weights for imbalance, ROC/PR curves, baseline Logistic Regression comparison. | **Ready** (Dual backend: PyTorch native on Win/Py3.14 + Keras support) |
| **Task 8** | **Model Interpretability** | Conv1D layer filter activation heatmaps across 4 diagnostic cases (TP, TN, FP, FN), LIME feature attributions, linear vs deep comparative analysis. | **Ready** (Interactive exploration profiles active) |

---

## 3. Project Directory Structure

```text
CIL_Integrated_Project/
├── app.py                         # KnowledgeFabric/MachineGuard Streamlit application
├── config.py                      # Centralized path, feature, and schema configuration
├── notebooks/                     # Preserved, read-only copies of original notebooks
│   ├── CIL3.ipynb                 # Original Task 3 notebook
│   ├── CIL456.ipynb               # Original Tasks 4-6 notebook
│   └── CIL78.ipynb                # Original Tasks 7-8 notebook
├── modules/                       # Reusable analytical modules
│   ├── __init__.py
│   ├── model_trainer.py           # Leakage-free training & artifact persistence pipeline
│   ├── inference.py               # Production inference engine with schema validation
│   ├── decision_layer.py          # Risk tier mapping & maintenance recommendations
│   ├── history_manager.py         # Assessment history logger & CSV exporter
│   ├── interpretability.py        # Tree attributions, what-if simulator & activations
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
│   │   └── test_set_split.joblib   # Held-out test split for demonstration
│   ├── assessment_history.json    # Persisted assessment audit log
│   ├── figures/                   # Exported charts and graphs
│   └── metadata.json              # Verified metrics, schema, and threshold curves
├── tests/                         # Automated test suite (42 tests)
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
```

---

## 4. Environment Setup & Installation

### Prerequisites
* Windows, macOS, or Linux
* Python 3.10 – 3.14 (Note: Python 3.14 on Windows runs PyTorch, Scikit-learn, and Streamlit natively; if TensorFlow is required, use Python $\le$ 3.12).

### Installation (PowerShell on Windows)
```powershell
# 1. Navigate to the integrated project folder
cd CIL_Integrated_Project

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install -r requirements.txt
```

---

## 5. How to Launch the Streamlit Application

Run the application using Streamlit:
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### Application Features:
* **Zero-Crash Startup:** The application starts immediately even when the dataset is absent.
* **Task 3 AI Planner:** Interactive immediately without external data.
* **Live File Upload:** If `ai4i2020.csv` is not in the filesystem, an interactive uploader on the Overview or EDA page allows instant in-session loading.
* **Dual Deep Learning Engine:** Automatically detects available frameworks (PyTorch or TensorFlow/Keras).

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
```

---

## 8. Deployment to Streamlit Community Cloud

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
