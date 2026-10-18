# CIL Integrated Project: Predictive Maintenance & AI Planning (Tasks 3–8)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.58.0-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)](tests/)

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
├── app.py                         # Unified Streamlit application entry point
├── config.py                      # Centralized path, feature, and schema configuration
├── notebooks/                     # Preserved, read-only copies of original notebooks
│   ├── CIL3.ipynb                 # Original Task 3 notebook
│   ├── CIL456.ipynb               # Original Tasks 4-6 notebook
│   └── CIL78.ipynb                # Original Tasks 7-8 notebook
├── modules/                       # Reusable analytical modules
│   ├── __init__.py
│   ├── planning.py                # Task 3: BFS, DFS, A*, validator
│   ├── eda.py                     # Task 4: EDA, outlier plots, correlation, scaling
│   ├── classical_ml.py            # Task 5: 4 Classifiers, feature importance, clustering
│   ├── regression.py              # Task 6: Tool wear linear regression & diagnostics
│   ├── cnn_model.py               # Task 7: 1D CNN architecture & evaluation
│   └── interpretability.py        # Task 8: Activation heatmaps & LIME integration
├── data/
│   ├── raw/                       # Place ai4i2020.csv here
│   └── processed/                 # Scaled and preprocessed datasets
├── artifacts/
│   ├── models/                    # Saved model weights (.joblib, .pt, .keras)
│   ├── figures/                   # Exported charts and graphs
│   └── metadata.json              # Historical metrics, parameters, and architecture schemas
├── tests/                         # Automated test suite (20 tests)
│   ├── test_app_startup.py        # App syntax and compilation tests
│   ├── test_config.py             # Path resolution and feature list tests
│   ├── test_planning.py           # Algorithmic correctness and optimality tests
│   ├── test_missing_dataset_behavior.py  # Missing file exception handling
│   └── test_modules_contracts.py  # Synthetic contract tests
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules for Python, IDEs, and artifacts
├── README.md                      # This comprehensive documentation
└── PROJECT_AUDIT.md               # Detailed cell-by-cell notebook audit report
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

Run the full pytest suite from the project root:
```powershell
python -m pytest tests -v
```

### Verified Test Results:
```text
tests/test_app_startup.py::test_app_compiles_cleanly PASSED              [  5%]
tests/test_config.py::test_paths_exist_or_can_be_created PASSED          [ 10%]
tests/test_config.py::test_feature_lists_integrity PASSED                [ 15%]
tests/test_config.py::test_locate_dataset_contract PASSED                [ 20%]
tests/test_missing_dataset_behavior.py::test_eda_load_dataset_missing_file_raises PASSED [ 25%]
tests/test_missing_dataset_behavior.py::test_classical_ml_load_trained_models_missing PASSED [ 30%]
tests/test_modules_contracts.py::test_eda_summary_contract PASSED        [ 35%]
tests/test_modules_contracts.py::test_eda_preprocessing_contract PASSED  [ 40%]
tests/test_modules_contracts.py::test_classical_ml_pipeline_contract PASSED [ 45%]
tests/test_modules_contracts.py::test_regression_pipeline_contract PASSED [ 50%]
tests/test_modules_contracts.py::test_cnn_prep_and_pytorch_contract PASSED [ 55%]
tests/test_planning.py::test_initial_and_goal_states PASSED              [ 60%]
tests/test_planning.py::test_heuristic_admissibility PASSED              [ 65%]
tests/test_planning.py::test_valid_action_prerequisites PASSED           [ 70%]
tests/test_planning.py::test_apply_action_transitions PASSED             [ 75%]
tests/test_planning.py::test_bfs_search PASSED                           [ 80%]
tests/test_planning.py::test_dfs_search PASSED                           [ 85%]
tests/test_planning.py::test_astar_search PASSED                         [ 90%]
tests/test_planning.py::test_path_validation PASSED                      [ 95%]
tests/test_planning.py::test_run_all_algorithms PASSED                   [100%]

============================= 20 passed in 4.83s ==============================
```

---

## 8. Deployment to Streamlit Community Cloud

This repository is prepared for zero-configuration deployment to **Streamlit Community Cloud**:
1. Commit the `CIL_Integrated_Project` directory to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io/), click **New app**.
3. Select your repository and branch.
4. Set Main file path: `CIL_Integrated_Project/app.py` (or `app.py` if deployed from folder root).
5. Deploy! The application will launch with Task 3 planning active and provide clear prompts to supply `ai4i2020.csv`.

---

## 9. Next Steps When the Dataset Becomes Available

When `ai4i2020.csv` is obtained, execute the following steps:
1. **Place File:** Copy `ai4i2020.csv` into `CIL_Integrated_Project/data/raw/` (or upload via the web UI).
2. **Execute Preprocessing:** In Task 4 page, click **Run Preprocessing** to generate `cleaned_predictive_maintenance.csv`.
3. **Train Classifiers:** In Task 5 page, click **Train All 4 Classifiers** to verify live classification accuracy against historical notebook benchmarks (RF ~98.4%).
4. **Evaluate Deep Learning:** In Task 7 page, execute 1D CNN training to verify convergence and ROC-AUC curve (~0.980).
5. **Inspect Live Activations:** In Task 8 page, inspect live activation heatmaps on actual test failure instances.
