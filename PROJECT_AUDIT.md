# Project Audit and Integration Plan: CIL Tasks 3–8

**Date:** 2026-10-20
**Project:** CIL Integrated Project (Tasks 3–8) → **KnowledgeFabric / MachineGuard**
**Environment:** Windows 11, Python 3.14.0, Streamlit 1.66.0, Scikit-learn 1.9.1, PyTorch 2.13.0
**Status:** ✅ Integration Complete — 42 Automated Tests Passing. Dataset Present. Production Model Deployed.

---

## 1. Executive Summary

This audit inspects and evaluates three Jupyter notebooks containing coursework implementations for **Tasks 3 through 8**:
- `CIL3.ipynb` (Task 3: State Space Search & AI Planning)
- `CIL456.ipynb` (Tasks 4, 5, 6: Exploratory Data Analysis, Classical Machine Learning, Clustering, and Regression)
- `CIL78 (1).ipynb` (Tasks 7, 8: 1D Convolutional Neural Network and Model Interpretability)

All three notebooks were parsed and analyzed at the cell level. **All tasks (3 through 8) are accounted for in the existing notebooks.** However, each notebook was originally developed as an isolated Google Colab script with hardcoded dependencies, in-memory execution chains, and inline model training.

Because the underlying predictive maintenance dataset (`ai4i2020.csv`) is temporarily unavailable and Python 3.14 is installed on the host system (where TensorFlow is currently unavailable), the integration architecture has been structured to:
1. Allow **Task 3 (AI Planning)** to run immediately and interactively without external data.
2. Abstract **Tasks 4–8** into modular, testable Python packages with safe fallback states, informative user guidance, and zero application crash risk when datasets or heavy ML frameworks are absent.
3. Decouple model training from inference and UI rendering.
4. Prepare standard artifact persistence so that once `ai4i2020.csv` is uploaded, models can be trained, evaluated, and saved to disk once, enabling instant runtime inference.

---

## 2. Notebook-by-Notebook Audit Findings

### 2.1 Notebook 1: `CIL3.ipynb`
* **File Size:** ~56 KB | **Cells:** 25 code cells, 0 markdown cells
* **Task Coverage:** **Task 3 — State Space Search & Maintenance Planning**
* **External Dataset Requirement:** None. Completely self-contained synthetic domain model.

#### Technical Details:
* **Domain Formulation:**
  * **Equipment:** Industrial Pump P-101.
  * **Fault Context:** Suspected Bearing Wear.
  * **State Representation:** 8-element boolean tuple:
    `state = (bearing_inspected, bearing_worn_confirmed, bearing_replaced, lubrication_checked, vibration_tested, temperature_tested, pump_tested, pump_operational)`
  * **Initial State:** `(False, False, False, False, False, False, False, False)`
  * **Goal State:** `(True, True, True, False, True, True, True, True)` (Lubrication is optional).
* **Action Space & Costs (9 Actions):**
  * `Inspect Bearing` (cost: 1)
  * `Confirm Bearing Wear` (cost: 1)
  * `Check Lubrication` (cost: 1)
  * `Lubricate Bearing` (cost: 2)
  * `Replace Bearing` (cost: 5)
  * `Run Vibration Test` (cost: 2)
  * `Run Temperature Test` (cost: 2)
  * `Test Pump` (cost: 2)
  * `Restore Pump Operation` (cost: 1)
* **Search Algorithms Implemented:**
  1. **Breadth-First Search (BFS):** FIFO queue (`collections.deque`), optimal for unweighted step counts.
  2. **Depth-First Search (DFS):** LIFO stack, explores deep paths first.
  3. **A\* Search:** Priority queue (`heapq`), evaluated with evaluation function $f(n) = g(n) + h(n)$.
* **Heuristic Function:**
  * Admissible remaining-cost heuristic summing the cost of all uncompleted mandatory goal predicates:
    $$h(s) = \sum_{a \in \text{unmet mandatory actions}} \text{cost}(a)$$
* **Benchmark Results Recorded in Notebook:**
  | Algorithm | Solution Found | Actions | Path Cost | Nodes Expanded | Execution Time (s) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **BFS** | True | 7 | 14 | 16 | ~0.00017 |
  | **DFS** | True | 8 | 15 | 9 | ~0.00008 |
  | **A\*** | True | 7 | 14 | 9 | ~0.00014 |
* **Path Validation:** `validate_path(initial_state, actions)` verifies transition legality and goal attainment.
* **Readiness:** **100% Complete.** Ready for direct extraction into `modules/planning.py`.

---

### 2.2 Notebook 2: `CIL456.ipynb`
* **File Size:** ~745 KB | **Cells:** 23 code cells, 2 markdown cells
* **Task Coverage:**
  * **Task 4: Exploratory Data Analysis & Data Preprocessing**
  * **Task 5: Feature Selection, Supervised Classification & Unsupervised Clustering**
  * **Task 6: Tool Wear Regression Analysis**
* **External Dataset Requirement:** `ai4i2020.csv` (10,000 rows × 14 columns).

#### Technical Details by Task:
* **Task 4 (EDA & Preprocessing):**
  * Expected columns: `['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']`
  * Missing value & duplicate row checks (0 missing, 0 duplicates confirmed in dataset).
  * Outlier and distribution visualizations: Boxplots and KDE histograms for 5 numerical features (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).
  * Correlation heatmap (`sns.heatmap`) across numerical features.
  * Cleaning: Drops `['UDI', 'Product ID']`. Scales the 5 numerical features with `StandardScaler` and outputs `cleaned_predictive_maintenance.csv`.
* **Task 5 (Supervised & Unsupervised Learning):**
  * Input features: `Type` (categorical) + 5 physical measurements.
  * Target: `Machine failure` (binary).
  * Encoding: `LabelEncoder` for `Type` (`H -> 0`, `L -> 1`, `M -> 2`).
  * Feature Importance: `RandomForestClassifier(random_state=42)` feature importances (`Torque [Nm]` and `Rotational speed [rpm]` identified as top contributors).
  * Train/Test Split: 80/20 random split (`random_state=42`).
  * 4 Supervised Classification Models:
    1. `LogisticRegression(random_state=42)` (Accuracy: 0.9735)
    2. `DecisionTreeClassifier(random_state=42, max_depth=5)` (Accuracy: 0.9790)
    3. `RandomForestClassifier(n_estimators=100, random_state=42)` (Accuracy: 0.9840)
    4. `SVC(kernel='rbf', random_state=42)` (Accuracy: 0.9760)
  * Evaluation: Confusion matrix heatmaps and classification reports.
  * Unsupervised Clustering:
    1. **K-Means Clustering:** `KMeans(n_clusters=3, random_state=42, n_init=10)` plotted on `Rotational speed` vs `Torque`.
    2. **Hierarchical Clustering:** Agglomerative linkage (`method='ward'`) on a 1,000-sample subset, visualised via `dendrogram(p=30, truncate_mode='lastp')`.
* **Task 6 (Tool Wear Regression):**
  * Target: `Tool wear [min]` (continuous).
  * Input features: Remaining 5 encoded features (`Type`, temperatures, speed, torque).
  * Model: `LinearRegression()`.
  * Results recorded in notebook: $\text{MSE} = 1.0554$, $R^2 = -0.0015$.
  * Visualization: Actual vs. Predicted scatter plot with reference diagonal.
  * Note on finding: $R^2 \approx 0$ indicates tool wear is poorly modeled by a simple linear combination of instantaneous operating parameters, a valuable analytical finding from the notebook.
* **Code Issues & Notebook Dependencies:**
  * Uses Colab-specific `files.upload()`.
  * In Cell 11, `StandardScaler` was applied to the entire dataset *before* splitting in Cell 16 (data leakage flaw in original notebook; documented below).
  * Cell 5 has a recorded `NameError: name 'df' is not defined` due to out-of-order cell execution in Colab.

---

### 2.3 Notebook 3: `CIL78 (1).ipynb`
* **File Size:** ~1.81 MB | **Cells:** 21 code cells, 17 markdown cells
* **Task Coverage:**
  * **Task 7: 1D Convolutional Neural Network (CNN) for Predictive Maintenance**
  * **Task 8: Model Interpretability (Conv1D Feature Maps & LIME Explanations)**
* **External Dataset Requirement:** `ai4i2020.csv`.

#### Technical Details by Task:
* **Task 7 (1D CNN Architecture & Evaluation):**
  * Target: `Machine failure`.
  * Features Excluded: `['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']` (strictly prevents data leakage from failure sub-types).
  * Encoding: One-hot encoding `pd.get_dummies(X, columns=['Type'], drop_first=True)` creating 7 features:
    `['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M']`.
  * Train/Test Split: 80/20 **Stratified** split (`stratify=y`, `random_state=42`) yielding 8,000 train / 2,000 test samples.
  * Scaling: `StandardScaler` fitted **strictly on `X_train`** and applied to `X_test` (correct leakage prevention).
  * Reshaping: `(samples, 7, 1)` for 1D convolution.
  * Imbalance Handling: Compute class weights:
    $$\text{Class 0: } \approx 0.52, \quad \text{Class 1: } \approx 14.76 \quad (\sim 3.39\% \text{ failure rate})$$
  * Model Architecture:
    * `Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(7, 1), name='conv1d_1')`
    * `BatchNormalization(name='batch_norm_1')`
    * `MaxPooling1D(pool_size=2, name='max_pooling_1')`
    * `Conv1D(filters=64, kernel_size=2, activation='relu', name='conv1d_2')`
    * `BatchNormalization(name='batch_norm_2')`
    * `GlobalAveragePooling1D(name='global_avg_pooling')`
    * `Dense(64, activation='relu', name='dense_1')`
    * `Dropout(0.5, name='dropout_1')`
    * `Dense(1, activation='sigmoid', name='output_layer')`
  * Optimizer & Loss: `Adam(learning_rate=0.001)`, `binary_crossentropy`.
  * Metrics: `accuracy`, `Precision`, `Recall`, `AUC`.
  * Callbacks: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`.
  * Evaluation on Test Set:
    * CNN Accuracy: 0.966, Precision: 0.500, Recall: 0.647, F1: 0.564, ROC-AUC: 0.98, PR-AUC: 0.58.
    * Confusion matrix, ROC curve, and Precision-Recall curve visualizations.
  * Baseline Comparison: Logistic Regression (`solver='liblinear'`, `class_weight='balanced'`).
    * Baseline Accuracy: 0.824, Precision: 0.142, Recall: 0.824, ROC-AUC: 0.90.
    * Comparison demonstrates CNN's superior precision (0.500 vs 0.142) and lower false positive rate on imbalanced data.
* **Task 8 (Interpretability & Explainability):**
  1. **CNN Activation / Feature Map Visualization:**
     * Multi-output Keras Model: `Model(inputs=model.layers[0].input, outputs=[conv1d_1.output, conv1d_2.output])`.
     * Tested on 4 operational cases:
       - True Positive (TP, index 21): Correctly predicted failure.
       - True Negative (TN, index 1): Correctly predicted normal operation.
       - False Positive (FP, index 0): Normal unit flagged as failure.
       - False Negative (FN, index 17): Failure missed by network.
     * Generates input bar chart and activation heatmaps across all 32 and 64 filters.
  2. **Comparative Conceptual Analysis:**
     * Linear model coefficients (direct, linear, high transparency) vs. CNN feature maps (hierarchical, non-linear pattern detectors, abstract).
  3. **LIME (Local Interpretable Model-agnostic Explanations):**
     * `lime_tabular.LimeTabularExplainer` configured for tabular data in scaled feature space.
     * Custom prediction wrapper `predict_fn_cnn(x_scaled)` adapting 2D tabular features into 3D CNN tensor.
     * Feature attribution plots for TP and TN cases highlighting how torque, tool wear, and speed drive failure probability.
* **Code Issues & Notebook Dependencies:**
  * Contains Colab commands (`!pip install lime -q`, `%matplotlib inline`).
  * Cells 15–19 and 21–25 are duplicate runs of earlier training cells where parameters were adjusted.
  * Requires TensorFlow/Keras and LIME.

---

## 3. Discrepancies and Methodological Differences Between Notebooks

| Aspect | Notebook `CIL456.ipynb` (Tasks 4–6) | Notebook `CIL78 (1).ipynb` (Tasks 7–8) | Reconciled Standard for Integrated App |
| :--- | :--- | :--- | :--- |
| **Categorical Encoding** | `LabelEncoder` (`H=0, L=1, M=2`) $\rightarrow$ 6 total features | `pd.get_dummies(..., drop_first=True)` $\rightarrow$ 7 total features (`Type_L`, `Type_M`) | **Configurable:** Support both modes. Preserves Task 4–6 compatibility (6 features) while feeding 7 features to the CNN/LIME pipeline. |
| **Data Leakage in Scaling** | `StandardScaler` fitted on entire dataset prior to split | `StandardScaler` fitted solely on `X_train` | **Methodological fix:** Fit scalers on `X_train` only to prevent data leakage, while maintaining identical feature distribution. |
| **Train/Test Splitting** | Simple split (`test_size=0.2, random_state=42`) | Stratified split (`stratify=y, random_state=42`) | Provide option for stratified split as best practice for imbalanced failure targets. |
| **Failure Sub-type Features** | Dropped in cleaning (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) | Dropped explicitly to prevent leakage | Maintained consistently across all modules. |

---

## 4. Host Environment & Dependency Audit

Inspection of the host system revealed:
* **Operating System:** Windows (PowerShell)
* **Python Runtime:** Python 3.14.0
* **Installed Packages:**
  * `streamlit` (1.58.0) — **Installed & Verified**
  * `scikit-learn` (1.8.0) — **Installed & Verified**
  * `pandas` (3.0.1) — **Installed & Verified**
  * `numpy` (2.4.2) — **Installed & Verified**
  * `matplotlib` (3.10.8) & `seaborn` (0.13.2) — **Installed & Verified**
  * `scipy` (1.17.1) — **Installed & Verified**
  * `pytest` (9.1.1) — **Installed & Verified**
  * `torch` (2.13.0) — **Installed & Verified**
* **Framework Gaps:**
  * `tensorflow`: **Not installed.** Official TensorFlow wheels are currently unavailable for Python 3.14 on Windows.
  * `lime`: **Not installed.**

### Architectural Implication:
To ensure the application is **fully deployable, crash-proof, and independent of external runtime constraints**, we will:
1. Make TensorFlow and LIME **optional/lazy imports** wrapped in robust diagnostic guards.
2. Provide a native PyTorch equivalent or lightweight numerical forward-pass emulator for the 1D CNN architecture if TensorFlow cannot be installed on Python 3.14, ensuring feature maps and activations can be explored interactively even when TF is absent!
3. Store pre-computed sample evaluation records and network architectures in `artifacts/metadata.json` so the user can inspect actual historical training curves and confusion matrices directly from the UI without having to retrain.

---

## 5. Target Project Architecture

```text
CIL_Integrated_Project/
├── app.py                         # KnowledgeFabric/MachineGuard Streamlit application
├── config.py                      # Central paths, feature lists, and constants
├── notebooks/                     # Read-only copies of the 3 original notebooks
│   ├── CIL3.ipynb
│   ├── CIL456.ipynb
│   └── CIL78.ipynb
├── modules/
│   ├── __init__.py
│   ├── model_trainer.py           # Leakage-free training & artifact persistence pipeline
│   ├── inference.py               # Production inference engine with schema validation
│   ├── decision_layer.py          # Risk tier mapping & maintenance recommendations
│   ├── history_manager.py         # Assessment history logger & CSV exporter
│   ├── interpretability.py        # Task 8: Tree attributions, what-if, activation maps
│   ├── planning.py                # Task 3: State Space Search (BFS, DFS, A*, validator)
│   ├── eda.py                     # Task 4: Dataset loading, cleaning, statistics, plots
│   ├── classical_ml.py            # Task 5: Feature importance, 4 classifiers, clustering
│   ├── regression.py              # Task 6: Tool wear linear regression & diagnostics
│   └── cnn_model.py               # Task 7: 1D CNN architecture, training & evaluation
├── data/
│   ├── raw/ai4i2020.csv           # AI4I 2020 dataset (present, 10,000 x 14)
│   └── processed/                 # Generated scaled & cleaned datasets
├── artifacts/
│   ├── models/
│   │   ├── production_model.joblib # Persisted Random Forest + ColumnTransformer (v1.0.0)
│   │   └── test_set_split.joblib   # Held-out test split (2,000 samples)
│   ├── assessment_history.json    # Persisted assessment audit log
│   ├── metadata.json              # Verified held-out metrics, schema, threshold curves
│   └── figures/                   # Cached figures and charts
├── tests/                         # Automated test suite (42 tests, 0 failures)
│   ├── test_machineguard_core.py  # 22 tests: inference, decision, history, validation
│   ├── test_planning.py           # 9 tests: Task 3 algorithmic unit tests
│   ├── test_modules_contracts.py  # 5 tests: module interface contracts
│   ├── test_config.py             # 3 tests: path and environment resolution
│   ├── test_missing_dataset_behavior.py  # 2 tests: graceful degradation
│   └── test_app_startup.py        # 1 test: Streamlit compilation smoke test
├── requirements.txt               # Pinned dependencies
├── .gitignore
├── README.md
└── PROJECT_AUDIT.md               # This document
```

---

## 6. Completed Milestones

### Phase 2: Project Structure Setup ✔
* `CIL_Integrated_Project` hierarchy created with all directories.
* Notebooks copied into `notebooks/`; originals preserved in repo root.
* `.gitignore` and `config.py` configured.

### Phase 3: Module Refactoring ✔
* `modules/planning.py`: BFS, DFS, A*, heuristic, path validator.
* `modules/eda.py`: Data loading, descriptive stats, outlier boxplots, correlation heatmap, scaling.
* `modules/classical_ml.py`: LogReg, DT, RF, SVM; K-Means; Ward hierarchical clustering.
* `modules/regression.py`: Tool wear linear regression and residual analysis.
* `modules/cnn_model.py`: 1D CNN definition, train/test metrics, ROC/PR curves, PyTorch backend.
* `modules/interpretability.py`: Tree-path attributions, what-if counterfactual simulator, activation maps, LIME.
* `modules/model_trainer.py`: Leakage-free pipeline — ColumnTransformer, stratified split, persists artifact.
* `modules/inference.py`: Production engine — schema validation, physical bounds checks, batch/single prediction.
* `modules/decision_layer.py`: Four-tier risk classification (NOMINAL/MONITORING/ELEVATED/CRITICAL).
* `modules/history_manager.py`: Assessment logger with threshold & model version tracking, CSV export.

### Phase 4: Streamlit UI Design ✔
* Five primary navigation sections:
  1. Machine Assessment — form, demo cases, batch upload, inference, what-if simulator.
  2. Assessment History — auditable log, CSV export.
  3. Model Performance — held-out metrics, confusion matrix, PR-AUC, threshold curves, subgroup analysis.
  4. Experiments (Tasks 3–8) — preserved interactive lab tasks.
  5. About, Ethics & Limitations — class imbalance, synthetic data caveats, safety guardrails.

### Phase 5: Artifacts, Diagnostics & Error Handling ✔
* `artifacts/models/production_model.joblib` — trained Random Forest (ROC-AUC 0.9696).
* `artifacts/models/test_set_split.joblib` — held-out 2,000-sample test split.
* `artifacts/metadata.json` — verified held-out metrics, threshold curves, subgroup performance.
* `artifacts/assessment_history.json` — persistent assessment audit log.
* Matplotlib `Agg` backend set in all modules to prevent Tk crashes in headless environments.
* PyTorch imported with `(ImportError, OSError)` guard for Windows DLL compatibility.

### Phase 6: Automated Testing & Verification ✔
* **42 tests, 0 failures** across 6 test files.
* Covers: inference engine, input validation, decision layer, history manager, batch/single prediction,
  missing-model controlled failure, corrupt-artifact detection, feature schema mismatch,
  BFS/DFS/A* optimality, EDA/ML/regression/CNN module contracts, app compilation.

### Phase 7: Deployment Documentation ✔
* `README.md` updated with correct test counts, versions, full module tree, and current status.
* `PROJECT_AUDIT.md` updated to reflect completed state and all fixes applied.
* `CIL_Integrated_Project/README.md` and `CIL_Integrated_Project/PROJECT_AUDIT.md` fully current.

---

## 7. Environment (Current State)

* **Operating System:** Windows 11 (PowerShell)
* **Python Runtime:** Python 3.14.0
* **Installed Packages:**
  * `streamlit` (1.66.0) — Installed & Verified
  * `scikit-learn` (1.9.1) — Installed & Verified
  * `pandas` (3.0.1) — Installed & Verified
  * `numpy` (2.4.2) — Installed & Verified
  * `matplotlib` (3.10.8) & `seaborn` (0.13.2) — Installed & Verified
  * `scipy` (1.17.1) — Installed & Verified
  * `joblib` — Installed & Verified
  * `pytest` (9.1.1) — Installed & Verified
  * `torch` (2.13.0) — Installed & Verified
* **Framework Notes:**
  * `tensorflow`: Not installed (unavailable for Python 3.14 on Windows). PyTorch backend used.
  * `lime`: Optional. LIME explainability gracefully disabled if not installed.
