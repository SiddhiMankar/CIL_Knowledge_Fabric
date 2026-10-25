# MachineGuard — Predictive Maintenance Intelligence

[![Python 3.10-3.14](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.58.0%2B-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-55%20passed-brightgreen.svg)](tests/)

**MachineGuard** is an industrial predictive-maintenance decision-support platform built by unifying laboratory tasks 3 through 8 of the CIL Integrated Project into an auditable, transparent, multi-model intelligence system.

MachineGuard features **user-selectable Machine Learning (ML) and Deep Learning (DL) models**, allowing engineers to choose the predictive architecture, compare model outputs across identical sensor telemetry, and inspect validated model-specific explanations.

The system addresses five core operational capabilities:
1. **Model Selection:** Explicit user selection between ML (Random Forest, Gradient Boosting, Logistic Regression) and DL (1D CNN) architectures.
2. **Predicted Failure Risk:** Real inference against persisted artifacts without runtime retraining.
3. **Model-Specific Explainability:** Tree-path probability deltas (RF/GB), linear log-odds attributions (LR), Conv1D filter activation heatmaps (CNN), and interactive what-if counterfactual sensitivity curves.
4. **Multi-Model Consensus & Comparison:** Simultaneous side-by-side evaluation of identical machine inputs across all available models.
5. **Decision Support & Actionable Maintenance:** Objective risk tiering (NOMINAL, MONITORING, ELEVATED, CRITICAL) connecting sensor violations to maintenance protocols.

---

## 1. Supported Model Inventory

All models are registered in the centralized **Model Registry** (`modules/model_registry.py`), verified for availability, and use standardized leakage-free preprocessing fitted strictly on training data:

| Model ID | Display Name | Category | Architecture / Specs | Operating Threshold | Explanation Technique |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `random_forest` | **Random Forest Classifier** | **ML** | 100 balanced trees, max depth 12 | `0.35` (Production Default) | Exact Tree-Path Probability Delta (MDI) |
| `gradient_boosting` | **Gradient Boosting Classifier** | **ML** | 100 shallow trees, lr=0.1, max depth 4 | `0.35` | Boosting Stage Leaf Attribution |
| `logistic_regression` | **Logistic Regression (Balanced)** | **ML** | L2 regularization, inverse class weights | `0.50` | Linear Feature Log-Odds Contributions |
| `cnn_1d` | **1D Convolutional Neural Network** | **DL** | Conv1D(32)->MaxPool->Conv1D(64)->AvgPool->Dense(64) | `0.50` | Conv1D Filter Activation Maps & What-If |

---

## 2. Core Modules and System Workflows

* **Machine Assessment (Primary Workflow):**
  - **Select Prediction Model:** Prominent model selector grouped into ML and DL with detailed architecture notes, prediction capabilities, and limitations.
  - **Validated Telemetry Form:** Physical sensor inputs with strict boundary, NaN, and infinity validation.
  - **Demonstration Profiles:** Test-set operating conditions (Nominal, Critical Wear, Overstrain, Heat Dissipation).
  - **Inference Execution:** The chosen model executes the prediction; no silent fallback occurs.
  - **Model-Specific Explanations:** Tailored attribution plots (tree-path for RF/GB, linear impact for LR, activation heatmaps for CNN).
  - **Counterfactual What-If Simulator:** Dynamically recalculates sensitivity curves using the active model.
  - **Multi-Model Comparison:** Evaluates identical inputs across all models with objective disagreement analysis.
* **Batch Assessment (CSV Upload):**
  - Select any registered model for batch processing.
  - Robust row-level validation: valid rows receive predictions while invalid rows are flagged without crashing the batch.
  - Appends `Selected_Model`, `Model_Type`, `Failure_Probability`, `Predicted_Failure`, `Risk_Level`, `Decision_Threshold`, and `Prediction_Status`.
  - One-click CSV export.
* **Assessment History:**
  - Auditable persistent logs (`artifacts/assessment_history.json`) tracking `model_id`, `model_name`, and `model_type`.
  - Filter history records by model.
  - Full backward compatibility for legacy records without model fields.
* **Model Performance Dashboard:**
  - Model selector allowing inspection of individual ROC-AUC, PR-AUC, F1, Precision, Recall, and Confusion Matrices.
  - Comprehensive multi-model benchmark comparison table on the common held-out test split.
* **Experiments (Tasks 3–8):**
  - Preserved coursework tasks: AI Planning (BFS/DFS/A*), EDA, Classical ML & Clustering, Regression Diagnostics, 1D CNN architecture, and Layer Interpretability.
* **Ethics, Limitations & Responsible AI:**
  - Transparent documentation of class imbalance (3.39% failure rate), synthetic data limitations, domain shift, and human-in-the-loop safety oversight.

---

## 3. Quick Start Commands (Windows PowerShell)

```powershell
# Navigate to project directory
cd .\CIL_Integrated_Project\

# Run all automated tests (expected: 55 passed, 0 failed)
pytest -v

# Launch MachineGuard Streamlit application
streamlit run app.py
```
Application URL: `http://localhost:8501`

To retrain and regenerate all model artifacts from scratch:
```powershell
python -m modules.model_trainer
```

---

## 4. Benchmark Performance Summary (Held-Out Test Set, 2,000 samples)

All models evaluated strictly on the exact same held-out test split (2,000 samples, 68 failures, stratified 80/20 split) with complete target leakage exclusion:

| Model | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.9672** | **0.8080** | **0.7934** | **0.7059** | **0.9057** | **0.9875** |
| **Random Forest (Production)** | **0.9696** | **0.7305** | **0.6753** | **0.7647** | **0.6047** | **0.9785** |
| **1D CNN (PyTorch)** | **0.9604** | **0.5153** | **0.3158** | **0.9265** | **0.1903** | **0.8635** |
| **Logistic Regression (Balanced)** | **0.9069** | **0.3816** | **0.2414** | **0.8235** | **0.1414** | **0.8240** |

*Note on Operating Thresholds:* Production Random Forest is tuned to threshold `0.35` to prioritize recall/safety (catching 52 of 68 breakdowns vs 43 at threshold 0.50), reflecting the asymmetric cost of catastrophic breakdown (~$3,500) vs preventive inspection (~$150).

---

## 5. Automated Test Coverage (55 Tests, 0 Failures)

| Test File | Tests | Coverage |
| :--- | :--- | :--- |
| `test_model_selection.py` | 13 | Registry resolution, multi-model execution, explanations, comparison, batch robustness, history metadata |
| `test_machineguard_core.py` | 22 | Inference engine, boundary validation, decision layer, history persistence, schema mismatch |
| `test_planning.py` | 9 | BFS, DFS, A* optimality, path cost validation |
| `test_modules_contracts.py` | 5 | EDA, classical ML, regression, CNN interface contracts |
| `test_config.py` | 3 | Path configuration and schema definitions |
| `test_missing_dataset_behavior.py` | 2 | Graceful degradation without raw CSV |
| `test_app_startup.py` | 1 | Streamlit app compilation smoke test |

---

## 6. Environment & Compatibility

- **OS / Shell:** Windows PowerShell
- **Python Version:** Python 3.10 – 3.14 (Verified on Python 3.14.0)
- **Deep Learning Backend:** PyTorch (`torch 2.13.0+cpu`) with graceful exception handling
- **Matplotlib backend:** Headless `matplotlib.use('Agg')` configured across all modules
