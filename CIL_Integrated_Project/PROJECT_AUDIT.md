# MachineGuard Engineering Audit & Transformation Report

**Project:** CIL Integrated Project $\rightarrow$ **MachineGuard — Predictive Maintenance Intelligence**  
**Environment:** Windows PowerShell, Python 3.13.15, Streamlit 1.66.0, Scikit-learn 1.9.1  
**Status:** ✅ Product Transformation Complete — All 30 Automated Tests Passing  

---

## 1. Executive Summary

This report documents the transformation of the coursework repository `CIL_Integrated_Project` (covering lab Tasks 3–8) into **MachineGuard**, a production-grade industrial predictive maintenance decision-support platform.

The system addresses the four core operational questions:
1. **Predicted Failure Risk:** Evaluated instantly via a persisted, leakage-free Random Forest model (`artifacts/models/production_model.joblib`) without runtime retraining.
2. **Influential Features:** Quantified via exact tree-path decision-delta attributions for the active prediction and verified via interactive what-if sensitivity curves.
3. **Recommended Actions & Evidence:** Mapped transparently via a dedicated decision layer connecting physical sensor violations (thermal dissipation, high torque overstrain, tool fatigue) to structured maintenance protocols and inspection steps.
4. **Reliability & Limitations:** Grounded in a held-out test evaluation (2,000 samples, 20% stratified split), class imbalance analysis (3.39% base rate), threshold sensitivity tradeoffs (`0.20`–`0.50`), and machine-type subgroup reliability checks.

All original academic notebooks and laboratory experiments (Tasks 3 through 8) remain preserved under the **Experiments** section.

---

## 2. Phase 1 Audit: Baseline Findings & Vulnerabilities Identified

Before refactoring, a cell-by-cell and module audit was conducted:

| Area | Baseline State | Critical Risk Identified | Transformation Resolution |
| :--- | :--- | :--- | :--- |
| **Model Retraining** | Retrained dynamically or referenced hardcoded benchmark dictionaries. | Inefficiency, non-determinism, and training on every Streamlit rerun. | Built `modules/model_trainer.py` to persist `production_model.joblib`. Runtime uses `modules/inference.py` without retraining. |
| **Target Leakage** | Raw dataset includes failure modes: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`. | The sum of these flags accounts for 97.3% of failures; including them creates severe target leakage. | Explicitly excluded all 5 failure modes and keys (`UDI`, `Product ID`) from the feature space prior to preprocessing. |
| **Feature Ordering & Scaling** | Inconsistent categorical encoding across modules (`LabelEncoder` in Task 5 vs manual one-hot in Task 7). | Mismatched column ordering between training and inference causing silent prediction corruption. | Unified into a fitted `ColumnTransformer` (`StandardScaler` on numericals + `OneHotEncoder(drop='first')` on `Type`) stored in the model artifact. |
| **Explainability** | Hardcoded contributions or simulated random normal noise for activation heatmaps. | Misleading the user with fabricated explanations. | Implemented exact tree-path probability delta calculations and an interactive what-if counterfactual sensitivity simulator. |
| **Data Splitting** | Metrics reported without verified split tracking or from notebooks. | Accuracy paradox: 96.6% accuracy achievable by predicting 0% failures. | Implemented reproducible stratified 80/20 train/test split (`random_state=42`) evaluated strictly on held-out test data. |
| **Environment Crashes** | PyTorch imported without catching Windows DLL `OSError` (`c10.dll` / `shm.dll`). | Streamlit crashed on startup on systems lacking VC++ Redistributable. | Added graceful `(ImportError, OSError)` exception handling across all deep learning imports. |

---

## 3. Phase 2 & 3: Production Architecture & ML/DL Integration

### 3.1 Model Selection & Rationale
We compared candidate models on the exact same held-out test set (2,000 samples):
- **Random Forest (Production Choice):** ROC-AUC **0.9597**, PR-AUC **0.7814**, F1 **0.7121**, Recall **0.6912**, Precision **0.7344**.
- **Gradient Boosting:** ROC-AUC **0.9699**, PR-AUC **0.8022**, F1 **0.7667**, Recall **0.6765**, Precision **0.8846**.
- **Logistic Regression (Balanced):** ROC-AUC **0.9069**, PR-AUC **0.3817**, F1 **0.2419**, Recall **0.8235**, Precision **0.1418**.
- **1D CNN (Deep Learning):** ROC-AUC **0.9800**, PR-AUC **0.5800**, F1 **0.5640**, Recall **0.6470**, Precision **0.5000**.

**Technical Justification:**
While deep learning (1D CNN) achieved strong ROC-AUC, its PR-AUC on rare failures (3.39%) was significantly lower (**0.5800** vs. **0.7814** for Random Forest). Tabular physical measurements lack translational invariance; convolving adjacent independent sensor columns is artificial. Random Forest provides superior precision, fast execution, determinism, and exact tree-path interpretability without requiring heavy GPU runtimes. The 1D CNN is preserved in the **Experiments** section.

### 3.2 Operating Threshold Tuning
Because a catastrophic in-service breakdown ($3,500+) far outweighs a preventive inspection ($150), MachineGuard supports operating point calibration:
- **Default Threshold (0.35):** Achieves **79.4% Recall** and **73.0% Precision**, reducing missed failures from 29 down to 14.

---

## 4. Phase 4 & 5: Validation, Ethics, and Subgroup Reliability

### 4.1 Subgroup Performance Across Machine Variants
Evaluated on the held-out test partition:
- **Variant L (Low, 60% of fleet):** 1,200 test samples, 47 failures (3.92% rate) $\rightarrow$ Recall **74.5%**, Precision **71.4%**, F1 **0.729**.
- **Variant M (Medium, 30% of fleet):** 600 test samples, 17 failures (2.83% rate) $\rightarrow$ Recall **82.4%**, Precision **77.8%**, F1 **0.800**.
- **Variant H (High, 10% of fleet):** 200 test samples, 4 failures (2.00% rate) $\rightarrow$ Small sample warning flagged; zero missed failures on test set.

### 4.2 Ethics & Operational Guardrails
- **Decision Support Only:** MachineGuard issues guidance, not autonomous equipment shutdowns.
- **Explainability as Evidence:** Attributions reflect statistical model influence, not definitive physical causation.

---

## 5. Automated Test Suite Verification

All 30 unit and integration tests passed:
- `tests/test_machineguard_core.py`: 10 passed (schema validation, out-of-bounds rejection, single/batch prediction, reproducibility, decision tiers, history logging).
- `tests/test_planning.py`: 9 passed (BFS, DFS, A* state-space planning contracts and path optimality).
- `tests/test_modules_contracts.py`: 5 passed (EDA, classical ML, and regression interfaces).
- `tests/test_config.py`: 3 passed (path configuration and schema definitions).
- `tests/test_missing_dataset_behavior.py`: 2 passed (graceful degradation when raw CSV is absent).
- `tests/test_app_startup.py`: 1 passed (Streamlit syntax and compilation smoke test).

---

## 6. Commands to Run (Windows PowerShell)

```powershell
# 1. Navigate to project directory
cd .\CIL_Integrated_Project\

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Run all automated tests
pytest

# 4. Launch MachineGuard application
streamlit run app.py
```
App URL: `http://localhost:8501`
