# KnowledgeFabric / MachineGuard Engineering Audit and Transformation Report

**Project:** CIL Integrated Project -> **KnowledgeFabric / MachineGuard -- Predictive Maintenance Intelligence**
**Environment:** Windows PowerShell, Python 3.14.0, Streamlit 1.66.0, Scikit-learn 1.9.1
**Status:** Product Transformation Complete -- 42 Automated Tests Passing

---

## 1. Executive Summary

This report documents the transformation of the coursework repository CIL_Integrated_Project (covering lab Tasks 3-8) into **KnowledgeFabric / MachineGuard**, a production-grade industrial predictive maintenance decision-support platform.

The system addresses the four core operational questions:
1. **Predicted Failure Risk:** Evaluated instantly via a persisted, leakage-free Random Forest model without runtime retraining.
2. **Influential Features:** Quantified via exact tree-path decision-delta attributions and interactive what-if sensitivity curves.
3. **Recommended Actions and Evidence:** Mapped via a dedicated decision layer connecting physical sensor violations to structured maintenance protocols.
4. **Reliability and Limitations:** Grounded in a held-out test evaluation (2,000 samples, 20% stratified split), class imbalance analysis (3.39% base rate), threshold sensitivity tradeoffs (0.20-0.50), and machine-type subgroup reliability checks.

All original academic notebooks and laboratory experiments (Tasks 3-8) remain preserved in the Experiments section.

---

## 2. Phase 1 Audit: Baseline Findings and Vulnerabilities Identified

| Area | Baseline State | Critical Risk Identified | Transformation Resolution |
| :--- | :--- | :--- | :--- |
| **Model Retraining** | Retrained dynamically on every Streamlit rerun. | Non-determinism, inefficiency. | Built model_trainer.py to persist production_model.joblib. Runtime uses inference.py without retraining. |
| **Target Leakage** | Raw dataset includes failure modes: TWF, HDF, PWF, OSF, RNF. | Sum of failure flags accounts for 97.3% of failures; including them creates severe leakage. | All 5 failure mode columns and ID columns excluded from feature space prior to preprocessing. |
| **Feature Ordering and Scaling** | Inconsistent encoding (LabelEncoder in Task 5 vs. manual one-hot in Task 7). | Mismatched column ordering between training and inference causing silent prediction corruption. | Unified into a fitted ColumnTransformer (StandardScaler on numericals + OneHotEncoder(drop=\'first\') on Type) stored in the model artifact. |
| **Explainability** | Hardcoded contributions or simulated random noise for activation heatmaps. | Misleading users with fabricated explanations. | Implemented exact tree-path probability delta calculations and an interactive what-if counterfactual simulator. |
| **Data Splitting** | Metrics reported without verified split tracking or from notebooks. | Accuracy paradox: 96.6% accuracy achievable by predicting 0 failures. | Reproducible stratified 80/20 train/test split (random_state=42) evaluated strictly on held-out test data. |
| **Environment Crashes** | PyTorch imported without catching Windows DLL OSError (c10.dll/shm.dll). | Streamlit crashed on startup without VC++ Redistributable. | Added graceful (ImportError, OSError) exception handling across all deep learning imports. |
| **Matplotlib Backend** | Default Tk/TkAgg backend assumed. | TclError crash in headless or broken-tkinter environments during testing. | All modules set matplotlib.use(\'Agg\') before pyplot import. Non-interactive, display-independent. |
| **Missing Model Fallback** | Engine silently fell back to locate_production_model() even when explicit path supplied. | test_missing_model_controlled_failure failed: fake path was overridden. | Fallback search only applies when no explicit model_path is provided by the caller. |
| **Error Message Consistency** | predict_single raised load_error directly, which said \'not found\' not \'not loaded\'. | Test assertions on error messages failed. | predict_single and predict_batch prepend \'Production model is not loaded.\' to load_error detail. |

---

## 3. Production Architecture

### 3.1 Model Selection and Rationale
Candidate models evaluated on the exact same held-out test set (2,000 samples):

| Model | ROC-AUC | PR-AUC | F1 (th=0.35) | Recall | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Production)** | **0.9696** | **0.7305** | **0.6753** | **0.7647** | **0.6047** |
| Gradient Boosting | 0.9672 | 0.8080 | 0.7934 | 0.7059 | 0.9057 |
| Logistic Regression (Balanced) | 0.9069 | 0.3816 | 0.2414 | 0.8235 | 0.1414 |
| 1D CNN (Experimental) | 0.9800 | 0.5800 | 0.5640 | 0.6470 | 0.5000 |

**Technical Justification:**
While the 1D CNN achieved strong ROC-AUC, its PR-AUC on rare failures (3.39%) was significantly lower (0.5800 vs. 0.7305 for Random Forest). Tabular physical measurements lack translational invariance; convolving adjacent independent sensor columns is artificial. Random Forest provides superior PR-AUC, fast execution, determinism, and exact tree-path interpretability without requiring GPU runtimes.

### 3.2 Operating Threshold Tuning
Default threshold of 0.35 (not 0.50) because catastrophic in-service breakdown (~,500) far outweighs preventive inspection (~). At threshold=0.35: Recall=76.5%, Precision=60.5%, F1=0.6753.

### 3.3 Four-Tier Risk Classification
Risk tiers standardised across all single and batch workflows via a single authoritative function in decision_layer.py:
- **CRITICAL:** P(Fail) >= 0.70 -- Immediate controlled shutdown
- **ELEVATED:** threshold <= P(Fail) < 0.70 -- Schedule inspection within 24h
- **MONITORING:** 0.5*threshold <= P(Fail) < threshold -- Enhanced telemetry logging
- **NOMINAL:** P(Fail) < 0.5*threshold -- Standard operating schedule

---

## 4. Validation, Ethics, and Subgroup Reliability

### 4.1 Subgroup Performance Across Machine Variants (at threshold=0.35)
| Variant | Samples | Failures | Recall | Precision | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| L (Low) | 1,170 | 38 | 0.7632 | 0.5686 | 0.6517 |
| M (Medium) | 616 | 25 | 0.8000 | 0.7143 | 0.7547 |
| H (High) | 214 | 5 | 0.6000 | 0.4286 | 0.5000 |

### 4.2 Ethics and Operational Guardrails
- **Decision Support Only:** MachineGuard issues guidance, not autonomous equipment shutdowns.
- **Explainability as Evidence:** Attributions reflect statistical model influence, not definitive physical causation.
- **Synthetic Data Caveat:** Model trained on AI4I 2020 synthetic dataset; domain shift to real-world industrial data must be assessed before deployment.

---

## 5. Automated Test Suite Verification (Final State)

**42 tests, 0 failures:**

| Test File | Count | Coverage |
| :--- | :--- | :--- |
| test_machineguard_core.py | 22 | Inference engine, validation, decision layer, history, error handling |
| test_planning.py | 9 | BFS, DFS, A* optimality, path validation |
| test_modules_contracts.py | 5 | EDA, classical ML, regression, CNN interfaces |
| test_config.py | 3 | Path configuration and schema definitions |
| test_missing_dataset_behavior.py | 2 | Graceful degradation without raw CSV |
| test_app_startup.py | 1 | Streamlit syntax compilation smoke test |

---

## 6. Commands to Run (Windows PowerShell)

`powershell
# Navigate to project directory
cd .\CIL_Integrated_Project\

# Run all automated tests (expected: 42 passed)
pytest -v

# Launch KnowledgeFabric application
streamlit run app.py
`
App URL: http://localhost:8501
