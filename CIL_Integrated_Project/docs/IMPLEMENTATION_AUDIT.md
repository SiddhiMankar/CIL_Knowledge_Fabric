# CIL Knowledge Fabric — Implementation Audit and Requirements Matrix

**Project:** CIL Knowledge Fabric — MachineGuard Predictive Maintenance Intelligence Platform  
**Repository:** https://github.com/SiddhiMankar/CIL_Knowledge_Fabric  
**Audit Date:** 2026-11-02  
**Auditor:** Senior AI/ML Engineer, QA Engineer & Technical Documentation Specialist  
**Host Environment:** Windows (PowerShell), Python 3.14.0, Streamlit 1.58.0, Scikit-Learn 1.8.0, PyTorch 2.13.0, NumPy 2.4.2, Pandas 3.0.1  

---

## 1. Executive Summary

This audit assesses the state of the CIL Knowledge Fabric repository against all ten syllabus tasks. The codebase historically consolidated three Jupyter notebooks (`CIL3.ipynb`, `CIL456.ipynb`, `CIL78 (1).ipynb`) into an initial Streamlit prototype (`CIL_Integrated_Project/app.py` and supporting modules). 

While algorithms from Tasks 3–8 had corresponding functions, critical architectural and engineering gaps exist:
1. **Task 1 (Knowledge Representation)** and **Task 2 (Rule-Based Inference Engine)** were completely absent as explicit software modules.
2. **Task 9 (Complete End-to-End Integration)** was incomplete: the user interface presented disconnected views rather than a unified interactive pipeline connecting input validation, model inference, rule-based decision-making, explainability, automated planning, and persistent history tracking.
3. **Task 10 (Validation, Ethics, Fairness & Subgroup Analysis)** lacked automated holdout validation reporting, input stress tests, and demographic/machine subgroup error analysis.
4. **Data Leakage & Preprocessing**: `modules/eda.py` fitted scalers on the full dataset before splitting. In addition, model artifacts were not serialized to disk, requiring inline training or static fallbacks.
5. **Execution Latency**: Matplotlib cache initialization under the Windows sandbox generated repeated warnings and 15–20s import delays, which can be mitigated with `MPLCONFIGDIR`.

The actual AI4I 2020 Predictive Maintenance dataset (`ai4i2020.csv`: 10,000 rows × 14 columns) has been provisioned and verified in `CIL_Integrated_Project/data/raw/ai4i2020.csv`.

---

## 2. Requirements-to-Implementation Matrix (Tasks 1–10)

| Task | Syllabus Requirement | Classification | Implementation Evidence / Relevant Files | Gaps & Identified Issues |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Knowledge Representation**: Formal domain schema (Machine, SensorReading, OperatingCondition, FailureType, PredictedFailureRisk, MaintenanceAction, MaintenanceRecommendation), constraints, validation logic, UI representation page. | **Missing** | None currently in `modules/`. | No formal ontology/frame representation; missing schema introspection; missing input constraint validation module; missing UI page. |
| **Task 2** | **Rule-Based Inference Engine**: Forward chaining inference system with structured IF–THEN rules (ID, name, conditions, action, priority, explanation), conflict resolution, transparent trace, and separation of ML probability from failure certainty. | **Missing** | None currently in `modules/`. Previous `app.py` had no rule engine. | Needs explicit `Rule` and `InferenceEngine` classes; needs conflict resolution; needs trace generation; needs dedicated UI view and tests. |
| **Task 3** | **AI Search & Planning**: 8-state representation, actions, BFS, DFS, A* with admissible heuristic, path validator, benchmark comparison. | **Implemented and verified** | `modules/planning.py`, `tests/test_planning.py`, `app.py` (Page 2). | Fully functional algorithms. Needs recommendation bridge connecting rule engine actions to planning scenarios, plus unreachable state handling tests. |
| **Task 4** | **Exploratory Data Analysis & Preprocessing**: Summary stats, missing/duplicate checks, distributions, outlier boxplots, correlation heatmap, target distribution, leakage-free scaling. | **Partially implemented** | `modules/eda.py`, `tests/test_modules_contracts.py`. | Scaling in `modules/eda.py` previously fitted on the entire dataset without prior train/test split (data leakage). Missing target distribution visualization and before-and-after cleaning stats. |
| **Task 5** | **Feature Engineering & Selection**: Categorical encoding, physical feature handling, feature transformations, feature selection vs. importance, consistent input matrix. | **Partially implemented** | `modules/classical_ml.py`, `config.py`. | Only LabelEncoder and raw RF importances present. Missing explicit feature engineering (temperature difference, power proxy, tool wear strain), and feature selection vs importance distinction. |
| **Task 6** | **Classical Machine Learning & Evaluation**: 5 classifiers (RF, DT, SVM, LogReg, Gradient Boosting), stratified train/test split, hyperparameter awareness, holdout evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, CM, failure-class metrics), misclassification analysis, Tool Wear Regression (MSE, RMSE, MAE, R²). | **Partially implemented** | `modules/classical_ml.py`, `modules/regression.py`. | Missing Gradient Boosting in `build_models`; classification summary only tracked Accuracy; missing ROC-AUC / PR-AUC and failure-class recall in summary table; missing detailed misclassification breakdown (TP, TN, FP, FN feature profiles). |
| **Task 7** | **Deep Learning Architecture**: 1D CNN architecture documentation (Conv1D, BatchNorm, MaxPool, GlobalAvgPool, Dense, Dropout), class weighting, holdout evaluation vs baseline LogReg, reproducible training and weight persistence. | **Implemented but needs verification** | `modules/cnn_model.py`, `artifacts/metadata.json`. | Native PyTorch implementation exists for Python 3.14. Needs pre-trained serialized weights saved to `artifacts/models/cnn_pytorch.pt` and holdout evaluation directly on real dataset. |
| **Task 8** | **Model Interpretability & Explainability**: Model-appropriate explanations (RF importance, linear coefficients, CNN activation maps on actual inputs, LIME surrogate), representative case studies (TP, TN, FP, FN), What-If interactive simulator. | **Partially implemented** | `modules/interpretability.py`. | Activation maps and LIME were based on hardcoded sample profiles. Missing interactive What-If simulator and live connection to loaded models. |
| **Task 9** | **Complete End-to-End Integration**: Unified workflow: Input -> Validation -> Preprocessing -> Model Selection (dynamic from registry) -> Inference -> Rule Engine -> Explanation -> Maintenance Planning -> Results Dashboard -> History Persistence. Automated E2E test. | **Missing** | Partial UI tabs in `app.py`, but workflow was fragmented; missing `model_registry.py` and `history_manager.py`. | No single interactive screen executing the complete sequence; models could not be dynamically selected and executed against individual machine inputs; missing persistent history tracking. |
| **Task 10** | **Validation, Ethics, Fairness & Subgroup Analysis**: Holdout test metrics, input validation stress testing, rule determinism verification, ethical analysis (class imbalance, safety risk, human oversight, privacy), subgroup fairness analysis (Types H, M, L). | **Missing** | None currently in `modules/` or documentation. | No formal validation report; no subgroup fairness error breakdown; no input stress tests for corrupt/out-of-bounds data. |

---

## 3. Implementation Plan

### A. What Can Be Reused
- `modules/planning.py`: State-space predicates, BFS, DFS, A*, admissible heuristic, and path validator (100% reusable with minor bridge addition).
- `modules/regression.py`: Tool wear linear regression pipeline, residual plots, and engineering domain finding ($R^2 \approx 0$).
- Baseline plotting functions in `modules/eda.py` and `modules/classical_ml.py`.
- PyTorch 1D CNN architecture in `modules/cnn_model.py` (`PyTorch1DCNN`).

### B. What Must Be Corrected
1. **Data Leakage in Preprocessing**: Refactor scaling so that `StandardScaler` and encoders are fit strictly on training data (`X_train`) and only transform validation/test/inference inputs.
2. **Evaluation Metrics**: Upgrade evaluation routines to report Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, and specifically **failure-class recall and precision** to address the ~3.39% class imbalance.
3. **Matplotlib Initialization**: Set `os.environ.setdefault("MPLCONFIGDIR", ...)` in `config.py` and test fixtures to eliminate 15-20s sandbox disk latency.
4. **Model Architecture and Persistence**: Train and serialize models (`Random Forest`, `Decision Tree`, `Logistic Regression`, `Gradient Boosting`, `SVM`, and `PyTorch 1D CNN`) to `artifacts/models/` alongside fitted scalers and encoders.

### C. What Must Be Added
1. **`modules/knowledge_representation.py` (Task 1)**:
   - Structured frames and entities: `Machine`, `SensorReading`, `OperatingCondition`, `FailureType`, `PredictedFailureRisk`, `MaintenanceAction`, `MaintenanceRecommendation`.
   - Realistic constraints derived from dataset distributions:
     * Air temperature: [295.0, 305.0] K
     * Process temperature: [305.0, 315.0] K
     * Rotational speed: [1100, 2900] rpm
     * Torque: [3.0, 80.0] Nm
     * Tool wear: [0, 260] min
     * Product type: `['L', 'M', 'H']`
   - Validation engine and programmatic schema reflection.
2. **`modules/decision_layer.py` (Task 2)**:
   - Structured forward-chaining rule engine:
     * Rule identifiers, priorities, IF predicates, THEN actions, and human-readable explanations.
     * Rules for critical failure risk, moderate risk, routine monitoring, heat dissipation strain, mechanical overstrain, tool wear exhaustion, and invalid inputs.
     * Conflict resolution mechanism and detailed execution trace.
     * Explicit separation of statistical ML probability from physical failure certainty.
3. **`modules/model_registry.py` (Task 9)**:
   - Dynamic model registry managing all classical and deep learning models.
   - Enforces correct preprocessing schemas (6 features vs 7 one-hot features).
4. **`modules/history_manager.py` (Task 9)**:
   - Thread-safe, JSON/CSV-backed assessment history logging with search, retrieval, and export.
5. **`modules/interpretability.py` Enhancements (Task 8)**:
   - Interactive What-If simulation engine allowing real-time parameter tweaking and recommendation delta tracking.
   - Activation map extraction directly from the trained PyTorch 1D CNN.
6. **Integration in `app.py`**:
   - Dedicated Knowledge Representation page.
   - Dedicated Rule-Based Inference page.
   - Dedicated Predictive Assessment & End-to-End Workflow page.
   - Dedicated Explainability & What-If Simulator page.
   - Dedicated Model Comparison page.
   - Dedicated Assessment History page.
   - Dedicated Validation, Ethics & Fairness page.
7. **Documentation & Reports**:
   - `docs/IMPLEMENTATION_AUDIT.md` (this file).
   - `docs/VALIDATION_AND_ETHICS.md` (detailed metrics, fairness analysis, stress testing, and ethical guidelines).
   - Root `README.md` and `CIL_Integrated_Project/README.md` updates.
8. **Automated Test Suite**:
   - Add unit and integration tests across knowledge representation, rule engine, end-to-end pipeline, input stress tests, and model registry contracts.
