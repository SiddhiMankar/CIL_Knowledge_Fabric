# KnowledgeFabric / MachineGuard — Predictive Maintenance Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.66.0-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-42%20passed-brightgreen.svg)](tests/)

**KnowledgeFabric / MachineGuard** is an industrial decision-support prototype built by transforming and unifying laboratory tasks 3 through 8 of the CIL Integrated Project into an auditable, explainable predictive maintenance platform.

It directly answers four essential operational questions:
1. **What is the predicted failure risk** for the supplied machine readings?
2. **Which features influenced the model\'s prediction** (exact tree-path local attributions)?
3. **What action should the user consider**, and what physical evidence supports it?
4. **How reliable is the system**, and what are its operational boundaries and limitations?

---

## 1. Core Architecture and Main Workflow

* **Machine Assessment (Primary Workflow):**
  - Validated Telemetry Form: Real-time sensor inputs with physical range validation.
  - Demonstration Profiles: Verified operating points (Nominal, Critical Wear, Overstrain, Heat Dissipation).
  - Batch Upload: Upload any operational CSV for bulk assessment and export results.
  - Instant Inference: Evaluates the persisted Random Forest model with zero retraining during runtime.
  - Real Local Feature Attributions: Exact tree-path probability deltas.
  - What-If Counterfactual Simulator: Interactive sensor sensitivity curves.
* **Assessment History:**
  - Auditable in-session and persistent logs (artifacts/assessment_history.json).
  - One-click CSV audit trail export.
* **Model Performance and Leakage-Free Validation:**
  - Evaluated strictly on an untouched held-out test split (2,000 samples, 20% stratified split).
  - Reports Precision, Recall, F1, ROC-AUC, and PR-AUC.
  - Full confusion matrix and operating threshold sensitivity analysis (0.20 to 0.50).
  - Subgroup reliability checks across machine quality variants (L, M, H).
* **Experiments (Tasks 3-8):**
  - Task 3: State Space Search AI Planner (BFS, DFS, A* for Pump P-101).
  - Task 4: EDA, outlier boxplots, and correlation heatmaps.
  - Task 5: Classical classifiers (RF, DT, SVM, LogReg) and K-Means clustering.
  - Task 6: Tool Wear linear regression residual diagnostics.
  - Task 7: 1D CNN deep learning architecture benchmark.
  - Task 8: Conv1D activation feature maps and LIME explainability.
* **About, Ethics and Limitations:**
  - Class imbalance consequences (3.39% failure rate).
  - Target leakage prevention (exclusion of TWF, HDF, PWF, OSF, RNF).
  - Synthetic data provenance and domain shift caveats.
  - Human-in-the-loop safety oversight.

---

## 2. Quick Start (Windows PowerShell)

```powershell
cd .\CIL_Integrated_Project\
pytest                        # Expected: 42 passed, 0 failed
streamlit run app.py          # Open http://localhost:8501
```

To retrain from scratch:
```powershell
python -m modules.model_trainer
```

---

## 3. Model Performance Summary (Held-Out Test Set, 2,000 samples)

| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Production, th=0.35)** | **0.9696** | **0.7305** | **0.6753** | **0.6047** | **0.7647** | Primary Engine |
| Gradient Boosting | 0.9672 | 0.8080 | 0.7934 | 0.9057 | 0.7059 | Benchmark |
| Logistic Regression (Balanced) | 0.9069 | 0.3816 | 0.2414 | 0.1414 | 0.8235 | Linear Baseline |
| 1D CNN | 0.9800 | 0.5800 | 0.5640 | 0.5000 | 0.6470 | Experimental DL |

Production RF metrics are reported at the **operating threshold of 0.35** (not 0.50). At this threshold, recall is 76.5% vs. 63.2% at 0.50, reducing missed failures at acceptable false-positive cost (~$50 inspection vs. ~$2,500+ breakdown).

---

## 4. Environment Notes

- **Matplotlib backend:** All modules set matplotlib.use(\'Agg\') before pyplot is imported, ensuring compatibility with headless/test environments where Tk/tkinter is unavailable.
- **PyTorch (Tasks 7/8):** Imported with (ImportError, OSError) handling for graceful startup on systems without VC++ Redistributable.
