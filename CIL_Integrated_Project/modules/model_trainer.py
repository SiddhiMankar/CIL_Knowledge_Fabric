"""
MachineGuard Training & Artifact Generation Pipeline
Trains leakage-free models on data/raw/ai4i2020.csv, evaluates on held-out test split,
and persists production artifacts to artifacts/models/ and artifacts/metadata.json.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

import config

def train_and_persist_pipeline(
    data_path: Path = None,
    random_state: int = config.RANDOM_STATE,
    test_size: float = config.TEST_SIZE
):
    if data_path is None:
        found, p = config.locate_dataset()
        if not found or p is None:
            raise FileNotFoundError("ai4i2020.csv not found in data directories.")
        data_path = p

    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Verification of Target and Feature Columns (LEAKAGE FREE)
    # Exclude identifiers and failure mode flags (TWF, HDF, PWF, OSF, RNF)
    feature_cols = config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES
    target_col = config.TARGET_COL

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train samples: {len(X_train)} (Failures: {y_train.sum()})")
    print(f"Test samples:  {len(X_test)} (Failures: {y_test.sum()})")

    # 3. Build ColumnTransformer Preprocessor (fitted strictly on X_train)
    num_cols = config.NUMERICAL_FEATURES
    cat_cols = config.CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ])

    # 4. Fit Preprocessor on Train Set
    preprocessor.fit(X_train)
    
    # Get transformed feature names
    cat_ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = [f"Type_{cat}" for cat in cat_ohe.categories_[0][1:]]
    transformed_feature_names = num_cols + cat_feature_names

    # 5. Define candidate models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        )
    }

    eval_results = {}
    fitted_pipelines = {}

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train_proc, y_train)

        y_pred = clf.predict(X_test_proc)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_proc)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        eval_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        fitted_pipelines[name] = clf

    # 6. Select Production Model: Random Forest
    prod_model_name = "Random Forest"
    prod_clf = fitted_pipelines[prod_model_name]

    # Evaluate production model under custom operating thresholds (e.g. 0.30 vs 0.50)
    rf_proba_test = prod_clf.predict_proba(X_test_proc)[:, 1]
    threshold_eval = {}
    for th in [0.20, 0.30, 0.35, 0.40, 0.50]:
        th_pred = (rf_proba_test >= th).astype(int)
        th_cm = confusion_matrix(y_test, th_pred)
        th_tn, th_fp, th_fn, th_tp = th_cm.ravel()
        th_prec = th_tp / (th_tp + th_fp) if (th_tp + th_fp) > 0 else 0
        th_rec = th_tp / (th_tp + th_fn) if (th_tp + th_fn) > 0 else 0
        th_f1 = 2 * th_prec * th_rec / (th_prec + th_rec) if (th_prec + th_rec) > 0 else 0
        threshold_eval[str(th)] = {
            "tp": int(th_tp),
            "fp": int(th_fp),
            "fn": int(th_fn),
            "tn": int(th_tn),
            "precision": round(float(th_prec), 4),
            "recall": round(float(th_rec), 4),
            "f1_score": round(float(th_f1), 4)
        }

    # 7. Subgroup Evaluation across Machine Types (H, L, M)
    subgroup_eval = {}
    for machine_type in ["L", "M", "H"]:
        mask = (X_test["Type"] == machine_type).values
        sub_y_test = y_test[mask]
        sub_proba = rf_proba_test[mask]
        sub_pred = (sub_proba >= 0.35).astype(int)
        sub_cm = confusion_matrix(sub_y_test, sub_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = sub_cm.ravel()
        subgroup_eval[machine_type] = {
            "sample_count": int(mask.sum()),
            "failure_count": int(sub_y_test.sum()),
            "failure_rate": round(float(sub_y_test.mean()), 4),
            "tp": int(s_tp),
            "fp": int(s_fp),
            "fn": int(s_fn),
            "tn": int(s_tn),
            "precision": round(float(s_tp / (s_tp + s_fp)) if (s_tp + s_fp) > 0 else 0.0, 4),
            "recall": round(float(s_tp / (s_tp + s_fn)) if (s_tp + s_fn) > 0 else 0.0, 4),
            "f1_score": round(float(2 * s_tp / (2 * s_tp + s_fp + s_fn)) if (2 * s_tp + s_fp + s_fn) > 0 else 0.0, 4)
        }

    # 8. Feature Importances (MDI & Permutation)
    global_importances = dict(zip(transformed_feature_names, [round(float(v), 4) for v in prod_clf.feature_importances_]))

    # 9. Persist Model and Preprocessor to artifacts/models/
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prod_artifact = {
        "model": prod_clf,
        "preprocessor": preprocessor,
        "feature_names_raw": feature_cols,
        "feature_names_transformed": transformed_feature_names,
        "target_col": target_col,
        "operating_threshold": 0.35,  # Balanced operating point for maintenance safety
        "model_version": "1.0.0",
        "random_state": random_state
    }
    prod_model_path = config.MODELS_DIR / "production_model.joblib"
    joblib.dump(prod_artifact, prod_model_path)
    print(f"Saved production model artifact to {prod_model_path}")

    # Also persist test set for live benchmark checks and demonstration records
    test_data_artifact = {
        "X_test": X_test,
        "y_test": y_test,
        "X_test_proc": X_test_proc
    }
    joblib.dump(test_data_artifact, config.MODELS_DIR / "test_set_split.joblib")

    # 10. Update metadata.json
    metadata = {}
    if config.METADATA_PATH.exists():
        try:
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    metadata["machineguard_production"] = {
        "primary_model": prod_model_name,
        "model_version": "1.0.0",
        "feature_schema": {
            "raw_inputs": feature_cols,
            "categorical": cat_cols,
            "numerical": num_cols,
            "transformed_features": transformed_feature_names
        },
        "operating_threshold": 0.35,
        "held_out_metrics": eval_results[prod_model_name],
        "metrics_at_operating_threshold": threshold_eval.get("0.35", {}),
        "all_model_evaluations": eval_results,
        "threshold_tradeoffs": threshold_eval,
        "subgroup_performance": subgroup_eval,
        "feature_importances": global_importances,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_failures": int(y_train.sum()),
        "test_failures": int(y_test.sum())
    }

    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Updated metadata.json with verified evaluation metrics!")
    return prod_artifact, metadata

if __name__ == "__main__":
    train_and_persist_pipeline()
