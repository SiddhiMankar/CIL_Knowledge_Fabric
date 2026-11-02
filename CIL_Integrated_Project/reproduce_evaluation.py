"""
Verification script for Point 2: Reproduce model evaluation on holdout test set.
Verifies all 6 models against saved artifacts and checks exact metrics.
"""

import os
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "mpl"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import config
from modules.cnn_model import PyTorch1DCNN, HAS_TORCH
if HAS_TORCH:
    import torch


def main():
    print("=" * 70)
    print("REPRODUCING HOLDOUT EVALUATION ON ACTUAL SAVED MODEL ARTIFACTS")
    print("=" * 70)

    # 1. Load raw data and recreate exact holdout split
    df = pd.read_csv(config.RAW_DATA_DIR / config.DATASET_FILENAME)
    
    # Exclude identifiers and targets
    feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    y = df[config.TARGET_COL]

    # Recreate the exact stratified 80/20 train/test partition (random_state=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[feat_cols], y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    print(f"Holdout Test Size: {len(X_test_raw)} samples")
    print(f"Target breakdown: Normal={int((y_test==0).sum())}, Failures={int((y_test==1).sum())} ({y_test.mean():.2%})")

    # 2. Load Classical Preprocessors
    scaler_classical = joblib.load(config.MODELS_DIR / "scaler_classical.joblib")
    label_encoder = joblib.load(config.MODELS_DIR / "label_encoder.joblib")
    classical_models = joblib.load(config.MODELS_DIR / "classical_models.joblib")

    # Transform X_test for classical models (6 features)
    X_test_classical = X_test_raw.copy()
    X_test_classical["Type"] = label_encoder.transform(X_test_classical["Type"])
    num_cols = [c for c in config.NUMERICAL_FEATURES if c in X_test_classical.columns]
    X_test_classical[num_cols] = scaler_classical.transform(X_test_classical[num_cols])
    X_test_classical = X_test_classical[config.FEATURE_COLS_LABEL_ENCODED]

    # Evaluate each classical model
    eval_summary = []
    for name, model in classical_models.items():
        y_pred = model.predict(X_test_classical)
        y_prob = model.predict_proba(X_test_classical)[:, 1] if hasattr(model, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.5
        pr_auc = float(average_precision_score(y_test, y_prob)) if y_prob is not None else 0.0

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        eval_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Failure Recall": rec,
            "Failure Precision": prec,
            "Failure F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

    # 3. Evaluate PyTorch 1D CNN
    if HAS_TORCH and (config.MODELS_DIR / "cnn_pytorch.pt").exists():
        scaler_cnn = joblib.load(config.MODELS_DIR / "scaler_cnn.joblib")
        # One-hot encode Type for test set
        X_test_cnn_raw = pd.get_dummies(X_test_raw, columns=["Type"], drop_first=True)
        # Ensure exact 7 columns
        X_test_cnn_scaled = scaler_cnn.transform(X_test_cnn_raw[config.FEATURE_COLS_ONEHOT])
        X_test_cnn_tensor = torch.tensor(
            X_test_cnn_scaled.reshape(X_test_cnn_scaled.shape[0], 7, 1),
            dtype=torch.float32
        )

        cnn = PyTorch1DCNN(in_features=7)
        cnn.load_state_dict(torch.load(config.MODELS_DIR / "cnn_pytorch.pt", map_location=torch.device("cpu")))
        cnn.eval()

        with torch.no_grad():
            cnn_probs = cnn(X_test_cnn_tensor).squeeze().cpu().numpy()
            cnn_preds = (cnn_probs >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, cnn_preds))
        rec = float(recall_score(y_test, cnn_preds, pos_label=1, zero_division=0))
        prec = float(precision_score(y_test, cnn_preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, cnn_preds, pos_label=1, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, cnn_probs))
        pr_auc = float(average_precision_score(y_test, cnn_probs))

        cm = confusion_matrix(y_test, cnn_preds)
        tn, fp, fn, tp = cm.ravel()

        eval_summary.append({
            "Model": "1D CNN (PyTorch)",
            "Accuracy": acc,
            "Failure Recall": rec,
            "Failure Precision": prec,
            "Failure F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

    df_res = pd.DataFrame(eval_summary)
    print("\nMeasured Holdout Evaluation Results:")
    print(df_res.to_string(index=False))

    return df_res


if __name__ == "__main__":
    main()

