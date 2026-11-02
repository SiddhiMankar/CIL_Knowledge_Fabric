"""
Model Training and Artifact Generation Script
Trains all Classical ML and PyTorch 1D CNN models on the AI4I 2020 dataset
using strict train/test separation to eliminate data leakage.
Saves serialized model artifacts and holdout test metrics.
"""

import os
import sys
from pathlib import Path
import json

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid Matplotlib sandbox latency
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "mpl"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

import config
from modules import classical_ml, regression, cnn_model


def main():
    print("=" * 70)
    print("MachineGuard: Training and Serializing All Models on AI4I 2020 Dataset")
    print("=" * 70)

    # 1. Locate and load dataset
    found, data_path = config.locate_dataset()
    if not found or data_path is None:
        raise FileNotFoundError(f"Dataset {config.DATASET_FILENAME} not found.")

    print(f"[1/6] Loading dataset from: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"      Loaded shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    target_counts = df_raw["Machine failure"].value_counts().to_dict()
    print(f"      Class distribution: Normal={target_counts.get(0, 0)}, Failure={target_counts.get(1, 0)} "
          f"({target_counts.get(1, 0) / len(df_raw):.2%} failure rate)")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Classical ML Preprocessing (Strict Train/Test Separation)
    print("[2/6] Preparing Classical ML Features & Stratified Split...")
    X_raw, y, le = classical_ml.prepare_classification_data(df_raw)
    
    # Stratified 80/20 split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    # Fit scaler strictly on X_train_raw numerical features
    num_cols = [c for c in config.NUMERICAL_FEATURES if c in X_train_raw.columns]
    scaler_classical = StandardScaler()
    
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()

    X_train_scaled[num_cols] = scaler_classical.fit_transform(X_train_raw[num_cols])
    X_test_scaled[num_cols] = scaler_classical.transform(X_test_raw[num_cols])

    # Save classical preprocessors
    joblib.dump(scaler_classical, config.MODELS_DIR / "scaler_classical.joblib")
    joblib.dump(le, config.MODELS_DIR / "label_encoder.joblib")
    print("      Saved scaler_classical.joblib and label_encoder.joblib")

    # 3. Train and Evaluate Classical Classifiers
    print("[3/6] Training 5 Classical Classifiers on 8,000 samples...")
    models = classical_ml.build_models(random_state=config.RANDOM_STATE)
    classical_results = {}
    classical_saved_bundle = {}

    for name, model in models.items():
        print(f"      Training {name}...", end=" ", flush=True)
        eval_res = classical_ml.evaluate_classifier(
            model, X_train_scaled, y_train, X_test_scaled, y_test
        )
        classical_results[name] = eval_res
        classical_saved_bundle[name] = eval_res["model"]
        print(f"Done. (Acc: {eval_res['accuracy']:.4f}, Failure F1: {eval_res['f1_failure']:.4f}, ROC-AUC: {eval_res['roc_auc']:.4f})")

    # Save classical models bundle
    joblib.dump(classical_saved_bundle, config.MODELS_DIR / "classical_models.joblib")
    print("      Saved classical_models.joblib")

    # 4. Tool Wear Linear Regression
    print("[4/6] Training Tool Wear Linear Regression...")
    X_reg, y_reg = regression.prepare_regression_data(X_train_scaled)
    X_reg_test, y_reg_test = regression.prepare_regression_data(X_test_scaled)

    lin_reg = regression.LinearRegression()
    lin_reg.fit(X_reg, y_reg)
    y_reg_pred = lin_reg.predict(X_reg_test)

    reg_mse = float(regression.mean_squared_error(y_reg_test, y_reg_pred))
    reg_rmse = float(np.sqrt(reg_mse))
    reg_mae = float(regression.mean_absolute_error(y_reg_test, y_reg_pred))
    reg_r2 = float(regression.r2_score(y_reg_test, y_reg_pred))
    print(f"      Regression holdout: MSE={reg_mse:.4f}, RMSE={reg_rmse:.4f}, R2={reg_r2:.4f}")

    # 5. Deep Learning: PyTorch 1D CNN
    print("[5/6] Preparing and Training PyTorch 1D CNN Architecture...")
    prep_cnn = cnn_model.prepare_cnn_dataset(df_raw, test_size=0.2, random_state=config.RANDOM_STATE)
    
    # Save CNN scaler
    joblib.dump(prep_cnn["scaler"], config.MODELS_DIR / "scaler_cnn.joblib")
    print("      Saved scaler_cnn.joblib")

    cnn_test_metrics = {}
    if cnn_model.HAS_TORCH:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(config.RANDOM_STATE)
        np.random.seed(config.RANDOM_STATE)

        in_feat = prep_cnn["num_features"]
        model_cnn = cnn_model.PyTorch1DCNN(in_features=in_feat)

        # Imbalanced class weighting
        pos_weight = torch.tensor([prep_cnn["class_weights"][1] / prep_cnn["class_weights"][0]], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Note: PyTorch1DCNN has a sigmoid output layer, so for BCEWithLogitsLoss we train without sigmoid or use BCELoss with weights
        criterion = nn.BCELoss(weight=None)
        optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

        X_train_t = torch.tensor(prep_cnn["X_train_cnn"], dtype=torch.float32)
        y_train_t = torch.tensor(prep_cnn["y_train"].values, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(prep_cnn["X_test_cnn"], dtype=torch.float32)
        y_test_t = prep_cnn["y_test"].values

        # Weights per sample for BCE
        sample_weights = torch.tensor(
            [prep_cnn["class_weights"][int(lbl)] for lbl in prep_cnn["y_train"].values],
            dtype=torch.float32
        ).unsqueeze(1)

        print("      Training 1D CNN (25 epochs with batch size 64)...")
        batch_size = 64
        num_batches = int(np.ceil(len(X_train_t) / batch_size))

        best_loss = float("inf")
        best_state = None

        model_cnn.train()
        for epoch in range(25):
            permutation = torch.randperm(X_train_t.size(0))
            epoch_loss = 0.0

            for b in range(num_batches):
                indices = permutation[b * batch_size : (b + 1) * batch_size]
                batch_x = X_train_t[indices]
                batch_y = y_train_t[indices]
                batch_w = sample_weights[indices]

                optimizer.zero_grad()
                out = model_cnn(batch_x)
                # Weighted BCELoss
                loss = nn.functional.binary_cross_entropy(out, batch_y, weight=batch_w)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Evaluate CNN on holdout test set
        model_cnn.eval()
        with torch.no_grad():
            cnn_probs = model_cnn(X_test_t).squeeze().cpu().numpy()
            cnn_preds = (cnn_probs >= 0.5).astype(int)

        cnn_acc = float(accuracy_score(y_test_t, cnn_preds))
        cnn_prec = float(precision_score(y_test_t, cnn_preds, pos_label=1, zero_division=0))
        cnn_rec = float(recall_score(y_test_t, cnn_preds, pos_label=1, zero_division=0))
        cnn_f1 = float(f1_score(y_test_t, cnn_preds, pos_label=1, zero_division=0))
        cnn_roc = float(roc_auc_score(y_test_t, cnn_probs))
        cnn_pr = float(average_precision_score(y_test_t, cnn_probs))
        cnn_cm = confusion_matrix(y_test_t, cnn_preds)

        cnn_test_metrics = {
            "accuracy": round(cnn_acc, 4),
            "precision_failure": round(cnn_prec, 4),
            "recall_failure": round(cnn_rec, 4),
            "f1_score": round(cnn_f1, 4),
            "roc_auc": round(cnn_roc, 4),
            "pr_auc": round(cnn_pr, 4),
            "confusion_matrix": cnn_cm.tolist(),
            "false_positives": int(cnn_cm[0, 1]),
            "false_negatives": int(cnn_cm[1, 0])
        }
        print(f"      CNN Holdout Evaluation: Accuracy={cnn_acc:.4f}, Failure Recall={cnn_rec:.4f}, Failure Precision={cnn_prec:.4f}, ROC-AUC={cnn_roc:.4f}")

        # Save PyTorch CNN weights
        torch.save(model_cnn.state_dict(), config.MODELS_DIR / "cnn_pytorch.pt")
        print("      Saved cnn_pytorch.pt")

    # 6. Save comprehensive metadata.json
    print("[6/6] Updating metadata.json with actual measured benchmarks...")
    
    classical_meta = {}
    for name, r in classical_results.items():
        classical_meta[name] = {
            "accuracy": round(r["accuracy"], 4),
            "precision_failure": round(r["precision_failure"], 4),
            "recall_failure": round(r["recall_failure"], 4),
            "f1_score": round(r["f1_failure"], 4),
            "roc_auc": round(r["roc_auc"], 4),
            "pr_auc": round(r["pr_auc"], 4),
            "false_negatives": r["false_negatives"],
            "false_positives": r["false_positives"],
            "confusion_matrix": r["confusion_matrix"].tolist()
        }

    metadata_payload = {
        "project": "CIL Knowledge Fabric — MachineGuard Platform",
        "dataset_spec": {
            "filename": "ai4i2020.csv",
            "total_samples": len(df_raw),
            "features_count": 14,
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
            "failure_samples_test": int(y_test.sum()),
            "normal_samples_test": int((y_test == 0).sum())
        },
        "task5_classical_ml_benchmarks": {
            "test_split_ratio": 0.2,
            "models": classical_meta
        },
        "task6_regression_benchmarks": {
            "model": "Linear Regression",
            "target": "Tool wear [min]",
            "mse": round(reg_mse, 4),
            "rmse": round(reg_rmse, 4),
            "mae": round(reg_mae, 4),
            "r2_score": round(reg_r2, 4),
            "coefficients": {k: round(float(v), 4) for k, v in zip(X_reg.columns, lin_reg.coef_)},
            "intercept": round(float(lin_reg.intercept_), 4),
            "observation": "R2 near zero confirms instantaneous telemetry alone does not linearly correlate with cumulative wear."
        },
        "task7_cnn_benchmarks": {
            "architecture": "1D CNN (Conv1D 32 -> BatchNorm -> MaxPool -> Conv1D 64 -> BatchNorm -> GlobalAvgPool -> Dense 64 -> Dropout 0.5 -> Dense 1)",
            "cnn_metrics": cnn_test_metrics,
            "baseline_logistic_regression_metrics": classical_meta.get("Logistic Regression", {})
        }
    }

    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    print(f"      Saved benchmarks to: {config.METADATA_PATH}")

    print("=" * 70)
    print("All Models Successfully Trained, Evaluated, and Serialized!")
    print("=" * 70)


if __name__ == "__main__":
    main()

