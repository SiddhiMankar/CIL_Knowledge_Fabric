"""
Task 7 Module: 1D Convolutional Neural Network (CNN) for Predictive Maintenance
Original source: CIL78 (1).ipynb

Implements:
- Feature preprocessing for 1D CNN: One-hot encoding, stratified train-test split,
  leakage-free scaling, and 3D reshaping (samples, features, 1).
- Class weight computation for imbalanced failure targets.
- 1D CNN architecture specification (Conv1D -> BatchNorm -> MaxPool -> Conv1D -> BatchNorm -> GlobalAvgPool -> Dense -> Dropout -> Sigmoid).
- Multi-backend architecture support:
  * Keras / TensorFlow when available in supported environments (Python <= 3.12).
  * PyTorch companion backend for Windows Python 3.14 environments.
- Comprehensive evaluation: Classification Report, Confusion Matrix, ROC-AUC, and PR-AUC curves.
- Comparison against baseline Logistic Regression model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; required for headless/test environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

import config

# Check framework availability
HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


def prepare_cnn_dataset(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Dict[str, Any]:
    """
    Prepares dataset for 1D CNN modeling:
    1. Excludes identifiers and failure mode indicators to prevent data leakage.
    2. Encodes 'Type' using pd.get_dummies(..., drop_first=True) creating 7 features.
    3. Performs stratified train-test split.
    4. Fits StandardScaler strictly on training set.
    5. Reshapes to (samples, features, 1) for Conv1D.
    Direct reproduction of CIL78 (1).ipynb Cells 5-8.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in dataframe.")

    y = df[target_col].copy()

    # Exclude identifiers and direct failure mode indicator columns
    feature_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    X_raw = df[feature_cols].copy()

    # One-hot encode categorical 'Type'
    if "Type" in X_raw.columns:
        X_encoded = pd.get_dummies(X_raw, columns=["Type"], drop_first=True)
    else:
        X_encoded = X_raw.copy()

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Leakage-free scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for Conv1D: (samples, timesteps/features, 1)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # Calculate class weights for imbalanced data
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_train), weights)}

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_cnn": X_train_cnn,
        "X_test_cnn": X_test_cnn,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
        "num_features": X_train.shape[1],
        "class_weights": class_weights_dict
    }


# ============================================================================
# PyTorch Equivalent Model Architecture (Native for Windows Python 3.14)
# ============================================================================
if HAS_TORCH:
    class PyTorch1DCNN(nn.Module):
        """
        PyTorch equivalent of the 1D CNN architecture from CIL78 (1).ipynb.
        Input shape: (Batch, Channels=1, Length=7)
        """
        def __init__(self, in_features: int = 7):
            super().__init__()
            # Conv1: filters=32, kernel_size=3
            # Length: 7 -> 5
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
            self.bn1 = nn.BatchNorm1d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=2)  # Length: 5 -> 2

            # Conv2: filters=64, kernel_size=2
            # Length: 2 -> 1
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.relu2 = nn.ReLU()
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # Length: 1

            self.dense1 = nn.Linear(64, 64)
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.out = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x input shape from (batch, 7, 1) -> transpose to (batch, 1, 7)
            if x.dim() == 3 and x.shape[-1] == 1:
                x = x.transpose(1, 2)
            h = self.conv1(x)
            h = self.bn1(h)
            h = self.relu1(h)
            h = self.pool1(h)

            h = self.conv2(h)
            h = self.bn2(h)
            h = self.relu2(h)
            h = self.global_pool(h)

            h = torch.flatten(h, 1)
            h = self.dense1(h)
            h = self.relu3(h)
            h = self.dropout(h)
            h = self.out(h)
            return self.sigmoid(h)

        def extract_conv_activations(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            """Extracts intermediate layer feature maps for interpretability."""
            self.eval()
            with torch.no_grad():
                if x.dim() == 3 and x.shape[-1] == 1:
                    x = x.transpose(1, 2)
                c1 = self.relu1(self.bn1(self.conv1(x)))
                p1 = self.pool1(c1)
                c2 = self.relu2(self.bn2(self.conv2(p1)))
                # Convert back to (samples, length, filters) for consistency
                c1_np = c1.transpose(1, 2).cpu().numpy()
                c2_np = c2.transpose(1, 2).cpu().numpy()
                return c1_np, c2_np


def build_keras_cnn_model(num_features: int = 7) -> Any:
    """
    Builds the exact Keras Sequential 1D CNN specified in CIL78 (1).ipynb Cell 9.
    Returns Keras Model if TensorFlow is installed, raises RuntimeError otherwise.
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed in the current environment.")

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_features, 1), name='conv1d_1'),
        BatchNormalization(name='batch_norm_1'),
        MaxPooling1D(pool_size=2, name='max_pooling_1'),
        Conv1D(filters=64, kernel_size=2, activation='relu', name='conv1d_2'),
        BatchNormalization(name='batch_norm_2'),
        GlobalAveragePooling1D(name='global_avg_pooling'),
        Dense(64, activation='relu', name='dense_1'),
        Dropout(0.5, name='dropout_1'),
        Dense(1, activation='sigmoid', name='output_layer')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculates full test set evaluation metrics including ROC-AUC and PR-AUC.
    Direct reproduction of CIL78 (1).ipynb Cells 12 and 13.
    """
    y_true_arr = np.array(y_true).ravel()
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true_arr, y_pred)
    fpr, tpr, _ = roc_curve(y_true_arr, y_pred_proba)
    roc_auc_val = float(auc(fpr, tpr))

    precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_pred_proba)
    pr_auc_val = float(average_precision_score(y_true_arr, y_pred_proba))

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "pr_curve": (recall_curve, precision_curve),
        "classification_report": classification_report(y_true_arr, y_pred, output_dict=True),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }


def train_baseline_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: Union[pd.Series, np.ndarray],
    X_test_scaled: np.ndarray,
    y_test: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Builds and evaluates the baseline Logistic Regression model from CIL78 (1).ipynb Cell 13.
    Uses class_weight='balanced' and solver='liblinear'.
    """
    basic_model = LogisticRegression(
        random_state=config.RANDOM_STATE,
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000
    )
    basic_model.fit(X_train_scaled, y_train)

    y_pred_proba = basic_model.predict_proba(X_test_scaled)[:, 1]
    eval_res = evaluate_predictions(y_test, y_pred_proba)
    eval_res["model"] = basic_model
    return eval_res


def build_comparison_table(basic_metrics: Dict[str, Any], cnn_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Constructs model performance comparison table.
    Direct reproduction of CIL78 (1).ipynb Cell 14.
    """
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]

    data = {
        "Metric": display_names,
        "Basic ML Model (Logistic Regression)": [round(basic_metrics.get(k, 0.0), 3) for k in keys],
        "1D CNN Model": [round(cnn_metrics.get(k, 0.0), 3) for k in keys]
    }
    return pd.DataFrame(data).set_index("Metric")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc_val: float, model_name: str = "CNN") -> plt.Figure:
    """Generates Receiver Operating Characteristic (ROC) curve plot."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_pr_curve(recall_vals: np.ndarray, precision_vals: np.ndarray, pr_auc_val: float, model_name: str = "CNN") -> plt.Figure:
    """Generates Precision-Recall curve plot."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall_vals, precision_vals, color="#1E88E5", lw=2, label=f"PR curve (AUC = {pr_auc_val:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_training_history(history_df: pd.DataFrame) -> plt.Figure:
    """Plots training vs. validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss plot
    if "loss" in history_df.columns:
        ax1.plot(history_df["loss"], label="Training Loss", color="#1F77B4", lw=2)
    if "val_loss" in history_df.columns:
        ax1.plot(history_df["val_loss"], label="Validation Loss", color="#FF7F0E", lw=2)
    ax1.set_title("Training and Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy plot
    if "accuracy" in history_df.columns:
        ax2.plot(history_df["accuracy"], label="Training Accuracy", color="#2CA02C", lw=2)
    if "val_accuracy" in history_df.columns:
        ax2.plot(history_df["val_accuracy"], label="Validation Accuracy", color="#D62728", lw=2)
    ax2.set_title("Training and Validation Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
