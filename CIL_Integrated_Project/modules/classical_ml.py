"""
Task 5 & 6 Module: Classical Machine Learning, Feature Engineering, & Clustering
Original source: CIL456.ipynb

Implements:
- Feature engineering: Domain-derived physical features (Temperature difference, Mechanical power, Overstrain).
- Feature selection and categorical encoding (LabelEncoder / OneHot).
- Random Forest feature importance analysis.
- Supervised classification with 5 models:
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (RBF kernel, calibrated probabilities)
  * Gradient Boosting Classifier
- Comprehensive holdout evaluation:
  * Accuracy, Precision, Recall, F1-Score
  * ROC-AUC and Precision-Recall AUC
  * Failure-class (Class 1) specific Recall and Precision
  * Confusion matrices and False Positive / False Negative counts
- Misclassification analysis:
  * Feature profiling of True Positives, True Negatives, False Positives, False Negatives
- Unsupervised learning:
  * K-Means clustering (3 clusters) on Torque vs Rotational Speed
  * Hierarchical agglomerative clustering dendrogram (Ward's method)
- Model persistence and loading via joblib.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; required for headless/test environments
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import config

# Set Matplotlib cache dir to Temp if not set
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "mpl"))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-derived physical features:
    1. Temperature difference [K] = Process temperature - Air temperature
    2. Mechanical power [W] = (2 * pi / 60) * Speed * Torque
    3. Tool wear torque strain = Torque * Tool wear
    """
    df_feat = df.copy()
    if "Process temperature [K]" in df_feat.columns and "Air temperature [K]" in df_feat.columns:
        df_feat["Temperature difference [K]"] = (
            df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
        )

    if "Rotational speed [rpm]" in df_feat.columns and "Torque [Nm]" in df_feat.columns:
        df_feat["Mechanical power [W]"] = (
            (2.0 * np.pi / 60.0) * df_feat["Rotational speed [rpm]"] * df_feat["Torque [Nm]"]
        )

    if "Torque [Nm]" in df_feat.columns and "Tool wear [min]" in df_feat.columns:
        df_feat["Tool wear torque strain"] = (
            df_feat["Torque [Nm]"] * df_feat["Tool wear [min]"]
        )

    return df_feat


def prepare_classification_data(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Extracts features and target, applying LabelEncoder to 'Type' categorical feature.
    Direct reproduction of CIL456.ipynb Cells 13-14.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in dataset.")

    default_features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    cols = feature_cols or [c for c in default_features if c in df.columns]

    X = df[cols].copy()
    y = df[target_col].copy()

    le = LabelEncoder()
    if "Type" in X.columns:
        X["Type"] = le.fit_transform(X["Type"].astype(str))

    return X, y, le


def compute_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Computes feature importances using RandomForestClassifier and generates horizontal bar plot.
    Direct reproduction of CIL456.ipynb Cell 15.
    """
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_names = X.columns.tolist()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2B5B84", edgecolor="black", alpha=0.85)
    ax.set_title("Random Forest Feature Importance for Machine Failure", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()

    return importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True), fig


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into train and holdout test sets.
    Uses stratification by default to preserve the ~3.39% failure class ratio.
    """
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def build_models(random_state: int = config.RANDOM_STATE) -> Dict[str, Any]:
    """
    Initializes 5 supervised classifiers covering classical, linear, tree, and ensemble paradigms.
    """
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=5),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000)
    }


def evaluate_classifier(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Fits and evaluates a single classifier with comprehensive metrics:
    Accuracy, Failure-class Recall, Failure-class Precision, Failure-class F1,
    ROC-AUC, PR-AUC, and Confusion Matrix breakdown.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC and PR-AUC
    y_prob = None
    roc_auc = 0.5
    pr_auc = 0.0
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
            pr_auc = float(average_precision_score(y_test, y_prob))
        except Exception:
            pass

    acc = float(accuracy_score(y_test, y_pred))
    prec_fail = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_fail = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_fail = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "precision_failure": prec_fail,
        "recall_failure": rec_fail,
        "f1_failure": f1_fail,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": report,
        "predictions": y_pred,
        "probabilities": y_prob,
        "model": model
    }


def train_and_evaluate_all_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Trains and evaluates all 5 classifiers, returning a comprehensive
    comparison table sorted by Failure-Class Recall & F1, plus full results.
    """
    models = build_models(random_state=random_state)
    results = {}
    table_rows = []

    for name, model in models.items():
        eval_res = evaluate_classifier(model, X_train, y_train, X_test, y_test)
        results[name] = eval_res
        table_rows.append({
            "Model": name,
            "Accuracy": eval_res["accuracy"],
            "Failure Recall": eval_res["recall_failure"],
            "Failure Precision": eval_res["precision_failure"],
            "Failure F1": eval_res["f1_failure"],
            "ROC-AUC": eval_res["roc_auc"],
            "PR-AUC": eval_res["pr_auc"],
            "False Negatives (Missed)": eval_res["false_negatives"],
            "False Positives (False Alarm)": eval_res["false_positives"]
        })

    comparison_df = pd.DataFrame(table_rows).sort_values(
        by=["Failure F1", "ROC-AUC"], ascending=False
    ).reset_index(drop=True)
    return comparison_df, results


def get_misclassification_profiles(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, pd.DataFrame]:
    """
    Extracts representative sample profiles for True Positives, True Negatives,
    False Positives, and False Negatives to explain error characteristics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0.0] * len(y_test)

    df_eval = X_test.copy()
    df_eval["Actual"] = y_test.values
    df_eval["Predicted"] = y_pred
    df_eval["Failure_Probability"] = np.round(y_prob, 4)

    tp_df = df_eval[(df_eval["Actual"] == 1) & (df_eval["Predicted"] == 1)]
    tn_df = df_eval[(df_eval["Actual"] == 0) & (df_eval["Predicted"] == 0)]
    fp_df = df_eval[(df_eval["Actual"] == 0) & (df_eval["Predicted"] == 1)]
    fn_df = df_eval[(df_eval["Actual"] == 1) & (df_eval["Predicted"] == 0)]

    return {
        "True Positives (Correct Failure)": tp_df,
        "True Negatives (Correct Normal)": tn_df,
        "False Positives (False Alarm)": fp_df,
        "False Negatives (Missed Failure)": fn_df
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str, cmap: str = "Blues") -> plt.Figure:
    """Generates confusion matrix heatmap plot for a trained classifier."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                annot_kws={"size": 11, "fontweight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("Actual Label", fontsize=9)
    ax.set_title(f"CM — {model_name}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def run_kmeans_clustering(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, KMeans, plt.Figure]:
    """
    Performs K-Means clustering and generates scatter plot of Torque vs Rotational Speed.
    Direct reproduction of CIL456.ipynb Cell 22.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x_col = "Rotational speed [rpm]" if "Rotational speed [rpm]" in X.columns else X.columns[0]
    y_col = "Torque [Nm]" if "Torque [Nm]" in X.columns else X.columns[1]

    scatter = ax.scatter(X[x_col], X[y_col], c=clusters, cmap="viridis", alpha=0.5, edgecolors="none")
    ax.set_title("K-Means Clustering: Torque vs Rotational Speed (k=3)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{x_col}", fontsize=9)
    ax.set_ylabel(f"{y_col}", fontsize=9)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster ID", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    return clusters, kmeans, fig


def run_hierarchical_clustering(
    X: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Performs Hierarchical Clustering using Ward's method on a subset of samples
    and plots the dendrogram.
    Direct reproduction of CIL456.ipynb Cell 23.
    """
    n = min(len(X), sample_size)
    X_sample = X.sample(n, random_state=random_state) if len(X) > n else X.copy()
    Z = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward's Linkage, {n} samples)", fontsize=11, fontweight="bold")
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax
    )
    ax.set_xlabel("Cluster Size", fontsize=9)
    ax.set_ylabel("Distance", fontsize=9)
    fig.tight_layout()

    return Z, fig


def save_trained_models(models_dict: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Saves trained classical models dictionary to disk."""
    save_path = path or config.CLASSICAL_MODELS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models_dict, save_path)
    return save_path


def load_trained_models(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Loads saved classical models dictionary from disk if available."""
    target_path = path or config.CLASSICAL_MODELS_PATH
    if target_path.exists():
        return joblib.load(target_path)
    return None
