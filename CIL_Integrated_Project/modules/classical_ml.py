"""
Task 5 Module: Classical Machine Learning & Clustering
Original source: CIL456.ipynb

Implements:
- Feature selection and categorical encoding (LabelEncoder / OneHot).
- Random Forest feature importance analysis.
- Supervised classification: Logistic Regression, Decision Tree, Random Forest, SVM.
- Performance evaluation: Accuracy, Classification Reports, Confusion Matrices.
- Unsupervised learning: K-Means clustering and Hierarchical agglomerative dendrograms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import config


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

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2B5B84", edgecolor="black", alpha=0.85)
    ax.set_title("Feature Importance for Machine Failure", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()

    return importance_df.sort_values(by="Importance", ascending=False), fig


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits features and target into train and test sets."""
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def build_models(random_state: int = config.RANDOM_STATE) -> Dict[str, Any]:
    """Initializes the 4 supervised classifiers specified in CIL456.ipynb."""
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": SVC(kernel="rbf", random_state=random_state)
    }


def evaluate_classifier(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Fits and evaluates a single classifier on train and test sets."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
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
    Trains and evaluates all 4 classifiers, returning a sorted comparison table
    and full evaluation details.
    """
    models = build_models(random_state=random_state)
    results = {}
    table_rows = []

    for name, model in models.items():
        eval_res = evaluate_classifier(model, X_train, y_train, X_test, y_test)
        results[name] = eval_res
        table_rows.append({
            "Model": name,
            "Accuracy Score": eval_res["accuracy"]
        })

    comparison_df = pd.DataFrame(table_rows).sort_values(by="Accuracy Score", ascending=False).reset_index(drop=True)
    return comparison_df, results


def plot_confusion_matrix(cm: np.ndarray, model_name: str, cmap: str = "Blues") -> plt.Figure:
    """Generates confusion matrix heatmap plot for a trained classifier."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=11, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_col = "Rotational speed [rpm]" if "Rotational speed [rpm]" in X.columns else X.columns[0]
    y_col = "Torque [Nm]" if "Torque [Nm]" in X.columns else X.columns[1]

    scatter = ax.scatter(X[x_col], X[y_col], c=clusters, cmap="viridis", alpha=0.5, edgecolors="none")
    ax.set_title("K-Means Clustering: Torque vs Rotational Speed", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{x_col} (Scaled)", fontsize=10)
    ax.set_ylabel(f"{y_col} (Scaled)", fontsize=10)
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

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title(f"Hierarchical Clustering Dendrogram (Sample of {n} points)", fontsize=11, fontweight="bold")
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax
    )
    ax.set_xlabel("Cluster Size", fontsize=10)
    ax.set_ylabel("Distance", fontsize=10)
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
