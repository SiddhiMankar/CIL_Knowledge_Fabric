"""
Task 4 Module: Exploratory Data Analysis & Data Preprocessing
Original source: CIL456.ipynb

Implements:
- Robust dataset loading and missing-dataset diagnostics.
- Statistical summary, data types, missing value and duplicate checks.
- Outlier and distribution visualizations (Boxplots, Histograms).
- Correlation matrix computation and visualization.
- Standardized cleaning and scaling pipeline using StandardScaler.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; required for headless/test environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import config

COLS_TO_PLOT = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the predictive maintenance dataset (ai4i2020.csv).
    If no path is provided, searches standard candidate directories via config.locate_dataset().
    Raises FileNotFoundError with clear instructions if dataset is missing.
    """
    if filepath is None:
        found, detected_path = config.locate_dataset()
        if not found or detected_path is None:
            raise FileNotFoundError(
                f"Dataset '{config.DATASET_FILENAME}' was not found in expected locations:\n"
                f" - {config.RAW_DATA_DIR / config.DATASET_FILENAME}\n"
                f" - {config.PROJECT_ROOT / config.DATASET_FILENAME}\n"
                "Please place 'ai4i2020.csv' in the project directory or 'data/raw/' to proceed."
            )
        filepath = detected_path

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found at specified path: {filepath}")

    df = pd.read_csv(filepath)
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary information: shape, missing values, duplicates,
    and column types breakdown.
    """
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicates_count = int(df.duplicated().sum())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_missing": total_missing,
        "missing_per_column": missing_counts,
        "duplicate_rows": duplicates_count,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "columns_list": df.columns.tolist()
    }


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Returns numerical summary statistics (.describe())."""
    return df.describe()


def plot_boxplots(df: pd.DataFrame, cols: Optional[List[str]] = None) -> plt.Figure:
    """
    Generates boxplots for continuous features to identify outliers.
    Direct reproduction of CIL456.ipynb Cell 8.
    """
    cols_to_use = cols or [c for c in COLS_TO_PLOT if c in df.columns]
    n_cols = len(cols_to_use)
    if n_cols == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No matching numerical columns found.", ha="center", va="center")
        return fig

    rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols_to_use):
        sns.boxplot(y=df[col], ax=axes[i], color="#4A90E2")
        axes[i].set_title(f"Boxplot of {col}", fontsize=11, fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, cols: Optional[List[str]] = None) -> plt.Figure:
    """
    Generates KDE histogram distributions for physical measurement features.
    Direct reproduction of CIL456.ipynb Cell 9.
    """
    cols_to_use = cols or [c for c in COLS_TO_PLOT if c in df.columns]
    n_cols = len(cols_to_use)
    if n_cols == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No matching numerical columns found.", ha="center", va="center")
        return fig

    rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols_to_use):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#00A896", edgecolor="white", alpha=0.6)
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Plots correlation matrix across numerical features.
    Direct reproduction of CIL456.ipynb Cell 10.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    corr = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def preprocess_and_scale(
    df: pd.DataFrame,
    features_to_scale: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Performs data cleaning and scaling:
    1. Creates a copy.
    2. Drops identifier columns (UDI, Product ID) if present.
    3. Fits and transforms numerical features with StandardScaler.
    Returns: (cleaned_df, fitted_scaler)
    """
    df_cleaned = df.copy()

    # Drop identifier columns
    cols_to_drop = [c for c in ["UDI", "Product ID"] if c in df_cleaned.columns]
    if cols_to_drop:
        df_cleaned = df_cleaned.drop(cols_to_drop, axis=1)

    scale_cols = features_to_scale or [c for c in COLS_TO_PLOT if c in df_cleaned.columns]

    scaler = StandardScaler()
    if scale_cols:
        df_cleaned[scale_cols] = scaler.fit_transform(df_cleaned[scale_cols])

    return df_cleaned, scaler


def save_cleaned_dataset(df_cleaned: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Saves cleaned DataFrame to processed data directory."""
    target_path = output_path or config.get_processed_data_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(target_path, index=False)
    return target_path
