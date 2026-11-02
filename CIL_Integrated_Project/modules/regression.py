"""
Task 6 Module: Tool Wear Regression Analysis
Original source: CIL456.ipynb

Implements:
- Feature extraction with 'Tool wear [min]' as continuous target.
- Linear Regression training and prediction.
- Evaluation metrics: MSE, RMSE, MAE, R-squared score.
- Actual vs. Predicted scatter visualization and residual analysis.
- Engineering interpretation of linear wear modeling.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; required for headless/test environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config


def prepare_regression_data(
    X_encoded: pd.DataFrame,
    target_col: str = config.REGRESSION_TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts features and target for Tool Wear regression.
    Direct reproduction of CIL456.ipynb Cell 24.
    """
    if target_col not in X_encoded.columns:
        raise KeyError(f"Target column '{target_col}' not found in provided features.")

    X_reg = X_encoded.drop(target_col, axis=1)
    y_reg = X_encoded[target_col]
    return X_reg, y_reg


def train_and_evaluate_regression(
    X_reg: pd.DataFrame,
    y_reg: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Dict[str, Any]:
    """
    Splits regression data, trains LinearRegression, and calculates evaluation metrics.
    """
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=random_state
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_reg, y_train_reg)

    y_pred_reg = lin_reg.predict(X_test_reg)

    mse = float(mean_squared_error(y_test_reg, y_pred_reg))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test_reg, y_pred_reg))
    r2 = float(r2_score(y_test_reg, y_pred_reg))

    return {
        "model": lin_reg,
        "X_train": X_train_reg,
        "X_test": X_test_reg,
        "y_train": y_train_reg,
        "y_test": y_test_reg,
        "y_pred": y_pred_reg,
        "metrics": {
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Absolute Error (MAE)": mae,
            "R2 Score": r2
        },
        "coefficients": dict(zip(X_reg.columns, lin_reg.coef_)),
        "intercept": float(lin_reg.intercept_)
    }


def plot_regression_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    title: str = "Linear Regression: Actual vs Predicted Tool Wear"
) -> plt.Figure:
    """
    Generates scatter plot of Actual vs Predicted Tool Wear with reference diagonal.
    Direct reproduction of CIL456.ipynb Cell 24.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(y_test, y_pred, alpha=0.35, color="#1D3557", edgecolors="none")

    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y=x)")

    ax.set_xlabel("Actual Tool Wear (Scaled)", fontsize=10)
    ax.set_ylabel("Predicted Tool Wear (Scaled)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    """Plots residual distribution for regression diagnostics."""
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, ax=ax, color="#E63946", alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", lw=1.5)
    ax.set_title("Residual Distribution (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def get_engineering_interpretation() -> str:
    """Returns the key analytical conclusion for Task 6."""
    return (
        "Engineering Conclusion from Task 6:\n"
        "The Linear Regression model achieves an R2 score near 0.00 (and slightly negative on the test split), "
        "with MSE ≈ 1.05. This demonstrates that tool wear cannot be accurately predicted solely as a static, "
        "linear combination of instantaneous sensor measurements (such as temperature, torque, and rotational speed). "
        "In physical machining operations, tool wear is an accumulating, non-linear degradation process governed by "
        "material fatigue, time-in-cut, and operational duty cycles, requiring non-linear models or temporal tracking."
    )
