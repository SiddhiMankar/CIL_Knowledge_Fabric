"""
Unit tests for data contracts, transformation pipelines, and model interfaces.
Uses a minimal, explicitly synthetic test fixture to verify function signatures.
NOTE: These synthetic fixtures verify code contracts only; they do NOT validate
or represent the real predictive maintenance dataset.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import eda, classical_ml, regression, cnn_model


@pytest.fixture
def synthetic_test_fixture() -> pd.DataFrame:
    """Minimal synthetic fixture for unit-testing interface contracts."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "UDI": list(range(1, n + 1)),
        "Product ID": [f"M{1000 + i}" for i in range(n)],
        "Type": np.random.choice(["L", "M", "H"], size=n),
        "Air temperature [K]": np.random.normal(300, 2, size=n),
        "Process temperature [K]": np.random.normal(310, 1.5, size=n),
        "Rotational speed [rpm]": np.random.normal(1500, 100, size=n),
        "Torque [Nm]": np.random.normal(40, 10, size=n),
        "Tool wear [min]": np.random.uniform(0, 200, size=n),
        "Machine failure": [0] * 25 + [1] * 5,  # Imbalanced failure target
        "TWF": [0] * n,
        "HDF": [0] * n,
        "PWF": [0] * n,
        "OSF": [0] * n,
        "RNF": [0] * n,
    })


def test_eda_summary_contract(synthetic_test_fixture):
    summary = eda.get_dataset_summary(synthetic_test_fixture)
    assert summary["rows"] == 30
    assert summary["columns"] == 14
    assert summary["total_missing"] == 0


def test_eda_preprocessing_contract(synthetic_test_fixture):
    cleaned_df, scaler = eda.preprocess_and_scale(synthetic_test_fixture)
    assert "UDI" not in cleaned_df.columns
    assert "Product ID" not in cleaned_df.columns
    # Check that scaled features have mean approx 0
    assert abs(cleaned_df["Torque [Nm]"].mean()) < 1e-6


def test_classical_ml_pipeline_contract(synthetic_test_fixture):
    cleaned_df, _ = eda.preprocess_and_scale(synthetic_test_fixture)
    X, y, le = classical_ml.prepare_classification_data(cleaned_df)
    assert X.shape[1] == 6
    assert len(y) == len(X)

    imp_df, fig = classical_ml.compute_feature_importance(X, y)
    assert len(imp_df) == 6
    assert fig is not None

    X_train, X_test, y_train, y_test = classical_ml.split_data(X, y, test_size=0.3, stratify=False)
    comp_df, results = classical_ml.train_and_evaluate_all_classifiers(X_train, y_train, X_test, y_test)
    assert len(comp_df) == 4
    assert "Random Forest" in results


def test_regression_pipeline_contract(synthetic_test_fixture):
    cleaned_df, _ = eda.preprocess_and_scale(synthetic_test_fixture)
    X, _, _ = classical_ml.prepare_classification_data(cleaned_df)
    X_reg, y_reg = regression.prepare_regression_data(X)
    assert "Tool wear [min]" not in X_reg.columns

    res = regression.train_and_evaluate_regression(X_reg, y_reg, test_size=0.3)
    assert "Mean Squared Error (MSE)" in res["metrics"]
    assert "R2 Score" in res["metrics"]


def test_cnn_prep_and_pytorch_contract(synthetic_test_fixture):
    prep = cnn_model.prepare_cnn_dataset(synthetic_test_fixture, test_size=0.3)
    assert prep["num_features"] == 7
    assert prep["X_train_cnn"].shape[1] == 7
    assert prep["X_train_cnn"].shape[2] == 1

    if cnn_model.HAS_TORCH:
        model = cnn_model.PyTorch1DCNN(in_features=7)
        import torch
        x_dummy = torch.randn(4, 7, 1)
        out = model(x_dummy)
        assert out.shape == (4, 1)
