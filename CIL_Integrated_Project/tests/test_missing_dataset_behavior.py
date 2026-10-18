"""
Unit tests verifying error handling when dataset or model artifacts are missing.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import eda, classical_ml


def test_eda_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError) as exc_info:
        eda.load_dataset(filepath="definitely_non_existent_dataset_12345.csv")
    assert "File not found" in str(exc_info.value)


def test_classical_ml_load_trained_models_missing():
    # Should safely return None instead of raising unhandled exception
    res = classical_ml.load_trained_models(path=Path("non_existent_models_123.joblib"))
    assert res is None
