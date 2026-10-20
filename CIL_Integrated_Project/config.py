"""
Configuration and Path Management for CIL Integrated Project (Tasks 3-8)
"""
import os
from pathlib import Path
from typing import Optional, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Expected Dataset Filename
DATASET_FILENAME = "ai4i2020.csv"
PROCESSED_FILENAME = "cleaned_predictive_maintenance.csv"

# Metadata & Saved Model Artifacts
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
CLASSICAL_MODELS_PATH = MODELS_DIR / "classical_models.joblib"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.joblib"
TEST_SET_SPLIT_PATH = MODELS_DIR / "test_set_split.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "assessment_history.json"

# Decision & Classification Constants
DEFAULT_OPERATING_THRESHOLD = 0.35
CRITICAL_PROBABILITY_THRESHOLD = 0.70
RISK_TIERS = ["NOMINAL", "MONITORING", "ELEVATED", "CRITICAL"]

# Feature Definitions
TARGET_COL = "Machine failure"
REGRESSION_TARGET_COL = "Tool wear [min]"

# Identifier and Failure Mode columns that must be excluded from feature space
EXCLUDE_COLS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

# Physical Numerical Features
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

CATEGORICAL_FEATURES = ["Type"]

# Feature ordering for Classical ML (Task 5: 6 features with LabelEncoded Type)
FEATURE_COLS_LABEL_ENCODED = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Feature ordering for Deep Learning / CNN (Task 7 & 8: 7 features with One-Hot Type)
FEATURE_COLS_ONEHOT = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M"
]

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2


def locate_dataset() -> Tuple[bool, Optional[Path]]:
    """
    Search for ai4i2020.csv in known standard locations:
    1. data/raw/ai4i2020.csv
    2. data/ai4i2020.csv
    3. project root / ai4i2020.csv
    4. parent workspace / ai4i2020.csv
    """
    candidate_paths = [
        RAW_DATA_DIR / DATASET_FILENAME,
        DATA_DIR / DATASET_FILENAME,
        PROJECT_ROOT / DATASET_FILENAME,
        PROJECT_ROOT.parent / DATASET_FILENAME,
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            return True, p
    return False, None


def locate_production_model() -> Tuple[bool, Optional[Path]]:
    """
    Search for production_model.joblib in known standard locations.
    """
    candidate_paths = [
        PRODUCTION_MODEL_PATH,
        MODELS_DIR / "production_model.joblib",
        PROJECT_ROOT / "artifacts" / "models" / "production_model.joblib",
        PROJECT_ROOT.parent / "CIL_Integrated_Project" / "artifacts" / "models" / "production_model.joblib"
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            return True, p
    return False, None


def get_processed_data_path() -> Path:
    """Return destination path for cleaned/processed dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_DIR / PROCESSED_FILENAME

